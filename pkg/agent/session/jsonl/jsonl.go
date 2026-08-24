// Package jsonl stores sessions on the filesystem: a directory per session,
// metadata in one file and entries in another, one JSON object per line.
package jsonl

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent/session"
)

const (
	metaFile    = "meta.json"
	entriesFile = "entries.jsonl"
)

// Store is a session store rooted at a directory.
type Store struct {
	root string

	mu sync.Mutex // serializes append and metadata writes
}

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonl: creating %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

var _ session.Store = (*Store)(nil)

// Create starts a session, assigning an id when none was given.
func (s *Store) Create(_ context.Context, meta session.Meta) (session.Meta, error) {
	if meta.ID == "" {
		meta.ID = newID()
	}
	now := time.Now().UTC()
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.UpdatedAt = now
	meta.Entries = 0

	s.mu.Lock()
	defer s.mu.Unlock()

	dir := s.dir(meta.ID)
	if _, err := os.Stat(dir); err == nil {
		return session.Meta{}, fmt.Errorf("jsonl: session %s already exists", meta.ID)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return session.Meta{}, err
	}
	if err := s.writeMeta(meta); err != nil {
		return session.Meta{}, err
	}
	return meta, nil
}

// Append writes entries, assigning each a sequence number.
func (s *Store) Append(_ context.Context, id string, entries ...session.Entry) error {
	if len(entries) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	meta, err := s.readMeta(id)
	if err != nil {
		return err
	}

	f, err := os.OpenFile(filepath.Join(s.dir(id), entriesFile), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, e := range entries {
		meta.Entries++
		if e.Seq == 0 {
			e.Seq = meta.Entries
		}
		if e.At.IsZero() {
			e.At = time.Now().UTC()
		}
		line, err := json.Marshal(e)
		if err != nil {
			return fmt.Errorf("jsonl: encoding entry %d: %w", e.Seq, err)
		}
		w.Write(line)
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		return err
	}

	meta.UpdatedAt = time.Now().UTC()
	return s.writeMeta(meta)
}

// Entries reads a session from the beginning.
func (s *Store) Entries(_ context.Context, id string) iter.Seq2[session.Entry, error] {
	return func(yield func(session.Entry, error) bool) {
		f, err := os.Open(filepath.Join(s.dir(id), entriesFile))
		if errors.Is(err, fs.ErrNotExist) {
			if _, statErr := os.Stat(s.dir(id)); errors.Is(statErr, fs.ErrNotExist) {
				yield(session.Entry{}, fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound))
			}
			return // an existing session with nothing in it yet
		}
		if err != nil {
			yield(session.Entry{}, err)
			return
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var e session.Entry
			if err := json.Unmarshal(line, &e); err != nil {
				return // a torn tail ends the session, it does not fail it
			}
			if !yield(e, nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(session.Entry{}, err)
		}
	}
}

// Meta reads one session's metadata.
func (s *Store) Meta(_ context.Context, id string) (session.Meta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readMeta(id)
}

// SetMeta replaces the caller-owned half of a session's metadata.
func (s *Store) SetMeta(_ context.Context, meta session.Meta) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, err := s.readMeta(meta.ID)
	if err != nil {
		return err
	}
	// Entries and CreatedAt belong to the store: a caller that echoes back a
	// stale Meta must not be able to rewrite history's length.
	meta.Entries = current.Entries
	meta.CreatedAt = current.CreatedAt
	meta.UpdatedAt = time.Now().UTC()
	return s.writeMeta(meta)
}

// List returns every session, most recently updated first.
func (s *Store) List(_ context.Context) ([]session.Meta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	dirs, err := os.ReadDir(s.root)
	if err != nil {
		return nil, err
	}
	out := make([]session.Meta, 0, len(dirs))
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		meta, err := s.readMeta(d.Name())
		if err != nil {
			continue // a directory that is not a session, or one being written
		}
		out = append(out, meta)
	}
	slices.SortFunc(out, func(a, b session.Meta) int { return b.UpdatedAt.Compare(a.UpdatedAt) })
	return out, nil
}

// Fork copies entries up to upto into a new session.
func (s *Store) Fork(ctx context.Context, id string, upto int64) (session.Meta, error) {
	source, err := s.Meta(ctx, id)
	if err != nil {
		return session.Meta{}, err
	}

	var kept []session.Entry
	for e, err := range s.Entries(ctx, id) {
		if err != nil {
			return session.Meta{}, err
		}
		if upto > 0 && e.Seq > upto {
			break
		}
		e.Seq = 0 // the new session numbers its own entries
		kept = append(kept, e)
	}

	forked, err := s.Create(ctx, session.Meta{
		Title:    source.Title,
		Model:    source.Model,
		Parent:   source.ID,
		ForkedAt: upto,
	})
	if err != nil {
		return session.Meta{}, err
	}
	if err := s.Append(ctx, forked.ID, kept...); err != nil {
		return session.Meta{}, err
	}
	return s.Meta(ctx, forked.ID)
}

// Delete removes a session and everything in it.
func (s *Store) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return os.RemoveAll(s.dir(id))
}

// Close releases nothing: this store holds no handle between calls, which is
// what makes it safe to have two processes appending to different sessions
// under one root.
func (s *Store) Close() error { return nil }

func (s *Store) dir(id string) string { return filepath.Join(s.root, id) }

func (s *Store) readMeta(id string) (session.Meta, error) {
	raw, err := os.ReadFile(filepath.Join(s.dir(id), metaFile))
	if errors.Is(err, fs.ErrNotExist) {
		return session.Meta{}, fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound)
	}
	if err != nil {
		return session.Meta{}, err
	}
	var meta session.Meta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return session.Meta{}, fmt.Errorf("jsonl: reading %s metadata: %w", id, err)
	}
	return meta, nil
}

// writeMeta replaces metadata by writing beside it and renaming, so a crash
// leaves the previous version rather than a truncated one.
func (s *Store) writeMeta(meta session.Meta) error {
	raw, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	dir := s.dir(meta.ID)
	tmp, err := os.CreateTemp(dir, "meta-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), filepath.Join(dir, metaFile))
}

func newID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return time.Now().UTC().Format("20060102-150405") + "-" + hex.EncodeToString(b[:4])
}
