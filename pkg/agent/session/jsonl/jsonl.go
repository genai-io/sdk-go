// Package jsonl stores sessions on the filesystem: a directory per session,
// metadata in one file and entries in another, one JSON object per line.
package jsonl

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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

	mu   sync.Mutex // guards live, and nothing else
	live map[string]*openSession
}

// openSession is what a session costs to keep appending to: the entries file
// held open, and the sequence number of the last entry written to it.
//
// meta.json is deliberately not on this path. Entries and UpdatedAt are a
// cache of what the entries file already knows — how many lines it has, and
// when it was last written — and rewriting a file atomically to maintain a
// cache, once per recorded event, cost more than everything else here put
// together. It is brought up to date when somebody reads it instead.
type openSession struct {
	mu      sync.Mutex
	file    *os.File
	seq     int64
	updated time.Time
	unsaved int64 // entries appended since meta.json last agreed
}

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonl: creating %s: %w", dir, err)
	}
	return &Store{root: dir, live: map[string]*openSession{}}, nil
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
	if err := writeMeta(dir, meta); err != nil {
		return session.Meta{}, err
	}
	return meta, nil
}

// Append writes entries, assigning each a sequence number.
func (s *Store) Append(_ context.Context, id string, entries ...session.Entry) error {
	if len(entries) == 0 {
		return nil
	}

	open, err := s.open(id)
	if err != nil {
		return err
	}

	open.mu.Lock()
	defer open.mu.Unlock()

	w := bufio.NewWriter(open.file)
	now := time.Now().UTC()
	for _, e := range entries {
		open.seq++
		if e.Seq == 0 {
			e.Seq = open.seq
		}
		if e.At.IsZero() {
			e.At = now
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
	open.updated = now
	open.unsaved += int64(len(entries))

	// Another process listing sessions reads meta.json, so it cannot be left
	// behind forever — but it can be left behind for a while.
	if open.unsaved >= metaEvery {
		return open.save(s.dir(id))
	}
	return nil
}

// metaEvery is how many entries may pass before meta.json is brought up to
// date anyway. Anyone reading it through this store gets it saved first; this
// bounds how stale it looks to a process reading the file directly.
const metaEvery = 64

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
	if err := s.saveAll(); err != nil {
		return session.Meta{}, err
	}
	return s.readMeta(id)
}

// SetMeta replaces the caller-owned half of a session's metadata.
func (s *Store) SetMeta(_ context.Context, meta session.Meta) error {
	if err := s.saveAll(); err != nil {
		return err
	}
	current, err := s.readMeta(meta.ID)
	if err != nil {
		return err
	}
	// Entries and CreatedAt belong to the store: a caller that echoes back a
	// stale Meta must not be able to rewrite history's length.
	meta.Entries = current.Entries
	meta.CreatedAt = current.CreatedAt
	meta.UpdatedAt = time.Now().UTC()
	return writeMeta(s.dir(meta.ID), meta)
}

// List returns every session, most recently updated first.
func (s *Store) List(_ context.Context) ([]session.Meta, error) {
	if err := s.saveAll(); err != nil {
		return nil, err
	}
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
	live, ok := s.live[id]
	delete(s.live, id)
	s.mu.Unlock()
	if ok {
		live.mu.Lock()
		live.file.Close()
		live.mu.Unlock()
	}
	return os.RemoveAll(s.dir(id))
}

// Close saves what was appended and releases the files. Not closing loses
// nothing an entries file holds — those are on disk the moment Append returns
// — only how up to date meta.json looks to somebody reading it from elsewhere.
func (s *Store) Close() error {
	s.mu.Lock()
	live := s.live
	s.live = map[string]*openSession{}
	s.mu.Unlock()

	var firstErr error
	for id, o := range live {
		if err := o.close(s.dir(id)); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (s *Store) dir(id string) string { return filepath.Join(s.root, id) }

// open returns the session's live state, opening the entries file and
// recovering its sequence number the first time it is asked for.
func (s *Store) open(id string) (*openSession, error) {
	s.mu.Lock()
	if live, ok := s.live[id]; ok {
		s.mu.Unlock()
		return live, nil
	}
	s.mu.Unlock()

	dir := s.dir(id)
	if _, err := os.Stat(filepath.Join(dir, metaFile)); errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound)
	}
	path := filepath.Join(dir, entriesFile)
	seq, err := lastSeq(path)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if live, ok := s.live[id]; ok { // somebody opened it while this one read
		f.Close()
		return live, nil
	}
	live := &openSession{file: f, seq: seq}
	s.live[id] = live
	return live, nil
}

// save brings meta.json up to date with what has been appended. The caller
// holds the session's lock.
func (o *openSession) save(dir string) error {
	if o.unsaved == 0 {
		return nil
	}
	raw, err := os.ReadFile(filepath.Join(dir, metaFile))
	if err != nil {
		return err
	}
	var meta session.Meta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return fmt.Errorf("jsonl: reading %s metadata: %w", filepath.Base(dir), err)
	}
	meta.Entries, meta.UpdatedAt = o.seq, o.updated
	if err := writeMeta(dir, meta); err != nil {
		return err
	}
	o.unsaved = 0
	return nil
}

func (o *openSession) close(dir string) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	err := o.save(dir)
	if closeErr := o.file.Close(); err == nil {
		err = closeErr
	}
	return err
}

// saveAll brings every open session's metadata up to date, so whoever is about
// to read it reads the truth.
func (s *Store) saveAll() error {
	s.mu.Lock()
	ids := make([]string, 0, len(s.live))
	live := make([]*openSession, 0, len(s.live))
	for id, o := range s.live {
		ids = append(ids, id)
		live = append(live, o)
	}
	s.mu.Unlock()

	var firstErr error
	for i, o := range live {
		o.mu.Lock()
		err := o.save(s.dir(ids[i]))
		o.mu.Unlock()
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// lastSeq recovers the number to carry on from by reading the end of the
// entries file. The file is the record, so this is the one count that cannot
// be wrong — meta.json can be behind, and after a crash it will be. It reads a
// block from the tail rather than the whole file, growing it only for an entry
// too long to fit.
func lastSeq(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	defer f.Close()

	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return 0, err
	}
	for window := int64(8 * 1024); ; window *= 4 {
		if window > size {
			window = size
		}
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return 0, err
		}
		trimmed := bytes.TrimRight(buf, "\n")
		if i := bytes.LastIndexByte(trimmed, '\n'); i >= 0 {
			return seqOf(trimmed[i+1:]), nil
		}
		if window == size {
			return seqOf(trimmed), nil // the whole file is one line, or none
		}
	}
}

func seqOf(line []byte) int64 {
	var e struct {
		Seq int64 `json:"seq"`
	}
	if json.Unmarshal(line, &e) != nil {
		return 0 // a torn tail numbers from where the readable part ended
	}
	return e.Seq
}

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
func writeMeta(dir string, meta session.Meta) error {
	raw, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
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
