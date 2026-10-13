// Package jsonl stores sessions on the filesystem: a directory per session,
// metadata in one file and entries in another, one JSON object per line.
package jsonl

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/session"
)

const (
	metaFile    = "meta.json"
	entriesFile = "entries.jsonl"
)

// Store is a session store rooted at a directory.
type Store struct{ root string }

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonl: creating %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

var _ session.Store = (*Store)(nil)

// Create starts a session, assigning an id when none was given.
func (s *Store) Create(ctx context.Context, meta session.Meta) (session.Meta, error) {
	if meta.ID == "" {
		meta.ID = newID()
	}
	meta = meta.Created(time.Now().UTC())

	dir, err := s.dir(meta.ID)
	if err != nil {
		return session.Meta{}, err
	}
	if err := ctx.Err(); err != nil {
		return session.Meta{}, err
	}
	if err := os.Mkdir(dir, 0o700); err != nil {
		return session.Meta{}, err
	}
	if err := writeMeta(dir, meta); err != nil {
		_ = os.RemoveAll(dir)
		return session.Meta{}, err
	}
	return meta, nil
}

// Append writes entries, assigning each a sequence number.
func (s *Store) Append(ctx context.Context, id string, entries ...session.Entry) (err error) {
	if len(entries) == 0 {
		return ctx.Err()
	}
	dir, err := s.dir(id)
	if err != nil {
		return err
	}
	if _, err = s.readMeta(id); err != nil {
		return err
	}
	lock, err := lockFile(ctx, filepath.Join(dir, ".lock"), true)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	if err = ctx.Err(); err != nil {
		return err
	}
	path := filepath.Join(dir, entriesFile)
	if err = trimTornTail(path); err != nil {
		return err
	}
	position, err := lastPosition(path)
	if err != nil {
		return err
	}
	seq := position.Seq
	now := time.Now().UTC()
	var data bytes.Buffer
	for _, e := range entries {
		next := seq + 1
		if e.Seq != 0 && e.Seq != next {
			return fmt.Errorf("jsonl: sequence %d, want %d", e.Seq, next)
		}
		e = e.Stamped(next, now)
		line, err := json.Marshal(e)
		if err != nil {
			return fmt.Errorf("jsonl: encoding entry %d: %w", e.Seq, err)
		}
		data.Write(line)
		data.WriteByte('\n')
		seq = next
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(data.Bytes())
	if err = errors.Join(writeErr, f.Close()); err != nil {
		return err
	}
	return nil
}

// Meta reads one session's metadata.
func (s *Store) Meta(ctx context.Context, id string) (session.Meta, error) {
	if err := ctx.Err(); err != nil {
		return session.Meta{}, err
	}
	meta, err := s.readMeta(id)
	if err != nil {
		return session.Meta{}, err
	}
	dir, _ := s.dir(id)
	lock, err := lockFile(ctx, filepath.Join(dir, ".lock"), true)
	if err != nil {
		return session.Meta{}, err
	}
	defer func() { _ = lock.Close() }()
	path := filepath.Join(dir, entriesFile)
	position, err := lastPosition(path)
	if err != nil {
		return session.Meta{}, err
	}
	meta.Entries = position.Seq
	if !position.At.IsZero() {
		meta.UpdatedAt = position.At
	}

	return meta, nil
}

// List returns every session, most recently updated first.
func (s *Store) List(ctx context.Context) ([]session.Meta, error) {
	if err := ctx.Err(); err != nil {
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
		meta, err := s.Meta(ctx, d.Name())
		if err != nil {
			continue // a directory that is not a session, or one being written
		}
		out = append(out, meta)
	}
	slices.SortFunc(out, session.ByRecency)
	return out, nil
}

// dir is where a session lives. Ids come from application input, so Join alone
// would let "" name the store itself and ".." a neighbour.
func (s *Store) dir(id string) (string, error) {
	if id == "" || id == "." || id == ".." || strings.ContainsAny(id, `/\`) {
		return "", fmt.Errorf("jsonl: %q is not a session id", id)
	}
	return filepath.Join(s.root, id), nil
}

func (s *Store) readMeta(id string) (session.Meta, error) {
	dir, err := s.dir(id)
	if err != nil {
		return session.Meta{}, err
	}
	raw, err := os.ReadFile(filepath.Join(dir, metaFile))
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
	defer func() { _ = os.Remove(tmp.Name()) }() // already renamed away on success

	if _, err := tmp.Write(raw); err != nil {
		_ = tmp.Close() // the write error is the one worth reporting
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
