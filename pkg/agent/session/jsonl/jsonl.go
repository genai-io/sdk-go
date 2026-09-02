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
	"strings"
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

// openSession is what a session costs to keep appending to: where it lives,
// the entries file held open, and the sequence number of the last entry
// written to it.
//
// meta.json is deliberately not on this path: Entries and UpdatedAt only cache
// what the entries file already knows, and rewriting a file atomically once
// per recorded event cost more than everything else here put together.
type openSession struct {
	dir     string // validated when the session was opened
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
	meta = meta.Created(time.Now().UTC())

	s.mu.Lock()
	defer s.mu.Unlock()

	dir, err := s.dir(meta.ID)
	if err != nil {
		return session.Meta{}, err
	}
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
		e = e.Stamped(open.seq, now)
		line, err := json.Marshal(e)
		if err != nil {
			return fmt.Errorf("jsonl: encoding entry %d: %w", e.Seq, err)
		}
		// bufio keeps the first write error and Flush below is where it is
		// read, so checking each write here would report it twice.
		_, _ = w.Write(line)
		_ = w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		return err
	}
	open.updated = now
	open.unsaved += int64(len(entries))

	// Another process listing sessions reads meta.json, so it cannot be left
	// behind forever — but it can be left behind for a while.
	if open.unsaved >= metaEvery {
		return open.save()
	}
	return nil
}

// metaEvery is how many entries may pass before meta.json is brought up to
// date anyway. Anyone reading it through this store gets it saved first; this
// bounds how stale it looks to a process reading the file directly.
const metaEvery = 64

// Entries reads a session from the beginning. A cancelled context ends the
// read with its error rather than quietly, because a read that stopped early
// and said nothing is indistinguishable from a shorter session.
func (s *Store) Entries(ctx context.Context, id string) iter.Seq2[session.Entry, error] {
	return func(yield func(session.Entry, error) bool) {
		dir, err := s.dir(id)
		if err != nil {
			yield(session.Entry{}, err)
			return
		}
		f, err := os.Open(filepath.Join(dir, entriesFile))
		if errors.Is(err, fs.ErrNotExist) {
			if _, statErr := os.Stat(dir); errors.Is(statErr, fs.ErrNotExist) {
				yield(session.Entry{}, fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound))
			}
			return // an existing session with nothing in it yet
		}
		if err != nil {
			yield(session.Entry{}, err)
			return
		}
		defer func() { _ = f.Close() }() // read-only: nothing is lost by closing it

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
		// A line that does not parse is held rather than acted on until the
		// next one says which it was. As the last line it is a process killed
		// mid-append and the session ends there, one entry short; with
		// anything after it, it is a hole in the middle, and a conversation
		// with a hole in it is worse than a short one.
		var torn error
		n := 0
		for scanner.Scan() {
			n++
			if err := ctx.Err(); err != nil {
				yield(session.Entry{}, err)
				return
			}
			if torn != nil {
				yield(session.Entry{}, torn)
				return
			}
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var e session.Entry
			if err := json.Unmarshal(line, &e); err != nil {
				torn = fmt.Errorf("jsonl: %s: line %d does not parse: %w", id, n, err)
				continue
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
	dir, err := s.dir(meta.ID)
	if err != nil {
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
	return writeMeta(dir, meta)
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
	slices.SortFunc(out, session.ByRecency)
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

// Delete removes a session and everything in it. Deleting one that is being
// appended to is a race the caller has to settle: the appender can recreate the
// entries file inside the directory being removed, and this then fails saying
// so rather than pretending the session is gone.
func (s *Store) Delete(_ context.Context, id string) error {
	dir, err := s.dir(id)
	if err != nil {
		return err
	}

	s.mu.Lock()
	live, ok := s.live[id]
	delete(s.live, id)
	s.mu.Unlock()
	if ok {
		live.mu.Lock()
		_ = live.file.Close() // the directory holding it is removed below
		live.mu.Unlock()
	}
	return os.RemoveAll(dir)
}

// Close saves what was appended and releases the files. Not closing loses no
// entry: Append hands each one to the operating system, which serves it to
// every other reader whether or not this store closed. What is lost is how up
// to date meta.json looks from outside — and, since nothing here fsyncs, the
// tail of a session on a machine that loses power.
func (s *Store) Close() error {
	s.mu.Lock()
	live := s.live
	s.live = map[string]*openSession{}
	s.mu.Unlock()

	var firstErr error
	for _, o := range live {
		if err := o.close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// dir is where a session lives. The id is checked rather than joined straight
// in: ids come from application input, and Join would let "" name the store
// itself and ".." name a directory beside it — either of which Delete would
// then remove.
func (s *Store) dir(id string) (string, error) {
	if id == "" || id == "." || id == ".." || strings.ContainsAny(id, `/\`) {
		return "", fmt.Errorf("jsonl: %q is not a session id", id)
	}
	return filepath.Join(s.root, id), nil
}

// open returns the session's live state, opening the entries file and
// recovering its sequence number the first time it is asked for.
func (s *Store) open(id string) (*openSession, error) {
	s.mu.Lock()
	if live, ok := s.live[id]; ok {
		s.mu.Unlock()
		return live, nil
	}
	s.mu.Unlock()

	dir, err := s.dir(id)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(filepath.Join(dir, metaFile)); errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound)
	}
	path := filepath.Join(dir, entriesFile)
	if err := trimTornTail(path); err != nil {
		return nil, err
	}
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
		_ = f.Close() // nothing was written through it
		return live, nil
	}
	live := &openSession{dir: dir, file: f, seq: seq}
	s.live[id] = live
	return live, nil
}

// save brings meta.json up to date with what has been appended. The caller
// holds the session's lock.
func (o *openSession) save() error {
	if o.unsaved == 0 {
		return nil
	}
	raw, err := os.ReadFile(filepath.Join(o.dir, metaFile))
	if errors.Is(err, fs.ErrNotExist) {
		// The session was deleted under whoever is still appending to it: there
		// is no metadata to bring up to date. Failing here would fail every
		// Meta and List on the store, for every other session too.
		o.unsaved = 0
		return nil
	}
	if err != nil {
		return err
	}
	var meta session.Meta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return fmt.Errorf("jsonl: reading %s metadata: %w", filepath.Base(o.dir), err)
	}
	meta.Entries, meta.UpdatedAt = o.seq, o.updated
	if err := writeMeta(o.dir, meta); err != nil {
		return err
	}
	o.unsaved = 0
	return nil
}

func (o *openSession) close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	err := o.save()
	if closeErr := o.file.Close(); err == nil {
		err = closeErr
	}
	return err
}

// saveAll brings every open session's metadata up to date, so whoever is about
// to read it reads the truth.
func (s *Store) saveAll() error {
	s.mu.Lock()
	live := make([]*openSession, 0, len(s.live))
	for _, o := range s.live {
		live = append(live, o)
	}
	s.mu.Unlock()

	var firstErr error
	for _, o := range live {
		o.mu.Lock()
		err := o.save()
		o.mu.Unlock()
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// trimTornTail drops a half-written last line, so that the next entry begins a
// line of its own rather than joining the wreckage of the one a killed process
// left. That entry is lost either way; this is what stops it taking the next
// one with it.
func trimTornTail(path string) (err error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	// This function truncates, so a failure to close is a failure to truncate
	// on any filesystem that reports one late.
	defer func() { err = errors.Join(err, f.Close()) }()

	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return err
	}
	var last [1]byte
	if _, err := f.ReadAt(last[:], size-1); err != nil {
		return err
	}
	if last[0] == '\n' {
		return nil
	}
	for window := int64(8 * 1024); ; window *= 4 {
		if window > size {
			window = size
		}
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return err
		}
		if i := bytes.LastIndexByte(buf, '\n'); i >= 0 {
			return f.Truncate(size - window + int64(i) + 1)
		}
		if window == size {
			return f.Truncate(0) // nothing was ever written whole
		}
	}
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
	defer func() { _ = f.Close() }() // read-only: nothing is lost by closing it

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
		// Back over whole lines until one parses: numbering from where the
		// readable part ended is the point, and a line that says nothing is
		// not where it ended.
		trimmed := bytes.TrimRight(buf, "\n")
		for {
			i := bytes.LastIndexByte(trimmed, '\n')
			if i < 0 {
				break
			}
			if seq, ok := seqOf(trimmed[i+1:]); ok {
				return seq, nil
			}
			trimmed = trimmed[:i]
		}
		if window == size { // the whole file is one line, or none
			seq, _ := seqOf(trimmed)
			return seq, nil
		}
	}
}

func seqOf(line []byte) (int64, bool) {
	var e struct {
		Seq int64 `json:"seq"`
	}
	if json.Unmarshal(line, &e) != nil {
		return 0, false
	}
	return e.Seq, true
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
