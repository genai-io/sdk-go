// Package jsonl stores sessions on the filesystem: a directory per session
// holding its entries, one JSON object per line.
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
	"time"

	"github.com/genai-io/sdk-go/pkg/session"
)

const (
	entriesFile = "entries.jsonl"
	lockFile    = ".lock"
)

// Store is a session store rooted at a directory.
type Store struct{ root string }

var _ session.Store = (*Store)(nil)

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonl: creating %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

// Create starts an empty session.
func (s *Store) Create(ctx context.Context) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	var b [4]byte
	_, _ = rand.Read(b[:])
	id := time.Now().UTC().Format("20060102-150405") + "-" + hex.EncodeToString(b[:])
	return id, os.Mkdir(filepath.Join(s.root, id), 0o700)
}

// Claim locks the session against every other writer, in this process or
// another, and repairs a final record a crash left half-written.
func (s *Store) Claim(ctx context.Context, id string) (func() error, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dir, err := s.dir(id)
	if err != nil {
		return nil, err
	}
	lock, err := os.OpenFile(filepath.Join(dir, lockFile), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := tryLock(lock); err != nil {
		_ = lock.Close()
		return nil, fmt.Errorf("jsonl: claiming session %s: %w", id, err)
	}
	if err := repairTail(filepath.Join(dir, entriesFile)); err != nil {
		_ = lock.Close()
		return nil, err
	}
	return lock.Close, nil
}

// Append writes one entry and syncs it to disk.
func (s *Store) Append(ctx context.Context, id string, entry session.Entry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	dir, err := s.dir(id)
	if err != nil {
		return err
	}
	line, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("jsonl: encoding entry: %w", err)
	}
	f, err := os.OpenFile(filepath.Join(dir, entriesFile), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	_, err = f.Write(append(line, '\n'))
	if err == nil {
		err = f.Sync()
	}
	return errors.Join(err, f.Close())
}

// Entries reads the log. An incomplete final record — a crash mid-write, or a
// writer still writing — ends it; a bad record anywhere else is an error.
func (s *Store) Entries(ctx context.Context, id string) iter.Seq2[session.Entry, error] {
	return func(yield func(session.Entry, error) bool) {
		dir, err := s.dir(id)
		if err != nil {
			yield(session.Entry{}, err)
			return
		}
		f, err := os.Open(filepath.Join(dir, entriesFile))
		if errors.Is(err, fs.ErrNotExist) {
			return
		}
		if err != nil {
			yield(session.Entry{}, err)
			return
		}
		defer func() { _ = f.Close() }() // read-only
		reader := bufio.NewReader(f)
		for n := 1; ; n++ {
			if err := ctx.Err(); err != nil {
				yield(session.Entry{}, err)
				return
			}
			line, readErr := reader.ReadBytes('\n')
			if readErr != nil && readErr != io.EOF {
				yield(session.Entry{}, readErr)
				return
			}
			if len(bytes.TrimSpace(line)) > 0 {
				var e session.Entry
				if err := json.Unmarshal(line, &e); err != nil {
					if readErr == io.EOF {
						return // torn
					}
					yield(session.Entry{}, fmt.Errorf("jsonl: %s: line %d does not parse: %w", id, n, err))
					return
				}
				if !yield(e, nil) {
					return
				}
			}
			if readErr == io.EOF {
				return
			}
		}
	}
}

// Meta is a session as List reports it.
type Meta struct {
	ID string
	// UpdatedAt is when the last entry was written, or when the session was
	// created if it has none.
	UpdatedAt time.Time
}

// List returns every session, most recently updated first.
func (s *Store) List(ctx context.Context) ([]Meta, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dirs, err := os.ReadDir(s.root)
	if err != nil {
		return nil, err
	}
	var out []Meta
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		info, err := d.Info()
		if err != nil {
			continue // removed while listing
		}
		at, err := lastAt(filepath.Join(s.root, d.Name(), entriesFile))
		if err != nil {
			return nil, err
		}
		if at.IsZero() {
			at = info.ModTime()
		}
		out = append(out, Meta{ID: d.Name(), UpdatedAt: at})
	}
	slices.SortFunc(out, func(a, b Meta) int { return b.UpdatedAt.Compare(a.UpdatedAt) })
	return out, nil
}

// dir is where an existing session lives. Ids come from application input,
// so Join alone would let "" name the store itself and ".." a neighbour.
func (s *Store) dir(id string) (string, error) {
	if id == "" || id == "." || id == ".." || strings.ContainsAny(id, `/\`) {
		return "", fmt.Errorf("jsonl: %q is not a session id", id)
	}
	dir := filepath.Join(s.root, id)
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		return "", fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound)
	}
	return dir, nil
}

// repairTail makes the log end on a newline, so the next record starts on a
// line of its own: a complete last record gets its newline, a torn one is cut.
func repairTail(path string) (err error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	// A close that fails is a truncate that may not have landed.
	defer func() { err = errors.Join(err, f.Close()) }()
	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return err
	}
	return readTail(f, size, func(buf []byte) (bool, error) {
		i := bytes.LastIndexByte(buf, '\n')
		switch {
		case i == len(buf)-1:
			return true, nil
		case i < 0 && int64(len(buf)) < size:
			return false, nil // the last line starts further back
		case json.Valid(buf[i+1:]):
			_, err := f.WriteAt([]byte{'\n'}, size)
			return true, err
		default:
			return true, f.Truncate(size - int64(len(buf)-i-1))
		}
	})
}

// lastAt is the time of the last complete record, zero if there is none. It
// reads from the end, so listing a long session costs no more than a short one.
func lastAt(path string) (time.Time, error) {
	f, err := os.Open(path)
	if errors.Is(err, fs.ErrNotExist) {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, err
	}
	defer func() { _ = f.Close() }() // read-only
	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return time.Time{}, err
	}
	var at time.Time
	err = readTail(f, size, func(buf []byte) (bool, error) {
		lines := bytes.Split(buf, []byte{'\n'})
		// The first line may be cut by the window, unless it is the file's start.
		first := 1
		if int64(len(buf)) == size {
			first = 0
		}
		for i := len(lines) - 1; i >= first; i-- {
			var e struct{ At time.Time }
			if json.Unmarshal(lines[i], &e) == nil {
				at = e.At
				return true, nil
			}
		}
		return false, nil
	})
	return at, err
}

// readTail hands fn the last bytes of a file, growing the window until fn is
// done or the window is the whole file.
func readTail(f *os.File, size int64, fn func(buf []byte) (done bool, err error)) error {
	for window := int64(8 * 1024); ; window *= 4 {
		window = min(window, size)
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return err
		}
		if done, err := fn(buf); done || err != nil || window == size {
			return err
		}
	}
}
