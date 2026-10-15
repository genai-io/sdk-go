// Package jsonl stores sessions on the filesystem: a directory per session
// holding a log of what happened to its conversation, one JSON object per line.
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
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

const (
	logFile  = "log.jsonl"
	lockFile = ".lock"
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
	if err := repairTail(filepath.Join(dir, logFile)); err != nil {
		_ = lock.Close()
		return nil, err
	}
	return lock.Close, nil
}

// record is one line of the log: a message appended, or the whole
// conversation replaced.
type record struct {
	At       time.Time    `json:"at"`
	Replace  bool         `json:"replace,omitempty"`
	Messages []ai.Message `json:"messages"`
}

// Append adds a message to the end of the conversation.
func (s *Store) Append(ctx context.Context, id string, msg ai.Message) error {
	return s.write(ctx, id, record{Messages: []ai.Message{msg}})
}

// Replace swaps the whole conversation for msgs.
func (s *Store) Replace(ctx context.Context, id string, msgs []ai.Message) error {
	return s.write(ctx, id, record{Replace: true, Messages: msgs})
}

// write appends one record and syncs it to disk.
func (s *Store) write(ctx context.Context, id string, r record) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	dir, err := s.dir(id)
	if err != nil {
		return err
	}
	r.At = time.Now().UTC()
	line, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("jsonl: encoding record: %w", err)
	}
	f, err := os.OpenFile(filepath.Join(dir, logFile), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	_, err = f.Write(append(line, '\n'))
	if err == nil {
		err = f.Sync()
	}
	return errors.Join(err, f.Close())
}

// Load folds the log into the conversation: messages append, a replacement
// starts over. An incomplete final record — a crash mid-write — ends it; a bad
// record anywhere else is an error, because a conversation with a hole in it
// is not a shorter one.
func (s *Store) Load(ctx context.Context, id string) ([]ai.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dir, err := s.dir(id)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(filepath.Join(dir, logFile))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }() // read-only
	var messages []ai.Message
	reader := bufio.NewReader(f)
	for n := 1; ; n++ {
		line, readErr := reader.ReadBytes('\n')
		if readErr != nil && readErr != io.EOF {
			return nil, readErr
		}
		if len(bytes.TrimSpace(line)) > 0 {
			var r record
			if err := json.Unmarshal(line, &r); err != nil {
				if readErr == io.EOF {
					return messages, nil // torn
				}
				return nil, fmt.Errorf("jsonl: %s: line %d does not parse: %w", id, n, err)
			}
			if r.Replace {
				messages = r.Messages
			} else {
				messages = append(messages, r.Messages...)
			}
		}
		if readErr == io.EOF {
			return messages, nil
		}
	}
}

// Meta is a session as List reports it.
type Meta struct {
	ID string
	// UpdatedAt is when the conversation last changed, or when the session
	// was created if it never has.
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
		at, err := lastAt(filepath.Join(s.root, d.Name(), logFile))
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
