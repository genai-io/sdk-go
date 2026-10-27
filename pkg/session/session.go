// Package session keeps conversations on disk, one append-only log each.
//
// A store is a directory, and a session is one file in it, <id>.jsonl, with
// one Record per line:
//
//	{"type":"message","time":…,"message":{…}}               a message joined the conversation
//	{"type":"truncate","time":…,"keep":3}                   the conversation is cut to its first 3 messages
//	{"type":"app","time":…,"name":"title","data":{…}}       anything the application keeps
//
// The conversation is what the messages and truncations replay to, and Load
// returns it. Records returns every record, in the order written, for
// whatever else the application keeps beside the conversation.
//
// The application's records are all of type "app", told apart by name, so
// the types are the package's alone: a later version can add one without
// meeting an application's, and a reader that meets a type it does not know
// fails rather than replaying the conversation wrong.
//
// A session has one writer at a time, across processes too. Save it on every
// change to the conversation:
//
//	w, err := store.Writer(ctx, id)  // session.ErrBusy if another writer holds it
//	defer w.Close()
//	history, err := store.Load(ctx, id)
//	a.SetMessages(history)
//	for e, err := range a.Run(ctx, msg) {
//		switch e.(type) {
//		case agent.MessageAdded, agent.MessagesReplaced:
//			err = w.Save(a.Messages())
//		}
//	}
//
// Save appends what is new. A conversation that grew costs a line per new
// message; one changed partway — an edit, a rewind, cleared tool output —
// costs a truncate to the unchanged prefix and the messages after it; a
// compaction is a truncate to nothing. To branch a session at message k, Fork
// it and Save the first k messages to the copy.
//
// A process killed mid-write leaves an unterminated last line, which reading
// ignores and the next writer cuts off. A reader that holds no writer may see
// a Save half done: take the writer before Load to resume.
package session

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

var (
	// ErrNotFound is returned for a session that does not exist.
	ErrNotFound = errors.New("session: not found")
	// ErrBusy is returned for a session another writer holds.
	ErrBusy = errors.New("session: in use by another writer")
)

// The types of Record.
const (
	TypeMessage  = "message"  // a message joined the conversation
	TypeTruncate = "truncate" // the conversation was cut to its first Keep messages
	TypeApp      = "app"      // the application's record, called Name, carrying Data
)

// Record is one line of a session.
type Record struct {
	Type    string          `json:"type"`
	Time    time.Time       `json:"time"`
	Message *ai.Message     `json:"message,omitempty"` // TypeMessage
	Keep    int             `json:"keep,omitempty"`    // TypeTruncate
	Name    string          `json:"name,omitempty"`    // TypeApp
	Data    json.RawMessage `json:"data,omitempty"`    // TypeApp
}

// Apply returns the conversation msgs becomes after r: a message is appended,
// a truncate cuts, and an app record leaves it as it is.
func (r Record) Apply(msgs []ai.Message) ([]ai.Message, error) {
	switch r.Type {
	case TypeMessage:
		return append(msgs, *r.Message), nil
	case TypeTruncate:
		if r.Keep > len(msgs) {
			return nil, fmt.Errorf("session: truncate keeps %d of %d messages", r.Keep, len(msgs))
		}
		return msgs[:r.Keep], nil
	case TypeApp:
		return msgs, nil
	}
	return nil, fmt.Errorf("session: record type %q is from a newer version", r.Type)
}

// conversation replays records into the conversation they end with.
func conversation(records iter.Seq2[Record, error]) ([]ai.Message, error) {
	var msgs []ai.Message
	for r, err := range records {
		if err == nil {
			msgs, err = r.Apply(msgs)
		}
		if err != nil {
			return nil, err
		}
	}
	return msgs, nil
}

const ext = ".jsonl"

// Store keeps sessions in a directory.
type Store struct{ dir string }

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("session: creating %s: %w", dir, err)
	}
	return &Store{dir: dir}, nil
}

// Load returns the session's conversation.
func (s *Store) Load(ctx context.Context, id string) ([]ai.Message, error) {
	return conversation(s.Records(ctx, id))
}

// Records returns the session's records in the order they were written. An
// unfinished last line is not one; a bad line anywhere else is an error.
func (s *Store) Records(ctx context.Context, id string) iter.Seq2[Record, error] {
	return func(yield func(Record, error) bool) {
		path, err := s.path(ctx, id)
		if err != nil {
			yield(Record{}, err)
			return
		}
		f, err := os.Open(path)
		if err != nil {
			yield(Record{}, notFound(id, err))
			return
		}
		defer func() { _ = f.Close() }()
		for r, err := range read(f, id, nil) {
			if err == nil {
				err = ctx.Err()
			}
			if !yield(r, err) || err != nil {
				return
			}
		}
	}
}

// read decodes complete lines from f.
func read(f io.Reader, id string, end *int64) iter.Seq2[Record, error] {
	return func(yield func(Record, error) bool) {
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 4096), math.MaxInt)
		scanner.Split(completeLine)
		n := 0
		for scanner.Scan() {
			n++
			line := scanner.Bytes()
			if end != nil {
				*end += int64(len(line))
			}
			var r Record
			if err := json.Unmarshal(line, &r); err != nil {
				yield(Record{}, fmt.Errorf("session: %s line %d: %w", id, n, err))
				return
			}
			if r.Type == "" || r.Type == TypeMessage && r.Message == nil || r.Keep < 0 || r.Type == TypeApp && r.Name == "" {
				yield(Record{}, fmt.Errorf("session: %s line %d: not a session record", id, n))
				return
			}
			if !yield(r, nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(Record{}, err)
		}
	}
}

// An unterminated tail may still be being written; only complete lines exist.
func completeLine(data []byte, atEOF bool) (int, []byte, error) {
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		return i + 1, data[:i+1], nil
	}
	if atEOF {
		return len(data), nil, nil
	}
	return 0, nil, nil
}

// Info is a session as List reports it.
type Info struct {
	ID      string
	ModTime time.Time // when it was last written
}

// List returns every session, most recently written first.
func (s *Store) List(ctx context.Context) ([]Info, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	files, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	var out []Info
	for _, f := range files {
		id, ok := strings.CutSuffix(f.Name(), ext)
		if !ok || f.IsDir() || validate(id) != nil {
			continue
		}
		info, err := f.Info()
		if err != nil {
			continue // removed while listing
		}
		out = append(out, Info{ID: id, ModTime: info.ModTime()})
	}
	slices.SortFunc(out, func(a, b Info) int { return b.ModTime.Compare(a.ModTime) })
	return out, nil
}

// Delete removes a session. It is ErrBusy while a writer holds it.
func (s *Store) Delete(ctx context.Context, id string) (err error) {
	path, err := s.path(ctx, id)
	if err != nil {
		return err
	}
	l, err := lock(path)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, l.unlock()) }()
	return notFound(id, os.Remove(path))
}

// Fork copies session from into a new session to. It is an error for to to
// exist already.
func (s *Store) Fork(ctx context.Context, from, to string) (err error) {
	src, err := s.path(ctx, from)
	if err != nil {
		return err
	}
	dst, err := s.path(ctx, to)
	if err != nil {
		return err
	}
	l, err := lock(dst)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, l.unlock()) }()
	if _, err := os.Stat(dst); err == nil {
		return fmt.Errorf("session: forking to %s: %w", to, fs.ErrExist)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return notFound(from, err)
	}
	data = data[:bytes.LastIndexByte(data, '\n')+1] // a line being written is not copied
	tmp := dst + ".tmp"
	if err := writeSynced(tmp, data); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

func writeSynced(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	return errors.Join(err, f.Close())
}

// Writer takes the session's writer, creating the session if it does not
// exist. It is ErrBusy while another writer, in this process or another,
// holds it. Close gives it back.
func (s *Store) Writer(ctx context.Context, id string) (*Writer, error) {
	path, err := s.path(ctx, id)
	if err != nil {
		return nil, err
	}
	l, err := lock(path)
	if err != nil {
		return nil, err
	}
	w, err := open(path, id)
	if err != nil {
		return nil, errors.Join(err, l.unlock())
	}
	w.lock = l
	return w, nil
}

// open reads the session the writer appends to, and cuts off a line an
// earlier writer did not finish.
func open(path, id string) (*Writer, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	w := &Writer{f: f}
	var end int64
	msgs, err := conversation(read(f, id, &end))
	if err == nil {
		w.saved, err = encode(msgs)
	}
	if err == nil {
		var info os.FileInfo
		if info, err = f.Stat(); err == nil && info.Size() != end {
			err = f.Truncate(end)
		}
	}
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	return w, nil
}

// Writer appends to one session. Its methods are safe for concurrent use.
type Writer struct {
	mu    sync.Mutex
	f     *os.File
	lock  *fileLock
	saved [][]byte // the conversation as the file holds it, one encoding per message
	err   error    // a failed write leaves the file unfit to append to
}

// Save makes the session hold msgs, appending only what differs from what it
// holds: the messages after the longest unchanged prefix, behind a truncate
// to that prefix when anything after it changed. It flushes the session to
// disk. Messages are compared by their encodings, so an edit that keeps its
// ID is saved, and a message read back from disk is not taken for an edit.
func (w *Writer) Save(msgs []ai.Message) error {
	encs, err := encode(msgs)
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	now := time.Now()
	keep := 0
	for keep < min(len(w.saved), len(encs)) && bytes.Equal(w.saved[keep], encs[keep]) {
		keep++
	}
	var lines []Record
	if keep < len(w.saved) {
		lines = append(lines, Record{Type: TypeTruncate, Time: now, Keep: keep})
	}
	for i := range msgs[keep:] {
		lines = append(lines, Record{Type: TypeMessage, Time: now, Message: &msgs[keep+i]})
	}
	if lines == nil {
		return w.err
	}
	if err := w.write(lines...); err != nil {
		return err
	}
	w.saved = encs
	if err := w.f.Sync(); err != nil {
		w.err = err
	}
	return w.err
}

func encode(msgs []ai.Message) ([][]byte, error) {
	encs := make([][]byte, len(msgs))
	for i, m := range msgs {
		var err error
		if encs[i], err = json.Marshal(m); err != nil {
			return nil, fmt.Errorf("session: encoding message %d: %w", i, err)
		}
	}
	return encs, nil
}

// Append adds an app record called name, carrying data as JSON. It is on
// disk once a Save or Close follows.
func (w *Writer) Append(name string, data any) error {
	if name == "" {
		return errors.New("session: an app record needs a name")
	}
	raw, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("session: encoding %s: %w", name, err)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.write(Record{Type: TypeApp, Time: time.Now(), Name: name, Data: raw})
}

func (w *Writer) write(records ...Record) error {
	if w.err != nil {
		return w.err
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			return fmt.Errorf("session: encoding %s: %w", r.Type, err)
		}
	}
	if w.f == nil {
		return os.ErrClosed
	}
	if _, err := w.f.Write(buf.Bytes()); err != nil {
		w.err = err
		return err
	}
	return nil
}

// Close flushes the session to disk and gives up the writer.
func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := errors.Join(w.f.Sync(), w.f.Close(), w.lock.unlock())
	w.f = nil
	return err
}

// path is where a session lives. Ids come from application input, so one that
// is not a plain file name is refused rather than joined onto the directory.
func (s *Store) path(ctx context.Context, id string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := validate(id); err != nil {
		return "", err
	}
	return filepath.Join(s.dir, id+ext), nil
}

func validate(id string) error {
	if id == "" || strings.HasPrefix(id, ".") || strings.ContainsAny(id, `/\`) {
		return fmt.Errorf("session: %q is not a session id", id)
	}
	return nil
}

func notFound(id string, err error) error {
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("session: %s: %w", id, ErrNotFound)
	}
	return err
}
