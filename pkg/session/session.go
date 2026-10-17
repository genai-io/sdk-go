// Package session keeps conversations on disk, one append-only log each.
//
// A store is a directory, and a session is one file in it, <id>.jsonl, with
// one Record per line:
//
//	{"type":"message","time":…,"message":{…}}   a message joined the conversation
//	{"type":"reset","time":…}                   the conversation starts over here
//	{"type":"title","time":…,"data":{…}}        anything the application keeps
//
// The conversation is what the messages and resets replay to, and Load
// returns it. Records returns every record, in the order written, for
// whatever else the application keeps beside the conversation.
//
// A session has one writer at a time, across processes too. Save it on every
// change to the conversation:
//
//	w, err := store.Write(ctx, id)   // session.ErrBusy if another writer holds it
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
// Save appends what is new: a conversation that grew costs a line per message,
// one that changed in place — compaction, an edit — a reset and the whole
// conversation. A process killed mid-write leaves an unterminated last line,
// which reading ignores and the next writer cuts off.
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
	"os"
	"path/filepath"
	"reflect"
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

// Record is one line of a session. Type "message" carries Message, "reset"
// carries nothing, and any other type is the application's, carrying Data.
type Record struct {
	Type    string          `json:"type"`
	Time    time.Time       `json:"time"`
	Message *ai.Message     `json:"message,omitempty"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// Conversation replays records into the conversation they end with: a
// message appends, a reset empties, and any other record is passed over.
func Conversation(records iter.Seq2[Record, error]) ([]ai.Message, error) {
	var msgs []ai.Message
	for r, err := range records {
		if err != nil {
			return nil, err
		}
		switch r.Type {
		case "message":
			msgs = append(msgs, *r.Message)
		case "reset":
			msgs = nil
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
	return Conversation(s.Records(ctx, id))
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
		for r, err := range read(f, id) {
			if !yield(r, err) || err != nil {
				return
			}
		}
	}
}

// read decodes complete lines from f.
func read(f io.Reader, id string) iter.Seq2[Record, error] {
	return func(yield func(Record, error) bool) {
		br := bufio.NewReader(f)
		for n := 1; ; n++ {
			line, err := br.ReadBytes('\n')
			if errors.Is(err, io.EOF) {
				return // nothing, or a line still being written
			}
			if err != nil {
				yield(Record{}, err)
				return
			}
			var r Record
			if err := json.Unmarshal(line, &r); err != nil {
				yield(Record{}, fmt.Errorf("session: %s line %d: %w", id, n, err))
				return
			}
			if r.Type == "" || r.Type == "message" && r.Message == nil {
				yield(Record{}, fmt.Errorf("session: %s line %d: not a session record", id, n))
				return
			}
			if !yield(r, nil) {
				return
			}
		}
	}
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

// Write takes the session's writer, creating the session if it does not
// exist. It is ErrBusy while another writer, in this process or another,
// holds it. Close gives it back.
func (s *Store) Write(ctx context.Context, id string) (*Writer, error) {
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
	data, err := io.ReadAll(f)
	w := &Writer{f: f}
	if err == nil {
		w.saved, err = Conversation(read(bytes.NewReader(data), id))
	}
	if err == nil {
		err = f.Truncate(int64(bytes.LastIndexByte(data, '\n') + 1))
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
	saved []ai.Message // the conversation as the file holds it
	err   error        // a failed write leaves the file unfit to append to
}

// Save appends what msgs adds to the conversation the session holds, or a
// reset and all of msgs when it is not an extension of it, and flushes the
// session to disk. Messages are matched by ID, or compared whole when they
// have none; give a changed message a new ID, or none.
func (w *Writer) Save(msgs []ai.Message) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	now := time.Now()
	var lines []Record
	start := len(w.saved)
	if len(msgs) < start || !slices.EqualFunc(w.saved, msgs[:start], same) {
		lines = append(lines, Record{Type: "reset", Time: now})
		start = 0
	}
	if start == len(msgs) && lines == nil {
		return w.err
	}
	for i := range msgs[start:] {
		lines = append(lines, Record{Type: "message", Time: now, Message: &msgs[start+i]})
	}
	if err := w.write(lines...); err != nil {
		return err
	}
	w.saved = slices.Clone(msgs)
	return w.sync()
}

func same(a, b ai.Message) bool {
	if a.ID != "" || b.ID != "" {
		return a.ID == b.ID
	}
	return reflect.DeepEqual(a, b)
}

// Append adds an application record of type typ carrying data as JSON. The
// types "message" and "reset" are the conversation's. It is on disk once a
// Save or Close follows.
func (w *Writer) Append(typ string, data any) error {
	if typ == "" || typ == "message" || typ == "reset" {
		return fmt.Errorf("session: %q is not an application record type", typ)
	}
	raw, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("session: encoding %s: %w", typ, err)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.write(Record{Type: typ, Time: time.Now(), Data: raw})
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

func (w *Writer) sync() error {
	if err := w.f.Sync(); err != nil {
		w.err = err
	}
	return w.err
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
