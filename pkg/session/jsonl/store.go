// Package jsonl keeps sessions on the filesystem as append-only logs.
//
// A store is a directory, and a session is one file in it, <id>.jsonl. Each
// line is one record, and a file is only ever appended to:
//
//	{"at":"…","type":"message","message":{…}}   a message joined the conversation
//	{"at":"…","type":"reset"}                   the conversation starts over here
//
// Load replays the lines in order: a message appends, a reset empties the
// conversation. So the file is the session's whole history and the
// conversation is what the replay ends with.
//
// Save appends only what the file does not hold yet. A conversation that grew
// costs one line per new message; one that changed — compaction, an edit —
// costs a reset and the new conversation. A process killed mid-write leaves an
// unterminated last line, which Load ignores and the next Save cuts off.
package jsonl

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

const ext = ".jsonl"

// Store keeps sessions in a directory.
type Store struct {
	root string

	mu sync.Mutex
	// written is what this store last left in each file it wrote, so a Save
	// can tell what is new without reading the file back. A file whose size
	// no longer matches was written by someone else, and is read again.
	written map[string]written
}

type written struct {
	size int64
	msgs []ai.Message
}

// record is one line of a session file.
type record struct {
	At      time.Time   `json:"at"`
	Type    string      `json:"type"` // "message" or "reset"
	Message *ai.Message `json:"message,omitempty"`
}

var _ session.Store = (*Store)(nil)

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonl: creating %s: %w", dir, err)
	}
	return &Store{root: dir, written: map[string]written{}}, nil
}

// Save appends to the session file whatever msgs adds to what it holds.
func (s *Store) Save(ctx context.Context, id string, msgs []ai.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.path(id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	prev, err := s.held(id, path)
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	var lines bytes.Buffer
	enc := json.NewEncoder(&lines)
	start := len(prev.msgs)
	if len(msgs) < start || !slices.EqualFunc(prev.msgs, msgs[:start], func(a, b ai.Message) bool {
		return reflect.DeepEqual(a, b)
	}) {
		_ = enc.Encode(record{At: now, Type: "reset"})
		start = 0
	}
	for _, m := range msgs[start:] {
		if err := enc.Encode(record{At: now, Type: "message", Message: &m}); err != nil {
			return fmt.Errorf("jsonl: encoding %s: %w", id, err)
		}
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	_, err = f.Write(lines.Bytes())
	if err == nil {
		err = f.Sync()
	}
	if err = errors.Join(err, f.Close()); err != nil {
		delete(s.written, id) // what reached the file is unknown; read it next time
		return err
	}
	s.written[id] = written{size: prev.size + int64(lines.Len()), msgs: clone(msgs)}
	return nil
}

// held is what the session file holds: remembered if this store wrote it last,
// read back otherwise, with an unfinished last line cut off so the next line
// starts on its own.
func (s *Store) held(id, path string) (written, error) {
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return written{}, nil
	}
	if err != nil {
		return written{}, err
	}
	if w, ok := s.written[id]; ok && w.size == info.Size() {
		return w, nil
	}
	msgs, end, err := replay(path)
	if err != nil {
		return written{}, err
	}
	if end < info.Size() {
		if err := os.Truncate(path, end); err != nil {
			return written{}, err
		}
	}
	return written{size: end, msgs: msgs}, nil
}

// Load replays the session file into its conversation.
func (s *Store) Load(ctx context.Context, id string) ([]ai.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path, err := s.path(id)
	if err != nil {
		return nil, err
	}
	msgs, _, err := replay(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("jsonl: %s: %w", id, session.ErrNotFound)
	}
	return msgs, err
}

// replay folds a session file's lines into the conversation they end with,
// and reports where the last complete line ends. An unterminated last line is
// a write that did not finish, and is not part of the session. A bad line
// anywhere else is an error: a conversation with a hole in it is not a
// shorter one.
func replay(path string) ([]ai.Message, int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var msgs []ai.Message
	var end int64
	for n := 1; ; n++ {
		i := bytes.IndexByte(data[end:], '\n')
		if i < 0 {
			return msgs, end, nil
		}
		line := data[end : end+int64(i)]
		end += int64(i) + 1
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, 0, fmt.Errorf("jsonl: %s line %d: %w", filepath.Base(path), n, err)
		}
		switch {
		case r.Type == "reset":
			msgs = nil
		case r.Type == "message" && r.Message != nil:
			msgs = append(msgs, *r.Message)
		default:
			return nil, 0, fmt.Errorf("jsonl: %s line %d: not a message or a reset", filepath.Base(path), n)
		}
	}
}

// Session is a session file as List reports it.
type Session struct {
	ID      string
	SavedAt time.Time
}

// List returns every session, most recently saved first.
func (s *Store) List(ctx context.Context) ([]Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	files, err := os.ReadDir(s.root)
	if err != nil {
		return nil, err
	}
	var out []Session
	for _, f := range files {
		id, ok := strings.CutSuffix(f.Name(), ext)
		if !ok || f.IsDir() || strings.HasPrefix(id, ".") {
			continue
		}
		info, err := f.Info()
		if err != nil {
			continue // removed while listing
		}
		out = append(out, Session{ID: id, SavedAt: info.ModTime()})
	}
	slices.SortFunc(out, func(a, b Session) int { return b.SavedAt.Compare(a.SavedAt) })
	return out, nil
}

// path is where a session lives. Ids come from application input, so one that
// is not a plain file name is refused rather than joined onto the root.
func (s *Store) path(id string) (string, error) {
	if id == "" || strings.HasPrefix(id, ".") || strings.ContainsAny(id, `/\`) {
		return "", fmt.Errorf("jsonl: %q is not a session id", id)
	}
	return filepath.Join(s.root, id+ext), nil
}

// clone copies what the caller still holds, so an edit after Save is a change
// the next Save sees rather than one it already agrees with.
func clone(msgs []ai.Message) []ai.Message {
	out := slices.Clone(msgs)
	for i := range out {
		out[i].Content = out[i].Content.Clone()
	}
	return out
}
