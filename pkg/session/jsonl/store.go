// Package jsonl keeps sessions on the filesystem as append-only logs.
//
// A store is a directory, and a session is one file in it, <id>.jsonl. Each
// line is one record, and a file is only ever appended to:
//
//	{"type":"message","message":{…}}   a message joined the conversation
//	{"type":"reset"}                   the conversation starts over here
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
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

const ext = ".jsonl"

// Store keeps sessions in a directory.
type Store struct{ root string }

// record is one line of a session file.
type record struct {
	Type    string      `json:"type"` // "message" or "reset"
	Message *ai.Message `json:"message,omitempty"`
}

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonl: creating %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

// Save appends to the session file whatever msgs adds to what it holds. It
// reads the file first, so what another process appended is not appended
// again, and an unfinished last line is cut off before the next one.
func (s *Store) Save(ctx context.Context, id string, msgs []ai.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.path(id)
	if err != nil {
		return err
	}
	held, end, err := replay(path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
	case err != nil:
		return err
	default:
		if err := os.Truncate(path, end); err != nil {
			return err
		}
	}

	var lines bytes.Buffer
	enc := json.NewEncoder(&lines)
	start := len(held)
	if len(msgs) < start || !slices.EqualFunc(held, msgs[:start], func(a, b ai.Message) bool {
		return reflect.DeepEqual(a, b)
	}) {
		_ = enc.Encode(record{Type: "reset"})
		start = 0
	}
	for _, m := range msgs[start:] {
		if err := enc.Encode(record{Type: "message", Message: &m}); err != nil {
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
	return errors.Join(err, f.Close())
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
