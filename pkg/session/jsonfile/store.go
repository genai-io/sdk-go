// Package jsonfile keeps sessions on the filesystem, one JSON file per session.
package jsonfile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

const ext = ".json"

// Store keeps sessions in a directory.
type Store struct{ root string }

var _ session.Store = (*Store)(nil)

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("jsonfile: creating %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

// Save writes the conversation beside the old one and renames it into place,
// so a crash leaves one or the other, never half of either.
func (s *Store) Save(ctx context.Context, id string, msgs []ai.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.path(id)
	if err != nil {
		return err
	}
	data, err := json.Marshal(msgs)
	if err != nil {
		return fmt.Errorf("jsonfile: encoding %s: %w", id, err)
	}
	tmp, err := os.CreateTemp(s.root, ".save-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }() // already renamed away on success
	_, err = tmp.Write(data)
	if err == nil {
		err = tmp.Sync()
	}
	if err = errors.Join(err, tmp.Close()); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// Load reads a session's conversation.
func (s *Store) Load(ctx context.Context, id string) ([]ai.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path, err := s.path(id)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("jsonfile: %s: %w", id, session.ErrNotFound)
	}
	if err != nil {
		return nil, err
	}
	var msgs []ai.Message
	if err := json.Unmarshal(data, &msgs); err != nil {
		return nil, fmt.Errorf("jsonfile: reading %s: %w", id, err)
	}
	return msgs, nil
}

// Session is a saved session as List reports it.
type Session struct {
	ID      string
	SavedAt time.Time
}

// List returns every saved session, most recently saved first.
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
		return "", fmt.Errorf("jsonfile: %q is not a session id", id)
	}
	return filepath.Join(s.root, id+ext), nil
}
