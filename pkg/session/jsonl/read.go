package jsonl

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

	"github.com/genai-io/sdk-go/pkg/session"
)

// Entries reads the log, ignoring only an incomplete final record after a crash.
func (s *Store) Entries(ctx context.Context, id string) iter.Seq2[session.Entry, error] {
	return func(yield func(session.Entry, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(session.Entry{}, err)
			return
		}
		dir, err := s.dir(id)
		if err != nil {
			yield(session.Entry{}, err)
			return
		}
		if _, err := s.readMeta(id); err != nil {
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
		defer func() { _ = f.Close() }()
		reader := bufio.NewReaderSize(f, 64*1024)
		for n := 1; ; n++ {
			line, readErr := reader.ReadBytes('\n')
			if err := ctx.Err(); err != nil {
				yield(session.Entry{}, err)
				return
			}
			if readErr != nil && readErr != io.EOF {
				yield(session.Entry{}, readErr)
				return
			}
			if len(bytes.TrimSpace(line)) == 0 {
				if readErr == io.EOF {
					return
				}
				continue
			}
			var e session.Entry
			if err := json.Unmarshal(line, &e); err != nil {
				if readErr == io.EOF {
					return
				}
				yield(session.Entry{}, fmt.Errorf("jsonl: %s: line %d does not parse: %w", id, n, err))
				return
			}
			if !yield(e, nil) || readErr == io.EOF {
				return
			}
		}
	}
}
