package jsonl_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session/jsonl"
)

func open(t *testing.T, dir string) *jsonl.Store {
	t.Helper()
	s, err := jsonl.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func save(t *testing.T, s *jsonl.Store, id string) {
	t.Helper()
	if err := s.Save(context.Background(), id, []ai.Message{ai.UserMessage(id)}); err != nil {
		t.Fatal(err)
	}
}

// A second process finds what the first saved, newest first, and the
// half-written file a save renames from is never listed or left behind.
func TestListIsNewestFirst(t *testing.T) {
	dir := t.TempDir()
	first := open(t, dir)
	save(t, first, "older")
	save(t, first, "newer")
	past := time.Now().Add(-time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "older.jsonl"), past, past); err != nil {
		t.Fatal(err)
	}

	list, err := open(t, dir).List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 || list[0].ID != "newer" || list[1].ID != "older" {
		t.Fatalf("List = %+v", list)
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Errorf("the store holds %d files for 2 sessions", len(files))
	}
}

// Ids come from application input — a -resume flag, a request — so one that
// is not a plain file name is refused rather than joined onto the root.
func TestAnIdThatIsNotOneIsRefused(t *testing.T) {
	root := t.TempDir()
	s := open(t, filepath.Join(root, "sessions"))
	for _, id := range []string{"", ".", "..", ".hidden", "../beside", "a/b", `a\b`} {
		t.Run(fmt.Sprintf("%q", id), func(t *testing.T) {
			if err := s.Save(context.Background(), id, nil); err == nil {
				t.Error("Save accepted it")
			}
			if _, err := s.Load(context.Background(), id); err == nil {
				t.Error("Load accepted it")
			}
		})
	}
	if _, err := os.Stat(filepath.Join(root, "beside.jsonl")); !os.IsNotExist(err) {
		t.Errorf("a file outside the store was written: %v", err)
	}
}

// One message per line is the format: a session can be tailed, grepped and
// counted with the tools that read lines.
func TestEachMessageIsOneLine(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	msgs := []ai.Message{ai.UserMessage("one\ntwo"), ai.UserMessage("three")}
	if err := s.Save(context.Background(), "s", msgs); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "s.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n"); len(lines) != 2 {
		t.Errorf("%d lines for 2 messages:\n%s", len(lines), data)
	}
}
