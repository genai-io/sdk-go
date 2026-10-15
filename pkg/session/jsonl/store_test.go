package jsonl_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
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

func lines(t *testing.T, dir, id string) []string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, id+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, l := range strings.Split(strings.TrimSuffix(string(data), "\n"), "\n") {
		var r struct{ Type string }
		if err := json.Unmarshal([]byte(l), &r); err != nil {
			t.Fatalf("line %q: %v", l, err)
		}
		out = append(out, r.Type)
	}
	return out
}

func texts(t *testing.T, s *jsonl.Store, id string) []string {
	t.Helper()
	msgs, err := s.Load(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, m := range msgs {
		out = append(out, m.Text())
	}
	return out
}

func saveTexts(t *testing.T, s *jsonl.Store, id string, words ...string) []ai.Message {
	t.Helper()
	var msgs []ai.Message
	for _, w := range words {
		msgs = append(msgs, ai.UserMessage(w))
	}
	if err := s.Save(context.Background(), id, msgs); err != nil {
		t.Fatal(err)
	}
	return msgs
}

// A conversation that only grew costs one line per new message, and one that
// was rewritten — compaction — costs a reset and the new conversation. The
// file keeps everything; the replay ends with what was saved last.
func TestASaveAppendsOnlyWhatIsNew(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	saveTexts(t, s, "s", "one")
	saveTexts(t, s, "s", "one", "two")
	saveTexts(t, s, "s", "one", "two")
	if got, want := lines(t, dir, "s"), []string{"message", "message"}; !slices.Equal(got, want) {
		t.Fatalf("after growing: %v, want %v", got, want)
	}
	saveTexts(t, s, "s", "summary")
	if got, want := lines(t, dir, "s"), []string{"message", "message", "reset", "message"}; !slices.Equal(got, want) {
		t.Fatalf("after compacting: %v, want %v", got, want)
	}
	if got := texts(t, open(t, dir), "s"); !slices.Equal(got, []string{"summary"}) {
		t.Errorf("replayed %v", got)
	}
}

// An edit made after a save is a change, not something the store already holds.
func TestAnEditAfterSavingIsSaved(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	msgs := saveTexts(t, s, "s", "as written")
	msgs[0].Content[0].Text = "edited"
	if err := s.Save(context.Background(), "s", msgs); err != nil {
		t.Fatal(err)
	}
	if got := texts(t, s, "s"); !slices.Equal(got, []string{"edited"}) {
		t.Errorf("loaded %v", got)
	}
}

// Another store — another process — appended since this one last wrote. This
// one reads the file again rather than appending against what it remembers.
func TestASaveSeesWhatAnotherStoreWrote(t *testing.T) {
	dir := t.TempDir()
	a, b := open(t, dir), open(t, dir)
	saveTexts(t, a, "s", "one")
	saveTexts(t, b, "s", "one", "two")
	saveTexts(t, a, "s", "one", "two", "three")
	if got, want := lines(t, dir, "s"), []string{"message", "message", "message"}; !slices.Equal(got, want) {
		t.Errorf("lines = %v, want %v", got, want)
	}
}

// A process killed mid-write leaves an unterminated last line. Load leaves it
// out, and the next save cuts it off so its own lines start clean.
func TestAnUnfinishedLastLineIsDropped(t *testing.T) {
	dir := t.TempDir()
	saveTexts(t, open(t, dir), "s", "one")
	f, err := os.OpenFile(filepath.Join(dir, "s.jsonl"), os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString(`{"at":"2026-01-01T00:00:00Z","type":"mess`)
	_ = f.Close()

	s := open(t, dir)
	if got := texts(t, s, "s"); !slices.Equal(got, []string{"one"}) {
		t.Fatalf("loaded %v", got)
	}
	saveTexts(t, s, "s", "one", "two")
	if got := texts(t, open(t, dir), "s"); !slices.Equal(got, []string{"one", "two"}) {
		t.Errorf("after the next save: %v", got)
	}
}

// A bad line that is not the last is a hole, and a conversation with a hole
// in it is not a shorter one.
func TestABadLineInTheMiddleIsAnError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "s.jsonl")
	if err := os.WriteFile(path, []byte("{broken\n"+`{"type":"reset"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := open(t, dir).Load(context.Background(), "s"); err == nil {
		t.Error("loaded past a broken line")
	}
}
