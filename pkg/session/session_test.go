package session_test

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

var ctx = context.Background()

func msg(id, text string) ai.Message {
	m := ai.UserMessage(text)
	m.ID = id
	return m
}

func open(t *testing.T) (*session.Store, string) {
	t.Helper()
	dir := t.TempDir()
	s, err := session.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	return s, dir
}

func write(t *testing.T, s *session.Store, id string) *session.Writer {
	t.Helper()
	w, err := s.Write(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	return w
}

func types(t *testing.T, s *session.Store, id string) []string {
	t.Helper()
	var out []string
	for r, err := range s.Records(ctx, id) {
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, r.Type)
	}
	return out
}

func ids(msgs []ai.Message) []string {
	var out []string
	for _, m := range msgs {
		out = append(out, m.ID+":"+m.Text())
	}
	return out
}

func TestSaveAppendsWhatIsNewAndResetsWhatChanged(t *testing.T) {
	s, _ := open(t)
	w := write(t, s, "a")
	a, b, c := msg("1", "a"), msg("2", "b"), msg("3", "c")
	steps := [][]ai.Message{{a}, {a, b}, {a, b}, {c}, {c, a}}
	for _, conv := range steps {
		if err := w.Save(conv); err != nil {
			t.Fatal(err)
		}
		if err := w.Append("note", map[string]int{"n": len(conv)}); err != nil {
			t.Fatal(err)
		}
	}
	got := types(t, s, "a")
	want := []string{"message", "note", "message", "note", "note", "reset", "message", "note", "message", "note"}
	if !slices.Equal(got, want) {
		t.Fatalf("records = %v, want %v", got, want)
	}
	loaded, err := s.Load(ctx, "a")
	if err != nil || !slices.Equal(ids(loaded), ids([]ai.Message{c, a})) {
		t.Fatalf("Load = %v, %v", ids(loaded), err)
	}

	// A reopened writer picks up where the file is: nothing new, nothing written.
	_ = w.Close()
	w = write(t, s, "a")
	if err := w.Save([]ai.Message{c, a}); err != nil {
		t.Fatal(err)
	}
	if n := len(types(t, s, "a")); n != len(want) {
		t.Fatalf("resave wrote %d records", n-len(want))
	}
}

func TestSaveComparesMessagesWithoutIDs(t *testing.T) {
	s, _ := open(t)
	w := write(t, s, "a")
	a, b := ai.UserMessage("a"), ai.UserMessage("b")
	for _, conv := range [][]ai.Message{{a}, {a, b}, {b}} {
		if err := w.Save(conv); err != nil {
			t.Fatal(err)
		}
	}
	if got := types(t, s, "a"); !slices.Equal(got, []string{"message", "message", "reset", "message"}) {
		t.Fatalf("records = %v", got)
	}
}

func TestUnfinishedLineIsIgnoredAndCut(t *testing.T) {
	s, dir := open(t)
	w := write(t, s, "a")
	if err := w.Save([]ai.Message{msg("1", "a")}); err != nil {
		t.Fatal(err)
	}
	_ = w.Close()
	path := filepath.Join(dir, "a.jsonl")
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	_, _ = f.WriteString(`{"type":"message","mess`)
	_ = f.Close()

	if got := types(t, s, "a"); !slices.Equal(got, []string{"message"}) {
		t.Fatalf("records = %v", got)
	}
	w = write(t, s, "a")
	if err := w.Save([]ai.Message{msg("1", "a"), msg("2", "b")}); err != nil {
		t.Fatal(err)
	}
	if got := types(t, s, "a"); !slices.Equal(got, []string{"message", "message"}) {
		t.Fatalf("records after a cut = %v", got)
	}
}

func TestBadLineIsAnError(t *testing.T) {
	s, dir := open(t)
	_ = os.WriteFile(filepath.Join(dir, "a.jsonl"), []byte("{broken}\n{\"type\":\"reset\"}\n"), 0o600)
	if _, err := s.Load(ctx, "a"); err == nil {
		t.Fatal("Load read past a bad line")
	}
	if _, err := s.Write(ctx, "a"); err == nil {
		t.Fatal("Write opened a session it cannot read")
	}
}

func TestOneWriterAtATime(t *testing.T) {
	s, dir := open(t)
	w := write(t, s, "a")
	if _, err := s.Write(ctx, "a"); !errors.Is(err, session.ErrBusy) {
		t.Fatalf("second Write = %v, want ErrBusy", err)
	}
	if err := s.Delete(ctx, "a"); !errors.Is(err, session.ErrBusy) {
		t.Fatalf("Delete while held = %v, want ErrBusy", err)
	}
	_ = w.Close()
	if err := w.Append("note", 1); err == nil {
		t.Fatal("Append after Close")
	}
	w = write(t, s, "a")
	_ = w.Close()
	if entries, _ := os.ReadDir(dir); len(entries) != 1 {
		t.Fatalf("left behind %v", entries)
	}
}

func TestDeleteForkList(t *testing.T) {
	s, _ := open(t)
	w := write(t, s, "a")
	_ = w.Save([]ai.Message{msg("1", "a")})
	_ = w.Append("note", "x")
	_ = w.Close()

	if err := s.Fork(ctx, "a", "b"); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(types(t, s, "b"), []string{"message", "note"}) {
		t.Fatalf("fork = %v", types(t, s, "b"))
	}
	if err := s.Fork(ctx, "a", "b"); !errors.Is(err, fs.ErrExist) {
		t.Fatalf("fork onto an existing session = %v", err)
	}
	if err := s.Fork(ctx, "missing", "c"); !errors.Is(err, session.ErrNotFound) {
		t.Fatalf("fork of a missing session = %v", err)
	}
	list, err := s.List(ctx)
	if err != nil || len(list) != 2 {
		t.Fatalf("List = %v, %v", list, err)
	}
	if err := s.Delete(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Load(ctx, "a"); !errors.Is(err, session.ErrNotFound) {
		t.Fatalf("Load after Delete = %v", err)
	}
	if err := s.Delete(ctx, "a"); !errors.Is(err, session.ErrNotFound) {
		t.Fatalf("second Delete = %v", err)
	}
}

func TestRefused(t *testing.T) {
	s, _ := open(t)
	for _, id := range []string{"", ".hidden", "a/b", `a\b`, ".."} {
		if _, err := s.Write(ctx, id); err == nil {
			t.Errorf("Write(%q) accepted", id)
		}
	}
	w := write(t, s, "a")
	for _, typ := range []string{"", "message", "reset"} {
		if err := w.Append(typ, nil); err == nil {
			t.Errorf("Append(%q) accepted", typ)
		}
	}
}

func TestConversation(t *testing.T) {
	a := msg("1", "a")
	recs := []session.Record{{Type: "message", Message: &a}, {Type: "note"}, {Type: "reset"}, {Type: "message", Message: &a}, {Type: "message", Message: &a}}
	got, err := session.Conversation(func(yield func(session.Record, error) bool) {
		for _, r := range recs {
			if !yield(r, nil) {
				return
			}
		}
	})
	if err != nil || len(got) != 2 {
		t.Fatalf("Conversation = %v, %v", got, err)
	}
}

func TestSaveKeepsAnEditThatKeepsItsID(t *testing.T) {
	s, _ := open(t)
	w := write(t, s, "a")
	conv := []ai.Message{msg("1", "draft")}
	if err := w.Save(conv); err != nil {
		t.Fatal(err)
	}
	conv[0].Content[0].Text = "edited" // in place, same ID
	if err := w.Save(conv); err != nil {
		t.Fatal(err)
	}
	loaded, err := s.Load(ctx, "a")
	if err != nil || len(loaded) != 1 || loaded[0].Text() != "edited" {
		t.Fatalf("Load = %v, %v; want the edit", ids(loaded), err)
	}
}
