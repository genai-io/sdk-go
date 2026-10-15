package jsonl_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
	"github.com/genai-io/sdk-go/pkg/session/jsonl"
)

func ctx() context.Context { return context.Background() }

func open(t *testing.T, dir string) *jsonl.Store {
	t.Helper()
	s, err := jsonl.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return s
}

func store(t *testing.T) *jsonl.Store { return open(t, t.TempDir()) }

func msg(text string) session.Entry {
	return session.Entry{Type: session.EntryMessage, Messages: []ai.Message{ai.UserMessage(text)}}
}

func create(t *testing.T, s *jsonl.Store) string {
	t.Helper()
	id, err := s.Create(ctx())
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	return id
}

func appendAll(t *testing.T, s *jsonl.Store, id string, entries ...session.Entry) {
	t.Helper()
	for _, e := range entries {
		if err := s.Append(ctx(), id, e); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
}

func texts(t *testing.T, s *jsonl.Store, id string) []string {
	t.Helper()
	var out []string
	for e, err := range s.Entries(ctx(), id) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		out = append(out, e.Messages[0].Text())
	}
	return out
}

// claim is what a new process does before it writes.
func claim(t *testing.T, s *jsonl.Store, id string) {
	t.Helper()
	release, err := s.Claim(ctx(), id)
	if err != nil {
		t.Fatalf("Claim: %v", err)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
}

func entriesPath(dir, id string) string { return filepath.Join(dir, id, "entries.jsonl") }

func TestEntriesAcceptBlankCRLFLines(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	line, err := json.Marshal(msg("restored"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(entriesPath(dir, id), []byte("\r\n"+string(line)+"\r\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := texts(t, s, id); !slices.Equal(got, []string{"restored"}) {
		t.Fatalf("CRLF entries = %v", got)
	}
}

func TestEmptySessionReadsAsEmpty(t *testing.T) {
	s := store(t)
	if got := texts(t, s, create(t, s)); len(got) != 0 {
		t.Errorf("a fresh session holds %d entries", len(got))
	}
}

func TestASessionSurvivesReopening(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	appendAll(t, s, id, msg("written before the restart"))
	if got := texts(t, open(t, dir), id); !slices.Equal(got, []string{"written before the restart"}) {
		t.Fatalf("after reopening: %v", got)
	}
}

func TestListIsNewestFirst(t *testing.T) {
	s := store(t)
	first := create(t, s)
	second := create(t, s)
	e := msg("touches the older one")
	e.At = time.Now().Add(time.Hour)
	appendAll(t, s, first, e)

	list, err := s.List(ctx())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 || list[0].ID != first || list[1].ID != second {
		t.Fatalf("List = %+v, want %s then %s", list, first, second)
	}
}

// Filesystems can round an append's mtime below a newer session's creation, so
// recency is the logged time — read from the end, past an entry longer than
// the first window.
func TestListUsesTheLoggedTime(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	first := create(t, s)
	second := create(t, s)
	at := time.Now().Add(time.Hour).UTC()
	long := msg(strings.Repeat("x", 64*1024))
	long.At = at
	appendAll(t, s, first, long)
	coarse := time.Now().Add(-time.Hour)
	if err := os.Chtimes(entriesPath(dir, first), coarse, coarse); err != nil {
		t.Fatal(err)
	}
	list, err := open(t, dir).List(ctx())
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 || list[0].ID != first || list[1].ID != second || !list[0].UpdatedAt.Equal(at) {
		t.Fatalf("List = %+v, want %s first at %s", list, first, at)
	}
}

// A process killed mid-append leaves a half-written last line. Losing that one
// entry is the cost; losing the session is not acceptable.
func TestATornLastLineEndsTheSessionRatherThanFailingIt(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	appendAll(t, s, id, msg("first"), msg("second"))
	raw, err := os.ReadFile(entriesPath(dir, id))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(entriesPath(dir, id), raw[:len(raw)-12], 0o600); err != nil {
		t.Fatal(err)
	}
	if got := texts(t, s, id); !slices.Equal(got, []string{"first"}) {
		t.Fatalf("read %v from a torn file, want the one complete entry", got)
	}
}

// What a store that appended straight onto the wreckage would bury inside an
// unreadable line is the next entry. A claim repairs the tail first.
func TestAClaimRepairsATornTail(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	appendAll(t, s, id, msg("one"), msg("two"), msg("three"))
	raw, err := os.ReadFile(entriesPath(dir, id))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(entriesPath(dir, id), raw[:len(raw)-20], 0o600); err != nil {
		t.Fatal(err)
	}

	second := open(t, dir)
	claim(t, second, id)
	appendAll(t, second, id, msg("after the crash"))
	if got, want := texts(t, second, id), []string{"one", "two", "after the crash"}; !slices.Equal(got, want) {
		t.Errorf("entries = %v, want %v", got, want)
	}
}

// A record that was written whole but lost its newline is kept, not cut.
func TestAClaimKeepsACompleteTailWithoutItsNewline(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	appendAll(t, s, id, msg("first"))
	raw, err := os.ReadFile(entriesPath(dir, id))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(entriesPath(dir, id), bytes.TrimSuffix(raw, []byte{'\n'}), 0o600); err != nil {
		t.Fatal(err)
	}
	claim(t, s, id)
	appendAll(t, s, id, msg("second"))
	if got, want := texts(t, s, id), []string{"first", "second"}; !slices.Equal(got, want) {
		t.Errorf("entries = %v, want %v", got, want)
	}
}

// Only the last line is allowed to be unreadable, because that is a process
// that was killed. One that ends in a newline is a hole, and a conversation
// with a hole is not a shorter one — so it is reported.
func TestACorruptRecordIsReported(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	appendAll(t, s, id, msg("one"), msg("two"), msg("three"))
	raw, err := os.ReadFile(entriesPath(dir, id))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	lines[1] = lines[1][:len(lines[1])/2]
	if err := os.WriteFile(entriesPath(dir, id), []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var failure error
	for _, err := range s.Entries(ctx(), id) {
		if err != nil {
			failure = err
		}
	}
	if failure == nil {
		t.Fatal("the middle of the session is missing and nothing said so")
	}
	if _, _, err := session.Open(ctx(), s, id); err == nil {
		t.Error("a conversation with a hole in it was restored")
	}
}

func TestLargeEntriesCanBeRestored(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	id := create(t, s)
	text := strings.Repeat("x", 16*1024*1024)
	appendAll(t, s, id, msg(text), msg("next"))
	opened, messages, err := session.Open(ctx(), open(t, dir), id)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = opened.Close() }()
	if len(messages) != 2 || messages[0].Text() != text || messages[1].Text() != "next" {
		t.Fatal("a large entry did not survive restoration")
	}
}

func TestIdleSessionsDoNotKeepAppendFilesOpen(t *testing.T) {
	var fdDir string
	for _, path := range []string{"/proc/self/fd", "/dev/fd"} {
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			fdDir = path
			break
		}
	}
	if fdDir == "" {
		t.Skip("file descriptor listing is unavailable")
	}
	count := func() int {
		t.Helper()
		f, err := os.Open(fdDir)
		if err != nil {
			t.Fatal(err)
		}
		names, err := f.Readdirnames(-1)
		_ = f.Close()
		if err != nil {
			t.Fatal(err)
		}
		return len(names)
	}
	s := store(t)
	before := count()
	for range 256 {
		appendAll(t, s, create(t, s), msg("saved"))
	}
	if after := count(); after > before+16 {
		t.Errorf("idle sessions retained file handles: before=%d, after=%d", before, after)
	}
}

// Ids come from application input — a -resume flag, a request, a filename — so
// one that is not a single path element is refused rather than joined onto the
// root.
func TestAnIdThatIsNotOneIsRefused(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "beside"), 0o700); err != nil {
		t.Fatal(err)
	}
	s := open(t, filepath.Join(root, "sessions"))
	for _, id := range []string{"", ".", "..", "../beside", "a/b", `a\b`} {
		t.Run(fmt.Sprintf("%q", id), func(t *testing.T) {
			if _, err := s.Claim(ctx(), id); err == nil {
				t.Error("Claim accepted it")
			}
			if err := s.Append(ctx(), id, msg("x")); err == nil {
				t.Error("Append accepted it")
			}
			var failed bool
			for _, err := range s.Entries(ctx(), id) {
				failed = failed || err != nil
			}
			if !failed {
				t.Error("Entries accepted it")
			}
		})
	}
	if _, err := os.Stat(filepath.Join(root, "beside", "entries.jsonl")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a directory outside the store was written to: %v", err)
	}
}

func TestACanceledAppendDoesNotWrite(t *testing.T) {
	s := store(t)
	id := create(t, s)
	canceled, cancel := context.WithCancel(ctx())
	cancel()
	if err := s.Append(canceled, id, msg("canceled")); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
	if got := texts(t, s, id); len(got) != 0 {
		t.Fatalf("a canceled append wrote %v", got)
	}
}

func BenchmarkAppendOneEntry(b *testing.B) {
	s, err := jsonl.Open(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	id, err := s.Create(context.Background())
	if err != nil {
		b.Fatal(err)
	}
	e := msg("a line of the sort a session writes")
	b.ReportAllocs()
	for b.Loop() {
		if err := s.Append(context.Background(), id, e); err != nil {
			b.Fatal(err)
		}
	}
}
