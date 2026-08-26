package jsonl_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/agent/session/jsonl"
	"github.com/genai-io/sdk-go/pkg/ai"
)

func ctx() context.Context { return context.Background() }

func open(t *testing.T, dir string) *jsonl.Store {
	t.Helper()
	s, err := jsonl.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func store(t *testing.T) *jsonl.Store { return open(t, t.TempDir()) }

func msg(text string) session.Entry {
	m := ai.UserMessage(text)
	return session.Entry{Type: session.EntryMessage, Message: &m}
}

func create(t *testing.T, s *jsonl.Store) session.Meta {
	t.Helper()
	meta, err := s.Create(ctx(), session.Meta{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	return meta
}

func read(t *testing.T, s *jsonl.Store, id string) []session.Entry {
	t.Helper()
	var out []session.Entry
	for e, err := range s.Entries(ctx(), id) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		out = append(out, e)
	}
	return out
}

func TestCreateAssignsAnID(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if meta.ID == "" {
		t.Fatal("Create left the ID blank")
	}
	if meta.CreatedAt.IsZero() {
		t.Error("Create left CreatedAt zero")
	}
	if second := create(t, s); second.ID == meta.ID {
		t.Error("two sessions were given the same ID")
	}
}

func TestAppendPreservesOrder(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("one"), msg("two"), msg("three")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	var got []string
	for _, e := range read(t, s, meta.ID) {
		got = append(got, e.Message.Text())
	}
	if want := []string{"one", "two", "three"}; strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("entries = %v, want %v", got, want)
	}
}

// A batched append and single appends must number the same way.
func TestSequenceNumbersRunFromOne(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("a"), msg("b")); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := s.Append(ctx(), meta.ID, msg("c")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	for i, e := range read(t, s, meta.ID) {
		if want := int64(i + 1); e.Seq != want {
			t.Errorf("entry %d has Seq %d, want %d", i, e.Seq, want)
		}
	}
}

func TestMissingSessionIsNotFound(t *testing.T) {
	s := store(t)
	if _, err := s.Meta(ctx(), "nope"); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Meta of an absent session = %v, want ErrNotFound", err)
	}
	if err := s.Append(ctx(), "nope", msg("x")); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Append to an absent session = %v, want ErrNotFound", err)
	}
}

func TestEmptySessionReadsAsEmpty(t *testing.T) {
	s := store(t)
	if got := read(t, s, create(t, s).ID); len(got) != 0 {
		t.Errorf("a fresh session holds %d entries", len(got))
	}
}

// The fold is what restore is: only message entries become the conversation.
func TestMessagesFoldsOnlyMessageEntries(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID,
		msg("hello"),
		session.Entry{Type: session.EntryInference, Inference: &session.Inference{Turn: 1, Attempt: 1}},
		msg("again"),
	); err != nil {
		t.Fatalf("Append: %v", err)
	}

	msgs, err := session.Messages(ctx(), s, meta.ID)
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("folded %d messages, want 2 — the inference entry is not one", len(msgs))
	}
	if msgs[0].Text() != "hello" || msgs[1].Text() != "again" {
		t.Errorf("folded %q and %q", msgs[0].Text(), msgs[1].Text())
	}
}

func TestMetaSurvivesAppend(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	meta.Title = "a title worth keeping"
	if err := s.SetMeta(ctx(), meta); err != nil {
		t.Fatalf("SetMeta: %v", err)
	}
	if err := s.Append(ctx(), meta.ID, msg("a")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	got, err := s.Meta(ctx(), meta.ID)
	if err != nil {
		t.Fatalf("Meta: %v", err)
	}
	if got.Title != "a title worth keeping" {
		t.Errorf("title = %q, want it preserved across an append", got.Title)
	}
	if got.Entries != 1 {
		t.Errorf("Entries = %d, want 1", got.Entries)
	}
}

// Entries is the store's to maintain, so a stale copy handed back must not
// undo what the store has counted since.
func TestSetMetaCannotRewriteTheCount(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("a"), msg("b")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	stale := meta // captured before the appends: Entries is 0 here
	stale.Title = "renamed"
	if err := s.SetMeta(ctx(), stale); err != nil {
		t.Fatalf("SetMeta: %v", err)
	}

	got, err := s.Meta(ctx(), meta.ID)
	if err != nil {
		t.Fatalf("Meta: %v", err)
	}
	if got.Entries != 2 {
		t.Errorf("Entries = %d after a stale SetMeta, want 2 — the store owns this field", got.Entries)
	}
	if got.Title != "renamed" {
		t.Errorf("title = %q, want the caller's field to have been taken", got.Title)
	}
}

func TestListIsNewestFirst(t *testing.T) {
	s := store(t)
	first := create(t, s)
	second := create(t, s)
	if err := s.Append(ctx(), first.ID, msg("touches the older one")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	list, err := s.List(ctx())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("listed %d sessions, want 2", len(list))
	}
	if list[0].ID != first.ID {
		t.Errorf("first listed = %q, want the most recently updated %q", list[0].ID, first.ID)
	}
	if list[1].ID != second.ID {
		t.Errorf("second listed = %q, want %q", list[1].ID, second.ID)
	}
}

func TestForkCopiesUpToTheCut(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("one"), msg("two"), msg("three")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	forked, err := s.Fork(ctx(), meta.ID, 2)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	if got := read(t, s, forked.ID); len(got) != 2 {
		t.Fatalf("fork holds %d entries, want 2", len(got))
	}
	if got := read(t, s, meta.ID); len(got) != 3 {
		t.Errorf("forking changed the original: %d entries", len(got))
	}
}

func TestForkRecordsWhereItCameFrom(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("one"), msg("two")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	forked, err := s.Fork(ctx(), meta.ID, 1)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	if forked.Parent != meta.ID {
		t.Errorf("fork's Parent = %q, want %q", forked.Parent, meta.ID)
	}
	if forked.ForkedAt != 1 {
		t.Errorf("fork's ForkedAt = %d, want 1", forked.ForkedAt)
	}
	if forked.ID == meta.ID {
		t.Error("a fork must be its own session")
	}
}

func TestDeleteIsIdempotent(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	if err := s.Delete(ctx(), meta.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := s.Delete(ctx(), meta.ID); err != nil {
		t.Errorf("deleting an absent session = %v, want nil", err)
	}
	if _, err := s.Meta(ctx(), meta.ID); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Meta after Delete = %v, want ErrNotFound", err)
	}
}

// Seq is the store's to assign, so concurrent writers must not collide on one.
func TestConcurrentAppendsKeepDistinctSequences(t *testing.T) {
	s := store(t)
	meta := create(t, s)

	const writers, each = 8, 12
	var wg sync.WaitGroup
	for w := range writers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range each {
				if err := s.Append(ctx(), meta.ID, msg(fmt.Sprintf("w%d-%d", w, i))); err != nil {
					t.Errorf("Append: %v", err)
					return
				}
			}
		}(w)
	}
	wg.Wait()

	entries := read(t, s, meta.ID)
	if len(entries) != writers*each {
		t.Fatalf("read %d entries, wrote %d — a concurrent append was lost", len(entries), writers*each)
	}
	seen := make(map[int64]bool, len(entries))
	for _, e := range entries {
		if seen[e.Seq] {
			t.Fatalf("sequence %d was assigned twice", e.Seq)
		}
		seen[e.Seq] = true
	}
}

func TestASessionSurvivesReopening(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("written before the restart")); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	entries := read(t, open(t, dir), meta.ID)
	if len(entries) != 1 {
		t.Fatalf("after reopening, session holds %d entries, want 1", len(entries))
	}
	if got := entries[0].Message.Text(); !strings.Contains(got, "before the restart") {
		t.Errorf("entry survived as %q", got)
	}
}

// A process killed mid-append leaves a half-written last line. Losing that one
// entry is the cost; losing the session is not acceptable.
func TestATornLastLineEndsTheSessionRatherThanFailingIt(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("first"), msg("second")); err != nil {
		t.Fatalf("Append: %v", err)
	}

	path := filepath.Join(dir, meta.ID, "entries.jsonl")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if err := os.WriteFile(path, raw[:len(raw)-12], 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got := read(t, s, meta.ID)
	if len(got) != 1 {
		t.Fatalf("read %d entries from a torn file, want the 1 complete one", len(got))
	}
	if got[0].Message.Text() != "first" {
		t.Errorf("surviving entry = %q", got[0].Message.Text())
	}
}

// meta.json is a cache of what the entries file already knows, and it is
// allowed to be behind — which means it is behind whenever a process died
// without closing the store. What must not happen is a second process
// carrying on from the stale number and writing entries that share a sequence
// with entries already there.
func TestASessionCarriesOnFromWhatWasWrittenNotFromTheMetadata(t *testing.T) {
	dir := t.TempDir()

	first := open(t, dir)
	meta, err := first.Create(ctx(), session.Meta{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range 5 {
		if err := first.Append(ctx(), meta.ID, msg(fmt.Sprintf("before %d", i))); err != nil {
			t.Fatal(err)
		}
	}
	// No Close: this is a process that died, so meta.json still says zero.
	onDisk := readMetaFile(t, filepath.Join(dir, meta.ID, "meta.json"))
	if onDisk.Entries != 0 {
		t.Logf("meta.json says %d entries — the test is weaker than intended", onDisk.Entries)
	}

	second := open(t, dir)
	if err := second.Append(ctx(), meta.ID, msg("after")); err != nil {
		t.Fatal(err)
	}

	var seqs []int64
	for e, err := range second.Entries(ctx(), meta.ID) {
		if err != nil {
			t.Fatal(err)
		}
		seqs = append(seqs, e.Seq)
	}
	if want := []int64{1, 2, 3, 4, 5, 6}; !slices.Equal(seqs, want) {
		t.Errorf("sequence numbers = %v, want %v — the new process reused numbers", seqs, want)
	}

	// And reading metadata through the store brings it up to date.
	got, err := second.Meta(ctx(), meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Entries != 6 {
		t.Errorf("Entries = %d, want 6", got.Entries)
	}
}

// An entry longer than the tail block lastSeq reads first must still be found.
func TestASessionCarriesOnPastAVeryLongEntry(t *testing.T) {
	dir := t.TempDir()
	first := open(t, dir)
	meta, err := first.Create(ctx(), session.Meta{})
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Append(ctx(), meta.ID, msg(strings.Repeat("x", 64*1024))); err != nil {
		t.Fatal(err)
	}

	second := open(t, dir)
	if err := second.Append(ctx(), meta.ID, msg("next")); err != nil {
		t.Fatal(err)
	}
	var last int64
	for e, err := range second.Entries(ctx(), meta.ID) {
		if err != nil {
			t.Fatal(err)
		}
		last = e.Seq
	}
	if last != 2 {
		t.Errorf("last Seq = %d, want 2", last)
	}
}

func readMetaFile(t *testing.T, path string) session.Meta {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m session.Meta
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func BenchmarkAppendOneEntry(b *testing.B) {
	dir := b.TempDir()
	s, err := jsonl.Open(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	meta, err := s.Create(context.Background(), session.Meta{})
	if err != nil {
		b.Fatal(err)
	}
	e := msg("a line of the sort a recorder writes")
	b.ReportAllocs()
	for b.Loop() {
		if err := s.Append(context.Background(), meta.ID, e); err != nil {
			b.Fatal(err)
		}
	}
}
