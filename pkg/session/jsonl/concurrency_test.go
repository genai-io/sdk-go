package jsonl_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/session"
	"github.com/genai-io/sdk-go/pkg/session/jsonl"
)

func TestIndependentStoresAssignOneSequence(t *testing.T) {
	dir := t.TempDir()
	a, b := open(t, dir), open(t, dir)
	meta := create(t, a)
	var wg sync.WaitGroup
	for _, s := range []*jsonl.Store{a, b} {
		wg.Go(func() {
			for range 20 {
				if err := s.Append(ctx(), meta.ID, msg("text")); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
	assertSequences(t, a, meta.ID, 40)
	fresh := open(t, dir)
	got, err := fresh.Meta(ctx(), meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Entries != 40 || !got.UpdatedAt.After(meta.CreatedAt) {
		t.Fatalf("stale metadata: %+v", got)
	}
	list, err := fresh.List(ctx())
	if err != nil || len(list) != 1 || list[0].Entries != 40 {
		t.Fatalf("stale listing: %+v, %v", list, err)
	}
}

func assertSequences(t *testing.T, s *jsonl.Store, id string, want int) {
	t.Helper()
	entries := read(t, s, id)
	if len(entries) != want {
		t.Fatalf("entries=%d, want %d", len(entries), want)
	}
	for i, e := range entries {
		if e.Seq != int64(i+1) {
			t.Fatalf("entry %d sequence=%d", i, e.Seq)
		}
	}
}

func TestAppendProcesses(t *testing.T) {
	if dir := os.Getenv("SDK_SESSION_TEST_DIR"); dir != "" {
		s := open(t, dir)
		for range 20 {
			if err := s.Append(ctx(), "shared", msg("process")); err != nil {
				t.Fatal(err)
			}
		}
		return
	}
	dir := t.TempDir()
	s := open(t, dir)
	if _, err := s.Create(ctx(), session.Meta{ID: "shared"}); err != nil {
		t.Fatal(err)
	}
	const processes = 3
	cmds := make([]*exec.Cmd, processes)
	output := make([]bytes.Buffer, processes)
	for i := range processes {
		cmds[i] = exec.Command(os.Args[0], "-test.run=^TestAppendProcesses$")
		cmds[i].Env = append(os.Environ(), "SDK_SESSION_TEST_DIR="+dir)
		cmds[i].Stdout = &output[i]
		cmds[i].Stderr = &output[i]
		if err := cmds[i].Start(); err != nil {
			t.Fatal(err)
		}
	}
	for i, cmd := range cmds {
		if err := cmd.Wait(); err != nil {
			t.Fatalf("process: %v, %s", err, output[i].String())
		}
	}
	assertSequences(t, s, "shared", processes*20)
}

func TestFailedBatchDoesNotConsumeASequence(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	bad := session.Entry{Type: session.EntryCustom, Custom: &session.Custom{Kind: "bad", Data: []byte("invalid json")}}
	if err := s.Append(ctx(), meta.ID, msg("uncommitted"), bad); err == nil {
		t.Fatal("invalid JSON accepted")
	}
	if err := s.Append(ctx(), meta.ID, msg("committed")); err != nil {
		t.Fatal(err)
	}
	assertSequences(t, s, meta.ID, 1)
	if got := read(t, s, meta.ID)[0].Message.Text(); got != "committed" {
		t.Fatal(got)
	}
}

func TestCompleteTailWithoutNewlineIsPreserved(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	meta := create(t, s)
	if err := s.Append(ctx(), meta.ID, msg("first")); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, meta.ID, "entries.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, bytes.TrimSuffix(data, []byte{'\n'}), 0600); err != nil {
		t.Fatal(err)
	}
	if err := open(t, dir).Append(ctx(), meta.ID, msg("second")); err != nil {
		t.Fatal(err)
	}
	assertSequences(t, s, meta.ID, 2)
}

func TestTerminatedCorruptTailIsReported(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	meta := create(t, s)
	path := filepath.Join(dir, meta.ID, "entries.jsonl")
	if err := os.WriteFile(path, []byte("{broken}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Append(ctx(), meta.ID, msg("after")); err == nil {
		t.Fatal("appended past corruption")
	}
	for _, err := range s.Entries(ctx(), meta.ID) {
		if err == nil {
			t.Fatal("corruption hidden")
		}
		return
	}
	t.Fatal("corruption hidden")
}

func TestCanceledAppendDoesNotWrite(t *testing.T) {
	s := store(t)
	meta := create(t, s)
	canceled, cancel := context.WithCancel(ctx())
	cancel()
	if err := s.Append(canceled, meta.ID, msg("canceled")); !errors.Is(err, context.Canceled) {
		t.Fatalf("error=%v", err)
	}
	assertSequences(t, s, meta.ID, 0)
}
