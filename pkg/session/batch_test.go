package session

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func batchFixture(t *testing.T) (*Store, *Writer) {
	t.Helper()
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	w, err := s.Writer(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	return s, w
}
func appBatch(id string, tail Tail) Batch {
	return Batch{ID: id, Expected: tail, Records: []Record{{Type: TypeApp, Name: "custom.event", Data: json.RawMessage(`{"n":1}`)}, {Type: TypeApp, Name: "another", Data: json.RawMessage(`null`)}}}
}

func TestCommitRetryAndModes(t *testing.T) {
	ctx := context.Background()
	s, w := batchFixture(t)
	b := appBatch("one", Tail{})
	r, err := w.Commit(ctx, b)
	if err != nil {
		t.Fatal(err)
	}
	if r.Tail.Count != 2 || r.Tail.Sequence != 1 || r.Tail.Hash == "" {
		t.Fatal(r)
	}
	b2 := appBatch("two", r.Tail)
	r2, err := w.Commit(ctx, b2)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := w.Commit(ctx, b); err != nil || got != r {
		t.Fatalf("stale retry: %v %v", got, err)
	}
	conflict := b
	conflict.Expected = r.Tail
	if _, err := w.Commit(ctx, conflict); !errors.Is(err, ErrConflict) {
		t.Fatal(err)
	}
	conflict = appBatch("one", Tail{})
	conflict.Records[0].Data = json.RawMessage(`2`)
	if _, err := w.Commit(ctx, conflict); !errors.Is(err, ErrConflict) {
		t.Fatal(err)
	}
	if _, err := w.Commit(ctx, appBatch("three", Tail{})); !errors.Is(err, ErrTail) {
		t.Fatal(err)
	}
	if err := w.Save(nil); !errors.Is(err, ErrMode) {
		t.Fatal(err)
	}
	if err := w.Append("x", 1); !errors.Is(err, ErrMode) {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	w, err = s.Writer(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	if got, err := w.Commit(ctx, b); err != nil || got != r {
		t.Fatalf("reopen retry: %v %v", got, err)
	}
	if tail, err := w.Tail(); err != nil || tail != r2.Tail {
		t.Fatal(tail, err)
	}
	if err := s.Fork(ctx, "a", "fork"); err != nil {
		t.Fatal(err)
	}
	fork, err := s.Writer(ctx, "fork")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = fork.Close() }()
	if err := fork.Save(nil); !errors.Is(err, ErrMode) {
		t.Fatal(err)
	}
	if _, err := s.Writer(ctx, "a"); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	if err := s.Delete(ctx, "a"); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
}

func TestCommitEveryTruncation(t *testing.T) {
	ctx := context.Background()
	s, w := batchFixture(t)
	r, err := w.Commit(ctx, appBatch("one", Tail{}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Commit(ctx, appBatch("two", r.Tail)); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(s.dir, "a.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	boundary := bytes.IndexByte(data, '\n') + 1
	for cut := 0; cut <= len(data); cut++ {
		if err := os.WriteFile(path, data[:cut], 0600); err != nil {
			t.Fatal(err)
		}
		want := 0
		if cut >= boundary {
			want = 2
		}
		if cut == len(data) {
			want = 4
		}
		n := 0
		for _, err := range s.Records(ctx, "a") {
			if err != nil {
				t.Fatalf("cut %d: %v", cut, err)
			}
			n++
		}
		if n != want {
			t.Fatalf("cut %d: got %d want %d", cut, n, want)
		}
		w, err := s.Writer(ctx, "a")
		if err != nil {
			t.Fatalf("cut %d: %v", cut, err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		stat, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		expected := 0
		if want == 2 {
			expected = boundary
		}
		if want == 4 {
			expected = len(data)
		}
		if stat.Size() != int64(expected) {
			t.Fatalf("cut %d recovered size %d", cut, stat.Size())
		}
	}
}

func TestBatchCorruptionIsNotRecovered(t *testing.T) {
	ctx := context.Background()
	s, w := batchFixture(t)
	r, err := w.Commit(ctx, appBatch("one", Tail{}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Commit(ctx, appBatch("two", r.Tail)); err != nil {
		t.Fatal(err)
	}
	_ = w.Close()
	path := filepath.Join(s.dir, "a.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string][]byte{
		"payload":     bytes.Replace(data, []byte(`"n":1`), []byte(`"n":2`), 1),
		"version":     bytes.Replace(data, []byte(`"batch_version":1`), []byte(`"batch_version":2`), 1),
		"malformed":   append([]byte("{bad}\n"), data...),
		"mixedBefore": append([]byte("{\"type\":\"app\",\"name\":\"x\"}\n"), data...),
		"mixedAfter":  append(bytes.Clone(data), []byte("{\"type\":\"app\",\"name\":\"x\"}\n")...),
		"chain":       bytes.Replace(data, []byte(`"count":2`), []byte(`"count":3`), 1),
	}
	for name, broken := range tests {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(path, broken, 0600); err != nil {
				t.Fatal(err)
			}
			found := false
			for _, err := range s.Records(ctx, "a") {
				if err != nil {
					found = true
					break
				}
			}
			if !found {
				t.Fatal("reader accepted corruption")
			}
			if w, err := s.Writer(ctx, "a"); err == nil {
				_ = w.Close()
				t.Fatal("writer recovered corruption")
			}
			actual, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(actual, broken) {
				t.Fatal("corruption truncated")
			}
		})
	}
}

func TestBatchConcurrentReaders(t *testing.T) {
	ctx := context.Background()
	s, w := batchFixture(t)
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 80 {
				n := 0
				for r, err := range s.Records(ctx, "a") {
					if err != nil {
						t.Error(err)
						return
					}
					if r.Type != TypeApp {
						t.Error(r.Type)
					}
					n++
				}
				if n%2 != 0 {
					t.Errorf("partial batch: %d records", n)
				}
				for b, err := range s.Batches(ctx, "a") {
					if err != nil {
						t.Error(err)
						return
					}
					if len(b.Batch.Records) != 2 {
						t.Error("partial batch")
					}
				}
			}
		}()
	}
	var tail Tail
	for i := range 40 {
		r, err := w.Commit(ctx, appBatch(fmt.Sprint(i), tail))
		if err != nil {
			t.Fatal(err)
		}
		tail = r.Tail
	}
	wg.Wait()
}

func TestCommitContextOwnershipAndFailure(t *testing.T) {
	ctx := context.Background()
	s, w := batchFixture(t)
	b := appBatch("one", Tail{})
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := w.Commit(canceled, b); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	r, err := w.Commit(ctx, b)
	if err != nil {
		t.Fatal(err)
	}
	b.Records[0].Name = "mutated"
	if got, err := w.Commit(ctx, appBatch("one", Tail{})); err != nil || got != r {
		t.Fatal(got, err)
	}
	// Retry must actually Sync; a closed descriptor makes that fail.
	if err := w.f.Close(); err != nil {
		t.Fatal(err)
	}
	_, failure := w.Commit(ctx, appBatch("one", Tail{}))
	if !errors.Is(failure, os.ErrClosed) {
		t.Fatal(failure)
	}
	if _, err := w.Commit(ctx, appBatch("two", r.Tail)); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	_ = w.Close()
	w, err = s.Writer(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	if got, err := w.Commit(ctx, appBatch("one", Tail{})); err != nil || got != r {
		t.Fatal(got, err)
	}
	// Unknown records and invalid truncations must not poison a writer.
	invalid := Batch{ID: "invalid", Expected: r.Tail, Records: []Record{{Type: "future"}}}
	if _, err := w.Commit(ctx, invalid); err == nil {
		t.Fatal("accepted unknown type")
	}
	invalid.Records = []Record{{Type: TypeTruncate, Keep: 1}}
	if _, err := w.Commit(ctx, invalid); err == nil {
		t.Fatal("accepted invalid truncate")
	}
	invalid.Records = []Record{{Type: TypeMessage, Message: &ai.Message{}}, {Type: TypeTruncate}}
	if _, err := w.Commit(ctx, invalid); err != nil {
		t.Fatal(err)
	}
	if msgs, err := s.Load(ctx, "a"); err != nil || len(msgs) != 0 {
		t.Fatal(msgs, err)
	}
}

func TestCommitPostWriteSyncFailure(t *testing.T) {
	_, w := batchFixture(t)
	r, pipe, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close(); _ = pipe.Close() }()
	original := w.f
	w.f = pipe // Writes succeed, but pipes cannot be synced.
	receipt, failure := w.Commit(context.Background(), appBatch("one", Tail{}))
	w.f = original
	if failure == nil || receipt != (Receipt{}) {
		t.Fatalf("acknowledged failed sync: %v %v", receipt, failure)
	}
	buf := make([]byte, 4096)
	n, err := r.Read(buf)
	if err != nil || !bytes.HasSuffix(buf[:n], []byte("\n")) {
		t.Fatalf("write did not complete before sync failed: %d %v", n, err)
	}
	if _, err := w.Commit(context.Background(), appBatch("one", Tail{})); !errors.Is(err, failure) {
		t.Fatalf("not sticky: %v", err)
	}
}

// Cancel exactly at the post-sync checkpoint, without timing-dependent races.
type cancelAfterWrite struct {
	context.Context
	checks int
}

func (c *cancelAfterWrite) Err() error {
	c.checks++
	if c.checks >= 3 {
		return context.Canceled
	}
	return nil
}

func TestCommitPostWriteCancellation(t *testing.T) {
	s, w := batchFixture(t)
	b := appBatch("one", Tail{})
	ctx := &cancelAfterWrite{Context: context.Background()}
	if receipt, err := w.Commit(ctx, b); !errors.Is(err, context.Canceled) || receipt != (Receipt{}) {
		t.Fatal(receipt, err)
	}
	if _, err := w.Commit(context.Background(), b); !errors.Is(err, context.Canceled) {
		t.Fatalf("not sticky: %v", err)
	}
	n := 0
	for _, err := range s.Batches(context.Background(), "a") {
		if err != nil {
			t.Fatal(err)
		}
		n++
	}
	if n != 1 {
		t.Fatal("cancellation claimed rollback")
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	w, err := s.Writer(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	if receipt, err := w.Commit(context.Background(), b); err != nil || receipt.Tail.Sequence != 1 {
		t.Fatal(receipt, err)
	}
}

func TestLegacyBatchMode(t *testing.T) {
	ctx := context.Background()
	s, w := batchFixture(t)
	if err := w.Append("old", 1); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Commit(ctx, appBatch("one", Tail{})); !errors.Is(err, ErrMode) {
		t.Fatal(err)
	}
	if _, err := w.Tail(); !errors.Is(err, ErrMode) {
		t.Fatal(err)
	}
	for _, err := range s.Batches(ctx, "a") {
		if !errors.Is(err, ErrMode) {
			t.Fatal(err)
		}
	}
	for _, r := range []Record{{Type: TypeMessage}, {Type: TypeTruncate, Keep: -1}} {
		if _, err := r.Apply(nil); err == nil {
			t.Fatal("invalid Apply succeeded")
		}
	}
}
