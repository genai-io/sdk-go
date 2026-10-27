package session_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

func BenchmarkReplay(b *testing.B) {
	store, err := session.Open(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	writer, err := store.Writer(context.Background(), "large")
	if err != nil {
		b.Fatal(err)
	}
	if err := writer.Save([]ai.Message{ai.UserMessage("prompt")}); err != nil {
		b.Fatal(err)
	}
	for range 40 {
		if err := writer.Append("audit", strings.Repeat("x", 256<<10)); err != nil {
			b.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		b.Fatal(err)
	}
	for _, write := range []bool{false, true} {
		b.Run(fmt.Sprintf("writer=%t", write), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if write {
					w, err := store.Writer(context.Background(), "large")
					if err != nil {
						b.Fatal(err)
					}
					if err := w.Close(); err != nil {
						b.Fatal(err)
					}
				} else {
					if _, err := store.Load(context.Background(), "large"); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

func TestLargeRecordsSurviveBufferReuseAndTailRecovery(t *testing.T) {
	store, dir := open(t)
	path := filepath.Join(dir, "large.jsonl")
	var data []byte
	for i := range 4 {
		raw, err := json.Marshal(session.Record{Type: session.TypeMessage, Message: new(msg(fmt.Sprint(i), strings.Repeat(fmt.Sprint(i), 128<<10)))})
		if err != nil {
			t.Fatal(err)
		}
		data = append(data, raw...)
		data = append(data, '\n')
	}
	complete := len(data)
	data = append(data, strings.Repeat("x", 300<<10)...)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	messages, err := store.Load(ctx, "large")
	if err != nil || len(messages) != 4 {
		t.Fatalf("Load: count=%d err=%v", len(messages), err)
	}
	for i, m := range messages {
		if m.Text() != strings.Repeat(fmt.Sprint(i), 128<<10) {
			t.Fatal("scanner buffer changed a prior message")
		}
	}
	writer, err := store.Writer(ctx, "large")
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(path)
	if err != nil || string(after) != string(data[:complete]) {
		t.Fatalf("tail recovery changed complete records: %v", err)
	}
}

func TestRecordsHonoursCancellationBetweenRecords(t *testing.T) {
	store, _ := open(t)
	writer := write(t, store, "cancel")
	if err := writer.Save([]ai.Message{msg("one", "first"), msg("two", "second")}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	count := 0
	for _, err := range store.Records(ctx, "cancel") {
		if count == 0 {
			if err != nil {
				t.Fatal(err)
			}
			cancel()
		} else if !errors.Is(err, context.Canceled) {
			t.Fatalf("iteration after cancel: %v", err)
		}
		count++
	}
	if count != 2 {
		t.Fatalf("iterator yielded %d times", count)
	}
}

func TestOpeningCompleteLogPreservesModificationTime(t *testing.T) {
	store, dir := open(t)
	writer := write(t, store, "unchanged")
	if err := writer.Save([]ai.Message{msg("one", "prompt")}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "unchanged.jsonl")
	before := time.Unix(1700000000, 0)
	if err := os.Chtimes(path, before, before); err != nil {
		t.Fatal(err)
	}
	writer, err := store.Writer(ctx, "unchanged")
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !after.ModTime().Equal(before) {
		t.Fatal("opening an intact log invalidated metadata keyed by its modification time")
	}
}
