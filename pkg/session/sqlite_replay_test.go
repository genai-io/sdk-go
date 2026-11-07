package session

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestBackendsNativeContent(t *testing.T) {
	image := ai.Image{MediaType: "image/png", Data: "aW1hZ2U=", FileName: "native.png"}
	msgs := []ai.Message{
		ai.UserMessage(strings.Repeat("large <&>\n", 32768), image),
		{ID: "assistant", Role: ai.RoleAssistant, Content: ai.Content{
			ai.ThinkingBlock("thought", "signature"),
			ai.ReasoningBlock(ai.ReasoningItem{ID: "reason", API: "openai-responses", EncryptedContent: "opaque", Summary: "summary"}),
			ai.ToolCallBlock(ai.ToolCall{ID: "call", Name: "tool", Input: `{ "x": 1 }`, Signature: []byte{0, 1, 255}}),
			ai.TextBlock("answer"),
		}},
		ai.ToolResultsMessage(ai.ToolResult{ToolCallID: "call", ToolName: "tool", IsError: true, Content: ai.Content{ai.TextBlock("result"), ai.ImageBlock(image)}}),
	}
	for _, sqlite := range []bool{false, true} {
		t.Run(fmt.Sprintf("sqlite=%t", sqlite), func(t *testing.T) {
			s := backend(t, sqlite)
			w, err := s.Writer(context.Background(), "native")
			must(t, err)
			must(t, w.Save(msgs))
			must(t, w.Save(msgs[:1]))
			must(t, w.Save(msgs))
			must(t, w.Close())
			got, err := s.Load(context.Background(), "native")
			must(t, err)
			if !reflect.DeepEqual(got, msgs) {
				t.Fatal("native content changed on replay")
			}
			records := collect(t, s, "native")
			if len(records) != 6 || records[3].Type != TypeTruncate {
				t.Fatal("lost historical truncate")
			}
			must(t, s.Fork(context.Background(), "native", "copy"))
			if !reflect.DeepEqual(records, collect(t, s, "copy")) {
				t.Fatal("fork changed native historical content")
			}
		})
	}
}

func TestSQLiteForkRollback(t *testing.T) {
	s := backend(t, true)
	ctx := context.Background()
	w, err := s.Writer(ctx, "source")
	must(t, err)
	defer w.Close()
	must(t, w.Save([]ai.Message{ai.UserMessage("a"), ai.UserMessage("b")}))
	_, err = s.sql.db.Exec(`CREATE TRIGGER fail_fork BEFORE INSERT ON session_events WHEN NEW.session_id='copy' AND NEW.seq=1 BEGIN SELECT RAISE(ABORT,'injected fork failure'); END`)
	must(t, err)
	if err = s.Fork(ctx, "source", "copy"); err == nil {
		t.Fatal("fork unexpectedly succeeded")
	}
	if _, err = s.Stat(ctx, "copy"); err != ErrNotFound {
		t.Fatalf("partial fork exists: %v", err)
	}
	_, err = s.sql.db.Exec("DROP TRIGGER fail_fork")
	must(t, err)
	must(t, s.Fork(ctx, "source", "copy"))
	if !reflect.DeepEqual(collect(t, s, "source"), collect(t, s, "copy")) {
		t.Fatal("fork lost records")
	}
}

func TestSQLiteRecordsCancellation(t *testing.T) {
	s := backend(t, true)
	w, err := s.Writer(context.Background(), "a")
	must(t, err)
	defer w.Close()
	must(t, w.Save([]ai.Message{ai.UserMessage("a"), ai.UserMessage("b")}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	count := 0
	for _, err := range s.Records(ctx, "a") {
		if count == 0 {
			must(t, err)
			cancel()
		} else if err != context.Canceled {
			t.Fatalf("got %v", err)
		}
		count++
	}
	if count != 2 {
		t.Fatal(count)
	}
}
