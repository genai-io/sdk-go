package session_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Compaction replaces the conversation, and a session that only folded what
// was appended would restore what the agent threw away. Nothing here says so:
// replacing the conversation is an event like any other, and the recorder
// consuming the stream already has it. That is the point of the test — a
// caller who forgets a step cannot exist if there is no step to forget.
func TestCompactionSurvivesARestore(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, text("one"), text("two"), text("three"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("first"), ai.UserMessage("second"))

	summary := []ai.Message{ai.UserMessage("(summary of the above)")}
	a.SetMessages(summary)

	converse(t, a, rec, ai.UserMessage("third"))

	restored, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(restored) != len(a.Messages()) {
		t.Fatalf("restored %d messages, agent holds %d — the compaction did not survive",
			len(restored), len(a.Messages()))
	}
	if got := restored[0].Text(); got != "(summary of the above)" {
		t.Errorf("the restore starts at %q, want the summary", got)
	}
}

// The shape both examples in this repository have: open a session, hand the
// agent whatever came back, answer. What comes back from a new session is
// nothing, and a second run still has to reopen what the first wrote.
func TestASessionSeededWithItsOwnEmptyHistoryReopens(t *testing.T) {
	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", store},
		{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
	} {
		t.Run(impl.name, func(t *testing.T) {
			st := impl.open(t)
			ctx := context.Background()

			rec, history, err := session.Open(ctx, st, "")
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			first := newAgent(t, nil, text("hello"))
			first.SetMessages(history) // what the examples do, verbatim
			converse(t, first, rec, ai.UserMessage("hi"))
			if err := rec.Err(); err != nil {
				t.Fatalf("recording failed: %v", err)
			}

			rec2, restored, err := session.Open(ctx, st, rec.ID())
			if err != nil {
				t.Fatalf("the session the first run wrote will not reopen: %v", err)
			}
			if len(restored) != 2 {
				t.Fatalf("restored %d messages, want the 2 that were recorded", len(restored))
			}

			second := newAgent(t, nil, text("still here"))
			second.SetMessages(restored)
			converse(t, second, rec2, ai.UserMessage("again"))

			all, err := session.Messages(ctx, st, rec.ID())
			if err != nil {
				t.Fatalf("Messages: %v", err)
			}
			if len(all) != 4 {
				t.Errorf("the session holds %d messages, want 4", len(all))
			}
		})
	}
}

// Clearing a conversation is a state a session has to hold: an empty snapshot
// says everything before it is gone, which is not a record that carries
// nothing. Both stores, because it is the wire format that loses the difference.
func TestAClearedConversationRestoresAsCleared(t *testing.T) {
	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", store},
		{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
	} {
		t.Run(impl.name, func(t *testing.T) {
			st := impl.open(t)
			ctx := context.Background()

			a := newAgent(t, nil, text("one"), text("two"))
			rec, _, err := session.Open(ctx, st, "")
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			converse(t, a, rec, ai.UserMessage("first"))

			a.SetMessages(nil)
			converse(t, a, rec, ai.UserMessage("starting over"))

			restored, err := session.Messages(ctx, st, rec.ID())
			if err != nil {
				t.Fatalf("Messages: %v", err)
			}
			if len(restored) != len(a.Messages()) {
				t.Fatalf("restored %d messages, the agent holds %d — what was cleared came back",
					len(restored), len(a.Messages()))
			}
			if got := restored[0].Text(); got != "starting over" {
				t.Errorf("the restore starts at %q, want the first thing said after the clearing", got)
			}
		})
	}
}
