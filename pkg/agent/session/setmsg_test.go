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
