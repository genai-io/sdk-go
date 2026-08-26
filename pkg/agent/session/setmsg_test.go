package session_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Compaction replaces the conversation, and a session that only folded what was
// appended would restore what the agent threw away. A snapshot is the reset the
// fold needs: everything before one is what the agent no longer has.
func TestCompactionSurvivesARestore(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, text("one"), text("two"), text("three"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("first"), ai.UserMessage("second"))

	// Compaction replaces the conversation, and the caller who replaced it is
	// the one who knows: it tells the session in the same breath.
	summary := []ai.Message{ai.UserMessage("(summary of the above)")}
	a.SetMessages(summary)
	rec.Snapshot(summary)

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
