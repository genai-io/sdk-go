package session_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

func TestResumeContinuesAfterAnIncompleteTurn(t *testing.T) {
	ctx := context.Background()
	st := store(t)
	meta, err := st.Create(ctx, session.Meta{})
	if err != nil {
		t.Fatal(err)
	}
	m := ai.UserMessage("interrupted")
	if err := st.Append(ctx, meta.ID, session.Entry{Turn: 3, Type: session.EntryMessage, Message: &m}); err != nil {
		t.Fatal(err)
	}
	w, history, err := session.Open(ctx, st, meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	w.RecordAgentEvent(ctx, agent.MessagesReplaced{Turn: 1, Messages: history})
	w.RecordAgentEvent(ctx, agent.MessageAdded{Turn: 1, Message: ai.UserMessage("continue")})
	if err := w.Err(); err != nil {
		t.Fatal(err)
	}
	var entries []session.Entry
	for e, err := range st.Entries(ctx, meta.ID) {
		if err != nil {
			t.Fatal(err)
		}
		entries = append(entries, e)
	}
	if len(entries) != 2 || entries[1].Turn != 4 {
		t.Fatalf("entries=%+v", entries)
	}
}
