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
	w.RecordEvent(ctx, agent.MessagesReplaced{Turn: 1, Messages: history})
	w.RecordEvent(ctx, agent.MessageAdded{Turn: 1, Message: ai.UserMessage("continue")})
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

func TestResumeSnapshotDoesNotAliasReturnedMessages(t *testing.T) {
	for _, nested := range []bool{false, true} {
		t.Run(map[bool]string{false: "replace_message", true: "edit_content"}[nested], func(t *testing.T) {
			ctx := context.Background()
			st := jsonlStore(t)
			s, err := session.Create(ctx, st, session.Meta{})
			if err != nil {
				t.Fatal(err)
			}
			msg := ai.UserMessage("before")
			if err := s.Append(ctx, session.Entry{Type: session.EntryMessage, Message: &msg}); err != nil {
				t.Fatal(err)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			opened, history, err := session.Open(ctx, st, s.ID())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = opened.Close() }()
			if nested {
				history[0].Content[0].Text = "edited"
			} else {
				history[0] = ai.UserMessage("edited")
			}
			opened.RecordEvent(ctx, agent.MessagesReplaced{Turn: 1, Messages: history})
			if err := opened.Err(); err != nil {
				t.Fatal(err)
			}
			got, err := messages(ctx, st, s.ID())
			if err != nil {
				t.Fatal(err)
			}
			if got[0].Text() != "edited" {
				t.Fatalf("replacement lost: %q", got[0].Text())
			}
		})
	}
}

func TestResumeDeduplicationEndsAfterAppendingMessage(t *testing.T) {
	ctx := context.Background()
	st := jsonlStore(t)
	s, err := session.Create(ctx, st, session.Meta{})
	if err != nil {
		t.Fatal(err)
	}
	msg := ai.UserMessage("original")
	if err := s.Append(ctx, session.Entry{Type: session.EntryMessage, Message: &msg}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	opened, history, err := session.Open(ctx, st, s.ID())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = opened.Close() }()
	msg = ai.UserMessage("discard this")
	if err := opened.Append(ctx, session.Entry{Type: session.EntryMessage, Message: &msg}); err != nil {
		t.Fatal(err)
	}
	if err := opened.Append(ctx, session.Entry{Type: session.EntryConversationReplaced, Messages: history}); err != nil {
		t.Fatal(err)
	}
	got, err := messages(ctx, st, s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Text() != "original" {
		t.Fatalf("replacement ignored: %+v", got)
	}
}
