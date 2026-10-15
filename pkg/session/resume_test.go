package session_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

func TestResumeSnapshotDoesNotAliasReturnedMessages(t *testing.T) {
	for _, nested := range []bool{false, true} {
		t.Run(map[bool]string{false: "replace_message", true: "edit_content"}[nested], func(t *testing.T) {
			ctx := context.Background()
			st := jsonlStore(t)
			s, err := session.Create(ctx, st)
			if err != nil {
				t.Fatal(err)
			}
			s.RecordEvent(ctx, agent.MessageAdded{Message: ai.UserMessage("before")})
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
			opened.RecordEvent(ctx, agent.MessagesReplaced{Messages: history})
			if err := opened.Err(); err != nil {
				t.Fatal(err)
			}
			got, err := st.Load(ctx, s.ID())
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
	s, err := session.Create(ctx, st)
	if err != nil {
		t.Fatal(err)
	}
	s.RecordEvent(ctx, agent.MessageAdded{Message: ai.UserMessage("original")})
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	opened, history, err := session.Open(ctx, st, s.ID())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = opened.Close() }()
	opened.RecordEvent(ctx, agent.MessageAdded{Message: ai.UserMessage("discard this")})
	opened.RecordEvent(ctx, agent.MessagesReplaced{Messages: history})
	if err := opened.Err(); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load(ctx, s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Text() != "original" {
		t.Fatalf("replacement ignored: %+v", got)
	}
}
