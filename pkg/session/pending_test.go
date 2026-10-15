package session_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
	"github.com/genai-io/sdk-go/pkg/session"
)

func TestPendingInputSurvivesFailedSessionResume(t *testing.T) {
	ctx := context.Background()
	st := jsonlStore(t)
	rec, err := session.Create(ctx, st, session.Meta{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = rec.Close() })
	a := newAgent(t, nil, aitest.Fails(&ai.Error{Kind: ai.KindAuth, Message: "invalid credentials"}))
	for event, err := range a.Run(ctx, ai.UserMessage("first")) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(agent.MessageStart); ok {
			a.AddMessages(ai.Message{ID: "pending", Role: ai.RoleUser, Content: ai.TextContent("follow-up")})
		}
		rec.RecordEvent(ctx, event)
	}
	if err := rec.Err(); err != nil {
		t.Fatal(err)
	}
	if err := rec.Close(); err != nil {
		t.Fatal(err)
	}
	messages, err := messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 || messages[1].ID != "pending" {
		t.Fatalf("resume lost accepted input: %+v", messages)
	}
}
