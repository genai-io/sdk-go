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
	rec, err := session.Create(ctx, st)
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
	got, err := st.Load(ctx, rec.ID())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[1].ID != "pending" {
		t.Fatalf("resume lost accepted input: %+v", got)
	}
}
