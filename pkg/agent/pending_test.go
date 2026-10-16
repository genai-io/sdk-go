package agent_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
)

func TestPendingInputIsAnnouncedBeforeAnUnfinishedTurnEnds(t *testing.T) {
	for _, cancelTurn := range []bool{false, true} {
		t.Run(map[bool]string{false: "failure", true: "cancellation"}[cancelTurn], func(t *testing.T) {
			a := newAgent(t, aitest.Always(aitest.Fails(&ai.Error{Kind: ai.KindAuth, Message: "invalid credentials"})))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			input := ai.Message{ID: "pending", Role: ai.RoleUser, Content: ai.TextContent("keep this follow-up")}
			var announced []ai.Message
			ended := false
			for event, err := range a.Run(ctx, ai.UserMessage("first")) {
				if err != nil {
					t.Fatal(err)
				}
				switch e := event.(type) {
				case agent.InferenceStart:
					a.AddMessages(input)
					if cancelTurn {
						cancel()
					}
				case agent.MessageAdded:
					announced = append(announced, e.Message)
				case agent.TurnEnd:
					ended = true
					if e.Err == nil || len(announced) != 2 || announced[1].ID != input.ID || a.Pending() != 0 {
						t.Fatalf("unfinished turn lost pending input: end=%+v, messages=%+v, pending=%d", e, announced, a.Pending())
					}
				}
			}
			if !ended || !reflect.DeepEqual(a.Messages(), announced) {
				t.Fatal("event history differs from the restorable conversation")
			}
		})
	}
}
