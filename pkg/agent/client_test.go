package agent_test

import (
	"context"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Where a call goes: the client the agent holds, and the one inference a hook
// sends somewhere else.

func stubClient(id string, d ai.Driver) *ai.Client {
	return ai.NewClientWithDriver(d, ai.Model{ID: id, API: "stub"})
}

// SetClient is a person switching model mid-session. Everything else about the
// agent — the conversation, the prompt, the tools — is what it was.
func TestSetClientRedirectsTheNextInference(t *testing.T) {
	first := &scripted{Scripts: [][]ai.Delta{text("first")}}
	second := &scripted{Scripts: [][]ai.Delta{text("second")}}
	a := newAgent(t, first)

	if out, err := outcome(t, a, ai.UserMessage("one")); err != nil || out.Message.Text() != "first" {
		t.Fatalf("first turn = %q, %v", out.Message.Text(), err)
	}

	a.SetClient(nil) // ignored: an agent without a client can do nothing
	a.SetClient(stubClient("second", second))

	out, err := outcome(t, a, ai.UserMessage("two"))
	if err != nil {
		t.Fatalf("second turn: %v", err)
	}
	if got := out.Message.Text(); got != "second" {
		t.Errorf("answer = %q, want the new client's", got)
	}
	if first.Calls() != 1 || second.Calls() != 1 {
		t.Errorf("calls = %d and %d, want one each", first.Calls(), second.Calls())
	}
	// And the agent says which model it is calling, not which it was built on.
	if got := a.String(); !strings.Contains(got, "second") {
		t.Errorf("String = %q, want the model it is calling now", got)
	}
}

// A hook routes one inference and no more: the call after it goes back to the
// agent's own client, because pointing a call somewhere is not moving the
// agent there.
func TestAHookRoutesOneInferenceWithoutMovingTheAgent(t *testing.T) {
	home := &scripted{Scripts: [][]ai.Delta{text("home")}}
	away := &scripted{Scripts: [][]ai.Delta{text("away")}}
	elsewhere := stubClient("away", away)

	once := true
	a := newAgent(t, home, agent.WithHooks(agent.Hook{
		PreInfer: func(_ context.Context, inf *agent.Inference) error {
			if once {
				once = false
				inf.Client = elsewhere
			}
			return nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("one"))
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	// The routing is on the stream, so a consumer knows which endpoint the
	// answer it is drawing came from.
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok {
			if got := v.Inference.Client.Model().ID; got != "away" {
				t.Errorf("MessageStart routed to %q, want the hook's client", got)
			}
		}
	}

	second, err := outcome(t, a, ai.UserMessage("two"))
	if err != nil {
		t.Fatalf("second turn: %v", err)
	}
	if got := second.Message.Text(); got != "home" {
		t.Errorf("second answer = %q, want the agent's own client", got)
	}
	if home.Calls() != 1 || away.Calls() != 1 {
		t.Errorf("calls = %d and %d, want one each", home.Calls(), away.Calls())
	}
	if got := a.Client().Model().ID; got != "stub" {
		t.Errorf("the agent is now on %q, want it where it started", got)
	}
}

// Every attempt is built and vetted again, so a retry may be sent where the
// attempt before it was not — which is what makes a fallback endpoint a hook
// and not a second loop.
func TestARetryCanBeRoutedToAnotherClient(t *testing.T) {
	primary := &scripted{
		Scripts: [][]ai.Delta{nil},
		Errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
	}
	fallback := &scripted{Scripts: [][]ai.Delta{text("the other endpoint")}}
	spare := stubClient("spare", fallback)

	asked := 0
	a := newAgent(t, primary, agent.WithRetry(2, 0), agent.WithHooks(agent.Hook{
		PreInfer: func(_ context.Context, inf *agent.Inference) error {
			asked++
			if asked > 1 { // the first attempt failed; try elsewhere
				inf.Client = spare
			}
			return nil
		},
	}))

	out, err := outcome(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if got := out.Message.Text(); got != "the other endpoint" {
		t.Errorf("answer = %q, want the fallback's", got)
	}
	if primary.Calls() != 1 || fallback.Calls() != 1 {
		t.Errorf("calls = %d and %d, want the failure and then the retry",
			primary.Calls(), fallback.Calls())
	}
}
