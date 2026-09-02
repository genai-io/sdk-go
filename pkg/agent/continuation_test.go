package agent_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

func cutOff(s string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.TextBlock(s)},
		{EndBlock: true},
		{StopReason: ai.StopMaxTokens},
	}
}

// A model stopped by the output cap was interrupted, not finished. Asked to,
// the loop takes another step in the same exchange rather than handing back
// half an answer.
func TestACutOffAnswerIsResumedWhenAsked(t *testing.T) {
	d := &scripted{Scripts: [][]ai.Delta{
		cutOff("the first half"),
		text("the second half"),
	}}
	a := newAgent(t, d, agent.WithContinuation(2, "carry on"))

	out, err := outcome(t, a, ai.UserMessage("write at length"))
	if err != nil {
		t.Fatal(err)
	}
	if out.StopReason != agent.StopEndTurn {
		t.Errorf("StopReason = %q, want end_turn — it finished on the second step", out.StopReason)
	}
	if got := out.Message.Text(); got != "the second half" {
		t.Errorf("Message = %q, want the resumed half", got)
	}
	if n := turns(t, a); n != 1 {
		t.Errorf("it took %d exchanges, want 1 — resuming is a step, not a turn", n)
	}

	// The prompt is in the conversation, so a session records what was asked.
	msgs := a.Messages()
	var asked bool
	for _, m := range msgs {
		if m.Text() == "carry on" {
			asked = true
		}
	}
	if !asked {
		t.Error("the continuation prompt never entered the conversation")
	}
}

// Off unless asked for: paying for more tokens is the application's call.
func TestACutOffAnswerStopsWhenNotAsked(t *testing.T) {
	d := &scripted{Scripts: [][]ai.Delta{cutOff("the first half"), text("never reached")}}
	a := newAgent(t, d)

	out, err := outcome(t, a, ai.UserMessage("write at length"))
	if err != nil {
		t.Fatal(err)
	}
	if out.StopReason != agent.StopMaxTokens {
		t.Errorf("StopReason = %q, want max_tokens", out.StopReason)
	}
	if n := d.Calls(); n != 1 {
		t.Errorf("the model was called %d times, want 1", n)
	}
}

// Running out of attempts is still a cut-off answer, and says so.
func TestResumingGivesUpAndStillSaysItWasCutOff(t *testing.T) {
	d := &scripted{Scripts: [][]ai.Delta{
		cutOff("one"), cutOff("two"), cutOff("three"), cutOff("four"),
	}}
	a := newAgent(t, d, agent.WithContinuation(2, "carry on"))

	out, err := outcome(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatal(err)
	}
	if out.StopReason != agent.StopMaxTokens {
		t.Errorf("StopReason = %q, want max_tokens", out.StopReason)
	}
	if n := d.Calls(); n != 3 {
		t.Errorf("the model was called %d times, want 3 — one plus two resumes", n)
	}
}

// turns counts the exchanges an agent has held, by running one more and
// reading the number it reports.
func turns(t *testing.T, a *agent.Agent) int {
	t.Helper()
	var n int
	for e, err := range a.Run(context.Background(), ai.UserMessage("ping")) {
		if err != nil {
			return n
		}
		if v, ok := e.(agent.TurnEnd); ok {
			n = v.Turn - 1 // this probe is the one after
		}
	}
	return n
}
