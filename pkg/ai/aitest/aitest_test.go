package aitest_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
)

func TestTheTurnsArePlayedInOrder(t *testing.T) {
	d := aitest.New(aitest.Says("first"), aitest.Says("second"))
	c := d.Client()

	for _, want := range []string{"first", "second"} {
		resp, err := c.Complete(context.Background(), []ai.Message{ai.UserMessage("go")})
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if got := resp.Text(); got != want {
			t.Errorf("text = %q, want %q", got, want)
		}
	}
	if d.Calls() != 2 {
		t.Errorf("Calls = %d, want 2", d.Calls())
	}
}

// The failure a hand-written double kept getting wrong: a test that provoked
// one more call than it wrote for used to see an empty answer, which reads as
// the model choosing to say nothing.
func TestRunningOutOfTurnsIsAnError(t *testing.T) {
	d := aitest.New(aitest.Says("only one"))
	c := d.Client()

	if _, err := c.Complete(context.Background(), []ai.Message{ai.UserMessage("go")}); err != nil {
		t.Fatalf("first call: %v", err)
	}
	_, err := c.Complete(context.Background(), []ai.Message{ai.UserMessage("again")})
	if err == nil {
		t.Fatal("a second call with no turn written for it succeeded")
	}
}

func TestAToolCallStopsForTheLoopToRunIt(t *testing.T) {
	d := aitest.New(aitest.Asks(ai.ToolCall{ID: "c1", Name: "read", Input: `{"path":"x"}`}))

	resp, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("go")})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != ai.StopToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, ai.StopToolUse)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "read" {
		t.Fatalf("ToolCalls = %+v, want one call to read", calls)
	}
}

// A response saying end_turn while carrying a tool call is a stream no
// provider produces, and one the loop would read as "nothing left to do".
func TestAskingForAToolOverridesAContradictoryStopReason(t *testing.T) {
	d := aitest.New(aitest.Replies(ai.Response{
		Content:    ai.Content{ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "read", Input: "{}"})},
		StopReason: ai.StopEndTurn,
	}))

	resp, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("go")})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != ai.StopToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, ai.StopToolUse)
	}
}

func TestWhatTheCallCostSurvivesTheStream(t *testing.T) {
	d := aitest.New(aitest.Replies(ai.Response{
		Content:    ai.TextContent("counted"),
		StopReason: ai.StopEndTurn,
		Usage:      ai.Usage{Input: 100, Output: 5},
	}))

	resp, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("go")})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Usage.Input != 100 || resp.Usage.Output != 5 {
		t.Errorf("Usage = %+v, want input 100 output 5", resp.Usage)
	}
}

func TestAFailingTurnReportsItsError(t *testing.T) {
	boom := errors.New("boom")
	d := aitest.New(aitest.Fails(boom))

	_, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("go")})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want %v", err, boom)
	}
}

func TestAHangingTurnEndsWhenTheCallerGivesUp(t *testing.T) {
	d := aitest.New(aitest.Hangs())
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := d.Client().Complete(ctx, []ai.Message{ai.UserMessage("go")})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a hanging call returned success")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("a hanging call outlived its context")
	}
}

// Sent is the request the provider would have seen, not the one the caller
// wrote: the client's own defaults are merged in on the way.
func TestSentIsTheRequestThatWentOut(t *testing.T) {
	d := aitest.New(aitest.Says("ok"))

	if d.Last() != nil {
		t.Error("a driver reported a request before it was called")
	}
	if _, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("hello")}, ai.WithSystem("be brief")); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	sent := d.Sent()
	if len(sent) != 1 {
		t.Fatalf("Sent has %d requests, want 1", len(sent))
	}
	if sent[0].System != "be brief" {
		t.Errorf("System = %q, want %q", sent[0].System, "be brief")
	}
	if d.Last() != sent[0] {
		t.Error("Last is not the most recent request")
	}
}

func TestAModelCanBeGivenWhenTheTestIsAboutTheModel(t *testing.T) {
	d := aitest.New(aitest.Says("ok"))
	d.Model = ai.Model{ID: "narrow", API: "aitest", ContextWindow: 1000}

	if got := d.Client().Model().ContextWindow; got != 1000 {
		t.Errorf("ContextWindow = %d, want 1000", got)
	}
}

// A call that failed still spent what it spent, and losing that hides real
// money. The error rides on the response rather than replacing it.
func TestAFailedCallStillReportsWhatItCost(t *testing.T) {
	boom := &ai.Error{Kind: ai.KindAuth, Message: "bad key"}
	d := aitest.New(aitest.Replies(ai.Response{
		Content: ai.TextContent("as far as it got"),
		Usage:   ai.Usage{Input: 120, Output: 4},
		Err:     boom,
	}))

	resp, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("go")})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want %v", err, boom)
	}
	if resp.Usage.Input != 120 || resp.Usage.Output != 4 {
		t.Errorf("Usage = %+v, want input 120 output 4", resp.Usage)
	}
	if got := resp.Text(); got != "as far as it got" {
		t.Errorf("text = %q, want what the stream managed before it failed", got)
	}
}

// An endpoint that is down stays down for as many attempts as the caller has
// budget for, without the test having to know how many that is.
func TestAlwaysAnswersEveryCallTheSameWay(t *testing.T) {
	boom := &ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}
	d := aitest.Always(aitest.Fails(boom))

	for i := range 5 {
		if _, err := d.Client().Complete(context.Background(), []ai.Message{ai.UserMessage("go")}); !errors.Is(err, boom) {
			t.Fatalf("call %d: err = %v, want %v", i+1, err, boom)
		}
	}
	if d.Calls() != 5 {
		t.Errorf("Calls = %d, want 5", d.Calls())
	}
}
