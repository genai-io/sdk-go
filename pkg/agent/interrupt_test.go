package agent_test

import (
	"context"
	"errors"
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/internal/scripted"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Breaking out of the range ends the exchange: a consumer that stopped reading
// has stopped caring about this turn.
func TestBreakingOutOfTheRangeEndsTheExchange(t *testing.T) {
	driver := &scripted.Driver{Scripts: [][]ai.Delta{
		toolCall("c1", "never", `{}`),
		text("unreachable"),
	}}
	a := newAgent(t, driver)

	for e, err := range a.Run(context.Background(), ai.UserMessage("go")) {
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if _, ok := e.(agent.MessageEnd); ok {
			break
		}
	}

	calls := driver.Calls()
	if calls != 1 {
		t.Errorf("the model was called %d times after the consumer left, want 1", calls)
	}
}

// Interrupt says when the exchange it ended has actually finished. The caller
// that presses escape is not the caller ranging over Run, so it cannot see the
// range end; without this it has no way to know when the agent stopped
// touching the conversation.
func TestInterruptSaysWhenTheExchangeIsOver(t *testing.T) {
	release := make(chan struct{})
	held := agent.ToolFunc("hold", "blocks until released",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			select {
			case <-release:
			case <-ctx.Done():
			}
			return agent.TextResult("let go"), nil
		})
	d := &scripted.Driver{Scripts: [][]ai.Delta{toolCall("1", "hold", "{}"), text("done")}}
	a := newAgent(t, d, agent.WithTools(held))

	inTool := make(chan struct{})
	stopped := make(chan (<-chan struct{}), 1)
	go func() {
		<-inTool
		stopped <- a.Interrupt() // from somewhere that never sees the range
	}()

	ranging := make(chan struct{})
	go func() {
		defer close(ranging)
		for e := range a.Run(context.Background(), ai.UserMessage("go")) {
			if _, ok := e.(agent.ToolStart); ok {
				close(inTool)
			}
		}
	}()

	select {
	case <-(<-stopped):
	case <-time.After(2 * time.Second):
		t.Fatal("the channel from Interrupt never closed")
	}

	// Closed means over: the range has ended and the agent is free again.
	select {
	case <-ranging:
	case <-time.After(time.Second):
		t.Error("the channel closed while the exchange was still running")
	}
	if _, err := outcome(t, a, ai.UserMessage("again")); err != nil {
		t.Errorf("the agent was still busy after it said it had stopped: %v", err)
	}
	close(release)
}

// Between exchanges there is nothing to interrupt, and waiting on it must not
// be a way to hang.
func TestInterruptBetweenExchangesDoesNotBlock(t *testing.T) {
	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{text("hi")}})

	select {
	case <-a.Interrupt():
	case <-time.After(time.Second):
		t.Fatal("Interrupt blocked with no exchange running")
	}

	// And it did not poison the next one.
	if out, err := outcome(t, a, ai.UserMessage("go")); err != nil || out.StopReason != agent.StopEndTurn {
		t.Errorf("the next exchange = %v / %q, want a clean end_turn", err, out.StopReason)
	}
}

// A model that is still talking when the user gives up on it: the stream is
// abandoned, and the exchange has to say so without pretending the call was
// free. The tokens were spent whether or not the answer was kept.
type blockingStream struct{ usage ai.Usage }

func (d *blockingStream) Name() string { return "blocking" }

func (d *blockingStream) Stream(ctx context.Context, _ *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		if !yield(ai.Delta{Block: ai.TextBlock("as far as it got"), Usage: &d.usage}, nil) {
			return
		}
		<-ctx.Done() // still thinking, until this exchange stops caring
	}
}

// Ending an exchange mid-stream, both ways round: Interrupt, which ends this
// turn, and cancelling the context Run was given, which ends everything.
func TestAnInterruptedStreamClosesItsSpanAndKeepsWhatItCost(t *testing.T) {
	for _, tc := range []struct {
		name string
		stop func(*agent.Agent, context.CancelFunc)
	}{
		{"interrupted", func(a *agent.Agent, _ context.CancelFunc) { a.Interrupt() }},
		{"the caller's context ended", func(_ *agent.Agent, cancel context.CancelFunc) { cancel() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newAgent(t, &blockingStream{usage: ai.Usage{Input: 100, Output: 5}})

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			var events []agent.Event
			for e, err := range a.Run(ctx, ai.UserMessage("go")) {
				if err != nil {
					t.Fatalf("the iterator yielded %v", err)
				}
				events = append(events, e)
				if _, ok := e.(agent.MessageUpdate); ok {
					tc.stop(a, cancel)
				}
			}

			var end agent.MessageEnd
			var closed, addedAfter bool
			for _, e := range events {
				switch v := e.(type) {
				case agent.MessageEnd:
					end, closed = v, true
				case agent.MessageAdded:
					if closed {
						addedAfter = true
					}
				}
			}
			if !closed {
				t.Fatal("the span that opened was never closed; a reader is still waiting on it")
			}
			if end.Err == nil {
				t.Error("MessageEnd says the call succeeded; nothing came of it")
			}
			if addedAfter {
				t.Error("an abandoned answer was added to the conversation")
			}
			if got := len(a.Messages()); got != 1 {
				t.Errorf("the conversation holds %d messages, want just the input", got)
			}

			last, ok := events[len(events)-1].(agent.TurnEnd)
			if !ok {
				t.Fatalf("the last event is %T, want TurnEnd", events[len(events)-1])
			}
			if last.StopReason != agent.StopCanceled {
				t.Errorf("stop reason = %q, want canceled", last.StopReason)
			}
			if got := last.Usage.Total(); got != 105 {
				t.Errorf("usage = %d tokens, want the 105 the abandoned call spent", got)
			}
		})
	}
}

// A batch that runs one tool at a time stops where the turn stopped: the calls
// that have not started must not, and each is answered the way one that watched
// its own context would have been.
func TestACancelledBatchStopsBeforeTheNextTool(t *testing.T) {
	var a *agent.Agent
	var ran atomic.Int64
	slow := agent.Sequential(agent.ToolFunc("step", "Do one thing at a time.",
		func(context.Context, struct{}) (agent.Result, error) {
			if ran.Add(1) == 1 {
				a.Interrupt() // the user gives up while the first one works
			}
			return agent.TextResult("done"), nil
		}))

	a = newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "step", Input: `{}`})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c2", Name: "step", Input: `{}`})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c3", Name: "step", Input: `{}`})},
			{StopReason: ai.StopToolUse},
		},
		text("never asked for"),
	}}, agent.WithTools(slow))

	// The turn ends canceled, so collect hands back that outcome's error;
	// what this test is about is what the batch did on the way out.
	events, _ := collect(t, a, ai.UserMessage("go"))
	if got := ran.Load(); got != 1 {
		t.Errorf("%d tools ran, want the 1 that had already started", got)
	}

	ends := map[string]error{}
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok {
			ends[v.ID] = v.Err
		}
	}
	if len(ends) != 3 {
		t.Fatalf("%d calls were closed, want all 3 — an unanswered call is one the model waits for", len(ends))
	}
	if ends["c1"] != nil {
		t.Errorf("the call that ran reported %v", ends["c1"])
	}
	for _, id := range []string{"c2", "c3"} {
		if !errors.Is(ends[id], context.Canceled) {
			t.Errorf("%s was closed with %v, want the cancellation that stopped it", id, ends[id])
		}
	}
	if last := events[len(events)-1].(agent.TurnEnd); last.StopReason != agent.StopCanceled {
		t.Errorf("stop reason = %q, want canceled", last.StopReason)
	}
}
