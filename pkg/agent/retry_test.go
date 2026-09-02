package agent_test

import (
	"context"
	"iter"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// The claim that lets a retry need no event of its own: it is two spans with
// nothing appended between them, and that absence is the signal.
func TestARetryAppendsNothingBetweenAttempts(t *testing.T) {
	a := newAgent(t, &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		Scripts: [][]ai.Delta{nil, text("second time lucky")},
	}, agent.WithRetry(3, 0))

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	assertSequence(t, events, []string{
		"TurnStart",
		"MessageAdded(user)",
		"MessageStart(attempt=1)",
		"MessageEnd(err)", // carries the error, and no message follows
		"MessageStart(attempt=2)",
		"MessageUpdate",
		"MessageUpdate",
		"MessageUpdate",
		"MessageEnd",
		"MessageAdded(assistant)",
		"TurnEnd",
	})

	// Between the failed attempt and the retry, nothing entered the
	// conversation. That is what tells a consumer to discard what it drew.
	firstEnd := slices.IndexFunc(events, func(e agent.Event) bool {
		v, ok := e.(agent.MessageEnd)
		return ok && v.Err != nil
	})
	secondStart := slices.IndexFunc(events, func(e agent.Event) bool {
		v, ok := e.(agent.MessageStart)
		return ok && v.Attempt == 2
	})
	for _, e := range events[firstEnd+1 : secondStart] {
		if v, ok := e.(agent.MessageEnd); ok && v.Err == nil {
			t.Fatal("a message was appended between a failed attempt and its retry")
		}
	}
}

// Retries that all fail must surface the failure. Returning a zero message and
// a nil error would read as a successful empty answer, and the turn would end
// on it as if the model had simply said nothing.
func TestExhaustingRetriesReturnsTheFailure(t *testing.T) {
	overloaded := &ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}
	a := newAgent(t, &scripted{Errs: []error{overloaded, overloaded, overloaded}})

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err == nil {
		t.Fatal("every attempt failed and the turn reported success")
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", last.StopReason)
	}
	// No assistant turn was produced, so none should have been appended.
	if got := a.Messages(); len(got) != 1 {
		t.Errorf("conversation holds %d messages, want just the input", len(got))
	}
}

// A call that failed still spent what it spent. Losing that hides real money.
func TestAFailedCallStillReportsItsCost(t *testing.T) {
	a := newAgent(t, &scripted{
		Errs: []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}},
		Scripts: [][]ai.Delta{
			{{Usage: &ai.Usage{Input: 120, Output: 4}}},
		},
	})

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err == nil {
		t.Fatal("expected the auth failure")
	}
	last := events[len(events)-1].(agent.TurnEnd)
	if got := last.Usage.TotalInput(); got != 120 {
		t.Errorf("input tokens = %d, want 120 — a failed call's cost was dropped", got)
	}
}

// stalling is an endpoint that says nothing until its context ends, which is
// the failure a stream watchdog exists for: it looks exactly like work.
type stalling struct {
	mu      sync.Mutex
	started chan struct{}
	calls   int
	// after is how many calls stall before one answers.
	after int
}

func (d *stalling) Name() string { return "stalling" }

func (d *stalling) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	d.mu.Unlock()

	return func(yield func(ai.Delta, error) bool) {
		if n >= d.after {
			for _, delta := range text("answered at last") {
				if !yield(delta, nil) {
					return
				}
			}
			return
		}
		if d.started != nil && n == 0 {
			close(d.started)
		}
		<-ctx.Done()
		yield(ai.Delta{}, ctx.Err())
	}
}

// A stream that goes quiet is a transient failure, and the one the client
// structurally cannot retry — ending the stall cancels the context ai.Retry
// would wait on — so it is what the agent's own budget is for.
func TestAStalledStreamIsRetried(t *testing.T) {
	a := newAgent(t, &stalling{after: 1},
		agent.WithStreamTimeout(20*time.Millisecond, 20*time.Millisecond),
		agent.WithRetry(2, 0))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("the stall was not recovered from: %v", err)
	}

	if n := steps(events); n != 1 {
		t.Errorf("steps = %d, want 1 — a retry is not a second step", n)
	}
	var attempts []int
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok {
			attempts = append(attempts, v.Attempt)
		}
	}
	if want := []int{1, 2}; !slices.Equal(attempts, want) {
		t.Errorf("attempts = %v, want %v", attempts, want)
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopEndTurn {
		t.Errorf("stop reason = %q, want the retry to have carried the turn", last.StopReason)
	}
}

// A watchdog nobody asked for does not fire: zero means no limit, and a slow
// endpoint is not an error.
func TestNoStreamTimeoutMeansNoWatchdog(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("fine")}},
		agent.WithStreamTimeout(0, 0))

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
}

// An endpoint that says how long to wait is the whole reason RetryAfter is on
// the error. Replaying immediately is what rate limiting exists to punish, and
// a loop that counts attempts without ever waiting does exactly that.
func TestARateLimitIsWaitedOutForAsLongAsItAsked(t *testing.T) {
	const askedFor = 60 * time.Millisecond
	a := newAgent(t, &scripted{
		Errs: []error{&ai.Error{
			Kind: ai.KindRateLimit, Message: "slow down", RetryAfter: askedFor,
		}},
		Scripts: [][]ai.Delta{nil, text("thank you for waiting")},
		// A backoff of zero, so anything waited for came from the endpoint.
	}, agent.WithRetry(2, 0))

	start := time.Now()
	if _, err := outcome(t, a, ai.UserMessage("hi")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if waited := time.Since(start); waited < askedFor {
		t.Errorf("retried after %v; the endpoint asked for %v", waited, askedFor)
	}
}

// Retry is off unless asked for, because the client already has ai.Retry and
// two budgets multiply: three attempts here on a client wrapping ai.Retry(3)
// is nine model calls for one step.
func TestAnAgentDoesNotRetryUnlessAsked(t *testing.T) {
	d := &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		Scripts: [][]ai.Delta{nil, text("would have been the retry")},
	}
	a := newAgent(t, d)

	if _, err := outcome(t, a, ai.UserMessage("hi")); err == nil {
		t.Fatal("the failure never reached the caller")
	}
	if n := d.Calls(); n != 1 {
		t.Errorf("the driver was called %d times, want 1 — retry is the client's", n)
	}
}
