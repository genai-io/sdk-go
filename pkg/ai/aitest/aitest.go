// Package aitest is a model that does what a test tells it to.
//
// Faking a model means faking the protocol, which is the seam [ai.Driver]
// already is and the one every real driver sits behind: a test says what the
// endpoint sends, and nothing between the endpoint and the caller has to be
// stubbed. Writing that driver by hand is a dozen lines of iterator plumbing
// per test file, and every copy gets to decide differently what happens when a
// test provokes one more call than it wrote for.
//
// The shape is a driver playing a list of turns:
//
//	d := aitest.New(
//		aitest.Asks(ai.ToolCall{ID: "1", Name: "read", Input: `{}`}),
//		aitest.Says("done"),
//	)
//	a, err := agent.New(d.Client(), agent.WithTools(myTool))
//
// A [Turn] is a function, so the constructors here are conveniences rather
// than a closed set: a behaviour they do not cover is an ordinary literal.
package aitest

import (
	"context"
	"errors"
	"iter"
	"strconv"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Turn is what the model does on one call: the delta stream that call
// produces.
//
// It is handed the call's context, so a turn can outlast the caller's patience
// on purpose — see [Hangs] — and the request, because a real model's answer
// depends on what it was asked, and a test standing in for one sometimes has
// to as well. The constructors below ignore both.
type Turn func(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error]

// Replies is the general turn: the model produces this answer.
//
// A [ai.Response] is already an ordered block sequence, so this walks it rather
// than reassembling a stream out of parallel fields. A response carrying tool
// calls stops on [ai.StopToolUse] whatever it says, because that is the only
// stop reason that leaves the loop a call to run; one with no stop reason at
// all ends the turn.
//
// A response with an Err is a call that got this far and then failed, which is
// what [ai.Response.Err] means everywhere else. It streams what it has — the
// usage included, because a failed call still spent what it spent — and ends
// on the error instead of on a stop reason.
func Replies(r ai.Response) Turn {
	var out []ai.Delta
	calls := 0
	for _, b := range r.Content {
		if b.Type == ai.BlockToolCall {
			calls++
			out = append(out, ai.Delta{Block: b})
			continue
		}
		out = append(out, ai.Delta{Block: b}, ai.Delta{EndBlock: true})
	}
	stop := r.StopReason
	switch {
	case r.Err != nil:
		// A stream that failed never delivered one.
		stop = ""
	case calls > 0:
		stop = ai.StopToolUse
	case stop == "":
		stop = ai.StopEndTurn
	}
	usage := r.Usage
	out = append(out, ai.Delta{StopReason: stop, Usage: &usage, Model: r.Model, ID: r.ID})

	if r.Err == nil {
		return Streams(out...)
	}
	return func(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) {
			for _, d := range out {
				if !yield(d, nil) {
					return
				}
			}
			yield(ai.Delta{}, r.Err)
		}
	}
}

// Says is a model that answers with one piece of text and finishes.
func Says(text string) Turn {
	return Stops(ai.StopEndTurn, text)
}

// Stops is a model that says this much and then stops for the given reason —
// [ai.StopMaxTokens] for an answer the output cap cut off, which is what
// agent.WithContinuation exists to resume, or a refusal, or a stop sequence.
func Stops(reason ai.StopReason, text string) Turn {
	return Replies(ai.Response{Content: ai.TextContent(text), StopReason: reason})
}

// Then is one turn made of several, played in order: what a call produced
// before it stalled or failed.
//
//	aitest.Then(aitest.Streams(partial...), aitest.Hangs())
func Then(turns ...Turn) Turn {
	return func(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) {
			for _, turn := range turns {
				for d, err := range turn(ctx, req) {
					if !yield(d, err) {
						return
					}
					if err != nil {
						return
					}
				}
			}
		}
	}
}

// Asks is a model that wants tools run. The loop runs them and calls again,
// so the turn after this one is the model seeing their results.
func Asks(calls ...ai.ToolCall) Turn {
	content := make(ai.Content, 0, len(calls))
	for _, c := range calls {
		content = append(content, ai.ToolCallBlock(c))
	}
	return Replies(ai.Response{Content: content, StopReason: ai.StopToolUse})
}

// Fails is a call that produces nothing and ends on an error. For one that got
// partway first — text, or the tokens it had already spent — put the error on
// the response and use [Replies].
func Fails(err error) Turn {
	return func(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) { yield(ai.Delta{}, err) }
	}
}

// Hangs is a model that never answers: the stream stays open until the context
// ends, which is what a caller giving up on a slow endpoint has to survive.
func Hangs() Turn {
	return func(ctx context.Context, _ *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) {
			<-ctx.Done()
			yield(ai.Delta{}, ctx.Err())
		}
	}
}

// Streams is the raw turn: send exactly these deltas. Use it for a stream
// whose shape is the point — a block left unclosed, usage arriving before the
// text it paid for.
func Streams(deltas ...ai.Delta) Turn {
	return func(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) {
			for _, d := range deltas {
				if !yield(d, nil) {
					return
				}
			}
		}
	}
}

// Driver plays one turn per call, in order.
//
// Running out of turns is itself an error rather than a silent end-of-turn: a
// test that provoked one more inference than it wrote for is told so, at the
// call that did it.
type Driver struct {
	// Model is what Client reports talking to. The zero value is a model with
	// no protocol rules of its own, so a test exercises the code under test
	// rather than request validation.
	Model ai.Model

	turns  []Turn
	always Turn

	mu    sync.Mutex
	calls int
	sent  []*ai.Request
}

// New returns a driver that answers with these turns, in order.
func New(turns ...Turn) *Driver { return &Driver{turns: turns} }

// Always returns a driver that answers every call the same way — an endpoint
// that is down and stays down, or one that never answers at all.
//
// Prefer it to repeating a turn as often as the retry budget allows: writing
// the budget into the model as well as into the assertion means a change to
// either leaves the test passing for the wrong reason, having run out of turns
// rather than out of attempts.
func Always(turn Turn) *Driver { return &Driver{always: turn} }

// Name identifies this driver the way a real one names its protocol.
func (d *Driver) Name() string { return "aitest" }

// Stream plays the turn for this call.
func (d *Driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	d.sent = append(d.sent, req)
	d.mu.Unlock()

	if d.always != nil {
		return d.always(ctx, req)
	}
	if n >= len(d.turns) {
		return Fails(errors.New("aitest: no turn written for call "+strconv.Itoa(n+1)))(ctx, req)
	}
	return d.turns[n](ctx, req)
}

// Client is this driver as the client the code under test takes.
func (d *Driver) Client() *ai.Client {
	model := d.Model
	if model.ID == "" {
		model = ai.Model{ID: "aitest", API: "aitest", ContextWindow: 200_000}
	}
	return ai.NewClientWithDriver(d, model)
}

// Calls reports how many inferences reached the driver.
func (d *Driver) Calls() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.calls
}

// Sent is what reached the driver, after the client merged its defaults in and
// repaired the history — which is the request the provider would have seen,
// not the one the caller wrote.
func (d *Driver) Sent() []*ai.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.sent
}

// Last is the most recent request, or nil before the first call.
func (d *Driver) Last() *ai.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.sent) == 0 {
		return nil
	}
	return d.sent[len(d.sent)-1]
}
