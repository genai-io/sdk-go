// Package aitest is a model that does what a test tells it to: an [ai.Driver]
// playing a list of turns.
//
//	d := aitest.New(
//		aitest.Asks(ai.ToolCall{ID: "1", Name: "read", Input: `{}`}),
//		aitest.Says("done"),
//	)
//	a, err := agent.New(d.Client(), agent.WithTools(myTool))
//
// [Turn] is a function, so the constructors below are conveniences, not a
// closed set.
package aitest

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strconv"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Turn is what the model does on one call. The constructors below ignore both
// arguments; a hand-written one can answer from the request, or outlast the
// context on purpose.
type Turn func(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error]

// Replies is the general turn: the model produces this answer. Tool calls force
// [ai.StopToolUse], no stop reason means end of turn, and an Err streams what
// the response has and then fails.
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

// Stops says this much and then stops for the given reason.
func Stops(reason ai.StopReason, text string) Turn {
	return Replies(ai.Response{Content: ai.TextContent(text), StopReason: reason})
}

// Then plays several turns as one call.
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

// Fails produces nothing and ends on an error. For one that got partway first,
// put the error on the response and use [Replies].
func Fails(err error) Turn {
	return func(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) { yield(ai.Delta{}, err) }
	}
}

// Hangs never answers: the stream stays open until the context ends.
func Hangs() Turn {
	return func(ctx context.Context, _ *ai.Request) iter.Seq2[ai.Delta, error] {
		return func(yield func(ai.Delta, error) bool) {
			<-ctx.Done()
			yield(ai.Delta{}, ctx.Err())
		}
	}
}

// Streams sends exactly these deltas, for a test about the shape of a stream.
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

// Driver plays one turn per call, in order. Running out of turns is an error
// naming the call that did it, not a silent end-of-turn.
type Driver struct {
	// Model is what Client reports talking to. The zero value has no protocol
	// rules, so a test exercises its subject and not request validation.
	Model ai.Model

	turns  []Turn
	always Turn

	mu    sync.Mutex
	calls int
	sent  []*ai.Request
}

// New answers with these turns, in order.
func New(turns ...Turn) *Driver { return &Driver{turns: turns} }

// Always answers every call the same way — an endpoint that is down and stays
// down. Prefer it to repeating a turn as often as a retry budget allows.
func Always(turn Turn) *Driver { return &Driver{always: turn} }

func (d *Driver) Name() string { return "aitest" }

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

// Client is this driver as an [ai.Client].
func (d *Driver) Client() *ai.Client {
	model := d.Model
	if model.ID == "" {
		model = ai.Model{ID: "aitest", API: "aitest", ContextWindow: 200_000}
	}
	return ai.NewClientWithDriver(d, model)
}

// Calls is how many inferences reached the driver.
func (d *Driver) Calls() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.calls
}

// Sent is what reached the driver: the request the provider would have seen,
// after the client merged its defaults in. A copy.
func (d *Driver) Sent() []*ai.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	return slices.Clone(d.sent)
}

// Last is the most recent request, nil before the first call.
func (d *Driver) Last() *ai.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.sent) == 0 {
		return nil
	}
	return d.sent[len(d.sent)-1]
}
