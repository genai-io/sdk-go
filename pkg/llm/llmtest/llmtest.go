// Package llmtest provides a scripted driver for testing code that calls a
// model, without a network or a credential.
//
//	drv := llmtest.Text("hello ", "world")
//	client := llm.New(drv, llmtest.Model)
//	resp, err := client.Complete(ctx, &llm.Request{Messages: ...})
//
// The driver records every request it is given, so a test can assert on what
// its subject actually sent — the system prompt it assembled, the tools it
// offered, the history it replayed.
package llmtest

import (
	"context"
	"iter"
	"sync"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Model is a stand-in model with generous limits and no reasoning.
var Model = llm.Model{
	ID:            "test-model",
	Name:          "Test Model",
	API:           "test",
	Vendor:        "llmtest",
	ContextWindow: 100_000,
	MaxOutput:     4_096,
}

// Call is one recorded invocation: what the subject asked for, and how.
type Call struct {
	Prompt  *llm.Prompt
	Options llm.Options
}

// Driver is a scripted llm.Driver.
//
// Turns are consumed in order; once the script runs out, the last turn repeats
// so a multi-step agent loop under test does not fall off the end.
type Driver struct {
	// Turns is the script. Each entry is one Generate call.
	Turns []Turn
	// Catalog is what Models returns.
	Catalog []llm.Model

	mu       sync.Mutex
	recorded []Call
	calls    int
}

// Turn is one scripted response: the deltas to emit, then optionally an error.
type Turn struct {
	Deltas []llm.Delta
	// Err, when set, terminates the stream after the deltas.
	Err error
}

// Text returns a driver that streams the given chunks as one answer.
func Text(chunks ...string) *Driver {
	deltas := make([]llm.Delta, 0, len(chunks)+1)
	for _, c := range chunks {
		deltas = append(deltas, llm.Delta{Text: c})
	}
	deltas = append(deltas, llm.Delta{
		StopReason: llm.StopEndTurn,
		Usage:      &llm.Usage{Input: 10, Output: len(chunks)},
	})
	return &Driver{Turns: []Turn{{Deltas: deltas}}}
}

// Fail returns a driver whose only turn is the given error.
func Fail(err error) *Driver {
	return &Driver{Turns: []Turn{{Err: err}}}
}

// Tool returns a driver that answers with one tool call.
func Tool(id, name, input string) *Driver {
	return &Driver{Turns: []Turn{{Deltas: []llm.Delta{
		{ToolCall: &llm.ToolCall{ID: id, Name: name, Input: input}},
		{StopReason: llm.StopToolUse, Usage: &llm.Usage{Input: 10, Output: 5}},
	}}}}
}

// Name identifies the driver.
func (d *Driver) Name() string { return "llmtest" }

// Generate replays the next scripted turn.
func (d *Driver) Generate(ctx context.Context, p *llm.Prompt, opts llm.Options) iter.Seq2[llm.Delta, error] {
	d.mu.Lock()
	d.recorded = append(d.recorded, Call{Prompt: p, Options: opts})
	turn := Turn{Deltas: []llm.Delta{{StopReason: llm.StopEndTurn}}}
	if len(d.Turns) > 0 {
		turn = d.Turns[min(d.calls, len(d.Turns)-1)]
	}
	d.calls++
	d.mu.Unlock()

	return func(yield func(llm.Delta, error) bool) {
		for _, delta := range turn.Deltas {
			if ctx.Err() != nil {
				yield(llm.Delta{}, ctx.Err())
				return
			}
			if !yield(delta, nil) {
				return
			}
		}
		if turn.Err != nil {
			yield(llm.Delta{}, turn.Err)
		}
	}
}

// Models returns the configured catalog, or the stand-in model.
func (d *Driver) Models(context.Context) ([]llm.Model, error) {
	if d.Catalog != nil {
		return d.Catalog, nil
	}
	return []llm.Model{Model}, nil
}

// Calls returns every invocation the driver was given, in order.
func (d *Driver) Calls() []Call {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := make([]Call, len(d.recorded))
	copy(out, d.recorded)
	return out
}

// Last returns the most recent invocation. It panics when there was none,
// which in a test is the clearer failure.
func (d *Driver) Last() Call {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.recorded) == 0 {
		panic("llmtest: no calls recorded")
	}
	return d.recorded[len(d.recorded)-1]
}

// CallCount returns how many times Generate was called.
func (d *Driver) CallCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.calls
}

// Client wraps the driver in a Client on the stand-in model.
func Client(d *Driver, opts ...llm.Option) *llm.Client { return llm.New(d, Model, opts...) }

var _ llm.Driver = (*Driver)(nil)
