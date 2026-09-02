// Package scripted is a model that answers from a script: a test says what the
// endpoint does as data, and asserts on what the agent did with it.
//
// It is here rather than in each test file because three packages were keeping
// a copy of it, and three copies of a stub are three definitions of what a
// model is allowed to do.
package scripted

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Driver answers each call with the next script and then with the matching
// error, which is how an endpoint really fails: some of the answer arrives,
// and the tokens for it are already spent. Running out of scripts is itself an
// error, so a test that provoked one more call than it wrote is told.
type Driver struct {
	// Scripts is what each call streams, in order.
	Scripts [][]ai.Delta
	// Errs is what each call fails with once its script has run, nil for one
	// that succeeds.
	Errs []error
	// Keep records what was sent, for Sent. Off by default: a test that runs
	// hundreds of exchanges and then measures the heap must not be measuring
	// the requests this held on to.
	Keep bool

	mu    sync.Mutex
	calls int
	got   []*ai.Request
}

func (d *Driver) Name() string { return "scripted" }

func (d *Driver) Stream(_ context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	if d.Keep {
		d.got = append(d.got, req)
	}
	d.mu.Unlock()

	return func(yield func(ai.Delta, error) bool) {
		if n < len(d.Scripts) {
			for _, delta := range d.Scripts[n] {
				if !yield(delta, nil) {
					return
				}
			}
		}
		if n < len(d.Errs) && d.Errs[n] != nil {
			yield(ai.Delta{}, d.Errs[n])
			return
		}
		if n >= len(d.Scripts) {
			yield(ai.Delta{}, errors.New("scripted: no script for call "+fmt.Sprint(n)))
		}
	}
}

// Calls is how many times the model was asked.
func (d *Driver) Calls() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.calls
}

// Sent is what reached the driver, after the client merged its defaults in and
// repaired the history: the last word on what really went out. Empty unless
// Keep was set.
func (d *Driver) Sent() []*ai.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.got
}

// Text is a model that says one thing and stops.
func Text(s string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.TextBlock(s)},
		{EndBlock: true},
		{StopReason: ai.StopEndTurn},
	}
}

// ToolCall is a model that asks for one tool and waits.
func ToolCall(id, name, input string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.ToolCallBlock(ai.ToolCall{ID: id, Name: name, Input: input})},
		{StopReason: ai.StopToolUse},
	}
}
