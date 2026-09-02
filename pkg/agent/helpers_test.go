package agent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// scripted answers each call with the next script, then with the matching
// error. Running out of scripts is itself an error, so a test that provoked
// one more call than it wrote is told.
type scripted struct {
	Scripts [][]ai.Delta
	Errs    []error
	Keep    bool // record requests for Sent; off by default so heap tests measure the agent

	mu    sync.Mutex
	calls int
	got   []*ai.Request
}

func (d *scripted) Name() string { return "scripted" }

func (d *scripted) Stream(_ context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
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

func (d *scripted) Calls() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.calls
}

// Sent is what reached the driver after the client merged its defaults in and
// repaired the history. Empty unless Keep was set.
func (d *scripted) Sent() []*ai.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.got
}

// text is a model that says one thing and stops.
func text(s string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.TextBlock(s)},
		{EndBlock: true},
		{StopReason: ai.StopEndTurn},
	}
}

// toolCall is a model that asks for one tool and waits.
func toolCall(id, name, input string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.ToolCallBlock(ai.ToolCall{ID: id, Name: name, Input: input})},
		{StopReason: ai.StopToolUse},
	}
}
