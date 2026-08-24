package agent

import (
	"context"
	"errors"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrBusy means this agent is already running, or has already run. One loop,
// one conversation: two loops appending to it would interleave their turns
// into a history neither asked for. Concurrency belongs between agents.
var ErrBusy = errors.New("agent: already running")

// Run takes one exchange at a time off In and reports what it does on Out.
// RunStart is the first event and RunEnd the last.
//
// It returns nil once In is closed and the exchange in flight has finished,
// ctx.Err() if the context ends, and ErrBusy if this agent has run before. A
// failed exchange is none of those: TurnEnd carries the error and Run serves
// the next message. To make a failure fatal, watch for it and close In.
func (a *Agent) Run(ctx context.Context) (err error) {
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return ErrBusy
	}
	a.running = true
	a.mu.Unlock()

	a.emit(ctx, RunStart{})

	// Whichever way the loop below leaves, it leaves the same way: one last
	// event, then the channel closed so a reader ranging over it stops.
	defer func() {
		select {
		case a.out <- RunEnd{Err: err}:
		default:
		}
		close(a.out)
	}()

	for {
		var batch []ai.Message
		select {
		case <-ctx.Done():
			return ctx.Err()
		case msg, ok := <-a.in:
			if !ok {
				return nil
			}
			// Everything queued behind it joins the same exchange: someone who
			// typed three lines while the agent worked meant them together.
			batch = append([]ai.Message{msg}, drain(a.in)...)
		}

		// A turn reports how it went on TurnEnd, in a form that says more than
		// an error does, so it hands nothing back here. The only failure that
		// ends a run is the context ending.
		a.turnCount.Add(1)
		a.turn(ctx, batch)
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
}

// emit hands one event to the reader. What happens when the reader is behind
// depends on the event, and this is the only place that decides.
func (a *Agent) emit(ctx context.Context, e Event) {
	switch e.(type) {
	case MessageUpdate, ToolUpdate:
		select {
		case a.out <- e:
		default:
		}
	default:
		select {
		case a.out <- e:
		case <-ctx.Done():
		}
	}
}

// add puts a message into the conversation and reports it, in that order: by
// the time a reader sees MessageAdded, Messages already holds it. The other
// order hands a handler news of a message and a conversation without it.
func (a *Agent) add(ctx context.Context, msg ai.Message) {
	a.mu.Lock()
	a.messages = append(a.messages, msg)
	a.mu.Unlock()

	a.emit(ctx, MessageAdded{Message: msg})
}

// drain takes everything already queued, without waiting for more.
func drain(ch <-chan ai.Message) []ai.Message {
	var out []ai.Message
	for {
		select {
		case m, ok := <-ch:
			if !ok {
				return out
			}
			out = append(out, m)
		default:
			return out
		}
	}
}
