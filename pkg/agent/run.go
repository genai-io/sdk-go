package agent

import (
	"context"
	"errors"
	"iter"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrBusy means an exchange is already running on this agent. One conversation
// advances one turn at a time: two of them appending to it would interleave
// into a history neither asked for. Concurrency belongs between agents.
var ErrBusy = errors.New("agent: an exchange is already running")

// Run advances the conversation one exchange and reports what it does as it
// goes. Range over it; the last event is TurnEnd, which carries how it went
// and the answer it came to.
//
// One exchange, not the agent's whole life — the same shape exec.Cmd.Run has,
// where running means doing this thing and being done.
//
//	for e, err := range a.Run(ctx, ai.UserMessage("what changed?")) {
//	    render(e)
//	}

// Breaking out of the range ends the exchange, the same as Interrupt: a
// consumer that stopped reading has stopped caring about this turn.
//
// The events arrive on the ranging goroutine, so a caller who needs the agent
// to run ahead of a slow reader forwards them to a buffer of its own — where
// how deep it is, and what to drop when it fills, are the caller's to decide.
//
// Repeating it is a for loop, and the loop is the caller's: how messages are
// batched into exchanges, what happens when one fails, and when to stop are
// all things the application knows and this package does not.
//
//	for batch := range myMessages {
//	    for e, err := range a.Run(ctx, batch...) { render(e) }
//	}
func (a *Agent) Run(ctx context.Context, in ...ai.Message) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if !a.running.CompareAndSwap(false, true) {
			yield(nil, ErrBusy)
			return
		}
		defer a.running.Store(false)

		a.yield = yield
		defer func() { a.yield = nil }()

		out := a.turn(ctx, in)
		if out.Err != nil && a.yield != nil {
			yield(nil, out.Err)
		}
	}
}

// emit hands one event to whoever is ranging. A consumer that breaks out of
// the range ends the exchange, which is what Interrupt does — so it is said
// the same way — and is never yielded to again: that is the iterator's rule,
// and a nil yield is how the rest of the turn knows nobody is listening.
//
// It must be called from the goroutine running the turn. A yield is not safe
// anywhere else, which is why a tool's progress travels back through a channel
// before it gets here.
func (a *Agent) emit(e Event) {
	if a.yield == nil {
		return
	}
	if !a.yield(e, nil) {
		a.yield = nil
		a.stopTurn()
	}
}

// add puts a message into the conversation and reports it, in that order: by
// the time a reader sees MessageAdded, Messages already holds it. The other
// order hands a handler news of a message and a conversation without it.
func (a *Agent) add(msg ai.Message) {
	a.mu.Lock()
	a.messages = append(a.messages, msg)
	a.mu.Unlock()

	a.emit(MessageAdded{Message: msg})
}
