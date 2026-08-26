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

// Stream advances the conversation one exchange and reports what it does as it
// goes. Range over it; the last event is TurnEnd.
//
//	for e, err := range a.Stream(ctx, ai.UserMessage("what changed?")) {
//	    render(e)
//	}
//
// Breaking out of the range ends the exchange, the same as Interrupt: a
// consumer that stopped reading has stopped caring about this turn.
//
// The events arrive on the ranging goroutine, so a caller who needs the agent
// to run ahead of a slow reader forwards them to a buffer of its own — where
// how deep it is, and what to drop when it fills, are the caller's to decide.
//
// Turn folds it for a caller that wants the answer rather than the progress.
func (a *Agent) Stream(ctx context.Context, in ...ai.Message) iter.Seq2[Event, error] {
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

// Turn advances the conversation one exchange and reports how it went, for a
// caller with nothing to render: a subagent behind a tool call, which owes the
// model that asked for it an answer rather than a stream.
//
//	out, err := sub.Turn(ctx, ai.UserMessage(task))
//	return agent.TextResult(out.Message.Text()), err
//
// It is Stream, folded. The same operation either way — what differs is what
// you get back.
func (a *Agent) Turn(ctx context.Context, in ...ai.Message) (TurnEnd, error) {
	var out TurnEnd
	for e, err := range a.Stream(ctx, in...) {
		if err != nil {
			return out, err
		}
		if v, ok := e.(TurnEnd); ok {
			out = v
		}
	}
	return out, out.Err
}

// Inject queues messages for the exchange in flight. They join it at the next
// step boundary — changing what the model is about to see is safe exactly once
// per call, and that is where.
//
// Between exchanges they wait for the next one.
func (a *Agent) Inject(msgs ...ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pending = append(a.pending, msgs...)
}

// injected takes what Inject queued, leaving nothing behind.
func (a *Agent) injected() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()

	msgs := a.pending
	a.pending = nil
	return msgs
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
