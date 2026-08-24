package agent

import (
	"context"
	"errors"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrBusy means this agent is already busy: Run has been called, or a Turn is
// in flight. One loop, one conversation — two of them appending to it would
// interleave their turns into a history neither asked for. Concurrency belongs
// between agents.
//
// Run latches it, because Run closes Out and that cannot be undone. Turn
// releases it, because a turn closes nothing.
var ErrBusy = errors.New("agent: already running")

// Run takes one exchange at a time off In and reports what it does on Out.
// RunStart is the first event and RunEnd the last.
//
// It returns nil once In is closed and the exchange in flight has finished,
// ctx.Err() if the context ends, and ErrBusy if this agent has run before. A
// failed exchange is none of those: TurnEnd carries the error and Run serves
// the next message. To make a failure fatal, watch for it and close In.
func (a *Agent) Run(ctx context.Context) (err error) {
	if !a.running.CompareAndSwap(false, true) {
		return ErrBusy
	}

	a.alive = ctx.Done()
	a.emit(RunStart{})

	// Whichever way the loop below leaves, it leaves the same way: one last
	// event, then the channel closed so a reader ranging over it stops.
	defer func() {
		if a.out == nil {
			return
		}
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
		//
		a.exchange(ctx, batch)
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
}

// Turn runs one exchange and hands back how it went, for a caller that wants
// an answer rather than a stream: a subagent standing behind a tool call, which
// has to return something to the model that asked for it.
//
//	out, err := sub.Turn(ctx, ai.UserMessage(task))
//	return agent.TextResult(out.Message.Text()), err
//
// It reports on the way through like everything else, and closes nothing: a
// short exchange fits the buffer, so a caller can drain Out afterwards, while a
// long one needs somebody reading as it goes. An agent built WithoutEvents
// reports nothing and needs neither. What Turn will not do is bypass the stream
// while pretending to have one, which is how a session comes to record an
// empty subagent.
//
// It emits no RunStart or RunEnd: a turn is not a run. Messages waiting on In
// join the exchange at the next step boundary, the same as under Run, and
// Interrupt ends it the same way. Calling it while Run holds the agent returns
// ErrBusy; calling it again after it returns is fine, because a turn does not
// close anything.
func (a *Agent) Turn(ctx context.Context, in ...ai.Message) (TurnEnd, error) {
	if !a.running.CompareAndSwap(false, true) {
		return TurnEnd{}, ErrBusy
	}
	defer a.running.Store(false)

	a.alive = ctx.Done()
	out := a.exchange(ctx, in)
	return out, out.Err
}

// exchange runs one turn under a context of its own, so Interrupt can end it
// without ending whatever is driving. Both drivers go through here rather than
// deriving it themselves, because the two must not come to disagree about what
// Interrupt reaches.
func (a *Agent) exchange(ctx context.Context, in []ai.Message) TurnEnd {
	turnCtx, stopTurn := context.WithCancel(ctx)
	defer stopTurn()

	a.mu.Lock()
	a.stopTurn = stopTurn
	a.mu.Unlock()

	a.turnCount.Add(1)
	return a.turn(turnCtx, in)
}

// emit hands one event to the reader. What happens when the reader is behind
// depends on the event, and this is the only place that decides.
func (a *Agent) emit(e Event) {
	if a.out == nil {
		return
	}
	switch e.(type) {
	case MessageUpdate, ToolUpdate:
		select {
		case a.out <- e:
		default:
		}
	default:
		select {
		case a.out <- e:
		case <-a.alive:
		}
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
