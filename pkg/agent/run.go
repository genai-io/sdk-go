package agent

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math"
	"runtime/debug"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrBusy means an exchange is already running on this agent. One conversation
// advances one turn at a time: two of them appending to it would interleave
// into a history neither asked for. Concurrency belongs between agents.
var ErrBusy = errors.New("agent: an exchange is already running")

// Run advances the conversation one exchange and reports what it does as it
// goes. The last event is TurnEnd, which says how it went.
//
//	for e, err := range a.Run(ctx, ai.UserMessage("what changed?")) {
//	    render(e)
//	}
//
// Breaking out of the range ends the exchange, the same as Interrupt. Events
// arrive on the ranging goroutine, so a caller who needs the agent to run
// ahead of a slow reader forwards them to a buffer of its own — how deep, and
// what to drop, being decisions only the caller can make.
//
// Repeating it is a for loop, and that loop is the caller's: how messages are
// batched, what a failure means and when to stop are things the application
// knows and this package does not.
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

		// gone remembers a consumer that broke out of the range: yielding to
		// one again is what the iterator forbids, and stopping reading is
		// also what ends the exchange.
		gone := false
		emit := func(e Event) {
			if gone {
				return
			}
			if !yield(e, nil) {
				gone = true
				a.stopTurn()
			}
		}

		out := a.turn(ctx, emit, in)
		if out.Err != nil && !gone {
			yield(nil, out.Err)
		}
	}
}

// add puts a message into the conversation and reports it, in that order: by
// the time a reader sees MessageAdded, Messages already holds it. The other
// order hands a handler news of a message and a conversation without it.
func (a *Agent) add(emit func(Event), msg ai.Message) {
	a.mu.Lock()
	a.messages = append(a.messages, msg)
	a.mu.Unlock()

	emit(MessageAdded{Turn: a.turnNow(), Message: msg})
}

// never is longer than any process runs, so a timer set to it is one that does
// not fire — which is what a caller who turned a timeout off asked for.
const never = time.Duration(math.MaxInt64)

// A turn is one exchange: someone said something, and the loop runs until the
// model stops asking for tools. It holds as many inferences as the tools
// require — that is the difference between the two words.

// reason asks the model what to do next: one call, retrying a stream that
// failed retryably. It returns the response and not its message, because the
// caller needs both halves — what the model said, and why it stopped.
func (a *Agent) reason(ctx context.Context, emit func(Event)) (*ai.Response, ai.Usage, error) {
	var spent ai.Usage
	var lastErr error

	wait := a.retryBackoff
	for attempt := 1; attempt <= a.retryAttempts; attempt++ {
		// Rebuilt every attempt, so no hook is handed its own last edit.
		inf := a.inference()
		if err := a.preInfer(ctx, inf); err != nil {
			return nil, spent, err
		}
		emit(MessageStart{Turn: a.turnNow(), Attempt: attempt, Inference: inf})

		resp, err := a.stream(ctx, emit, inf)

		if ctx.Err() != nil {
			// Abandoned, not failed — but the span still closes: the reader
			// that saw it open has not gone anywhere.
			emit(MessageEnd{Turn: a.turnNow(), Attempt: attempt, Inference: inf,
				Response: resp, Err: ctx.Err()})
			return nil, spent, ctx.Err()
		}

		// A failed call is paid for too: the tokens are spent either way.
		if resp != nil {
			spent.Add(resp.Usage)
		}

		// Read before PostInfer can overwrite err: a failed call earns another
		// go, an objection to the answer does not.
		retry := err != nil && ai.IsRetryable(err)
		if err == nil {
			err = a.postInfer(ctx, resp)
		}

		emit(MessageEnd{Turn: a.turnNow(), Attempt: attempt, Inference: inf,
			Response: resp, Err: err})

		switch {
		case retry:
			lastErr = err
			if attempt == a.retryAttempts {
				break // nothing to wait for
			}
			if err := pause(ctx, wait, err); err != nil {
				return nil, spent, err
			}
			wait *= 2
		case err != nil:
			return nil, spent, err
		default:
			return resp, spent, nil
		}
	}
	return nil, spent, lastErr
}

// PanicError is a tool that panicked. Error is deliberately one line, because
// it is what the model is told and a stack trace is neither useful to it nor
// cheap to send; Stack is the whole thing, for whoever is watching ToolEnd.
type PanicError struct {
	Tool  string
	Value any
	Stack []byte
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("tool %s panicked: %v", e.Tool, e.Value)
}

// errStalled is a stream that has stopped saying anything: the connection is
// open, nothing is arriving, and nobody has raised an error. It is a network
// failure because that is what it is, and it is one value rather than one
// minted per stall so that what ended a call is compared, not parsed.
var errStalled = &ai.Error{Kind: ai.KindNetwork, Message: "agent: the stream went silent"}

// errNoResponse is a stream that ended without ever finishing. ai.Collect
// calls the same failure retryable, and so does this — a bare error here would
// be read as permanent by ai.IsRetryable.
var errNoResponse = &ai.Error{Kind: ai.KindNetwork, Message: "agent: the stream ended without a response"}

// stream makes one model call. Silence is bounded at both ends — streamFirst
// before the endpoint says anything, streamIdle between events once it has —
// and running out cancels the stream, only the stream, with errStalled as the
// cause. Why a call ended is then read off the context that ended it.
func (a *Agent) stream(ctx context.Context, emit func(Event), inf *Inference) (*ai.Response, error) {
	streamCtx, stop := context.WithCancelCause(ctx)
	defer stop(nil)

	quiet := time.AfterFunc(a.streamFirst, func() { stop(errStalled) })
	defer quiet.Stop()

	var resp *ai.Response
	var err error
	for evt, streamErr := range a.client.Stream(streamCtx, inf.Messages, inf.options()...) {
		quiet.Reset(a.streamIdle)
		if streamErr != nil {
			// A failed call still spent tokens and may have produced text.
			resp, err = evt.Response, streamErr
			continue
		}
		// Fragments only; the finished response is announced by the caller,
		// once PostInfer has had it.
		switch evt.Type {
		case ai.EventBlockStart, ai.EventBlockDelta, ai.EventBlockEnd:
			emit(MessageUpdate{Delta: evt})
		case ai.EventDone:
			resp = evt.Response
		}
	}

	switch {
	case errors.Is(context.Cause(streamCtx), errStalled):
		return resp, errStalled
	case err != nil:
		return resp, err
	case resp == nil:
		return nil, errNoResponse
	}
	return resp, nil
}

// call is one tool call and what became of it, kept in the model's order.
type call struct {
	ai.ToolCall // as any PreTool hook rewrote it

	tool   Tool // nil when the model named one that does not exist
	result Result

	// err before the call runs is why it never did; after, how it failed.
	err error
	// stop is the Terminate votes on this call — gate, tool, hook — or'd
	// together in finish.
	stop bool
}

// act runs the tools a model asked for: vet the batch, run what survives,
// close each as it lands, reply.
func (a *Agent) act(ctx context.Context, emit func(Event), calls []ai.ToolCall) ([]ai.ToolResult, bool) {
	batch := make([]call, len(calls))
	messages := a.Messages()

	// Vet: the batch cannot choose a concurrency until it knows every tool.
	// One at a time here, so a gate never reasons about concurrency.
	for i := range calls {
		batch[i] = call{ToolCall: calls[i]}
		c := &batch[i]
		emit(ToolStart{Turn: a.turnNow(), ID: c.ID, Name: c.Name, Args: c.Input})

		tool, ok := a.toolNamed(c.Name)
		if !ok {
			c.err = fmt.Errorf("no tool named %q is available", c.Name)
			continue
		}
		c.tool = tool

		// Arguments are model output, so checking them turns a mistake the
		// model could correct into a tool error it sees.
		if c.err = tool.Schema().Validate(c.Input); c.err != nil {
			continue
		}

		// First refusal is final: a gate that gets weaker as you add to it is
		// not a gate.
		for _, h := range a.hookSet() {
			if h.PreTool == nil {
				continue
			}
			decision, err := h.PreTool(ctx, PreToolContext{
				Call: c.ToolCall, Tool: tool, Messages: messages,
			})
			if err != nil {
				c.err = err
				break
			}
			c.stop = c.stop || decision.Terminate
			if decision.Block {
				reason := decision.Reason
				if reason == "" {
					reason = "blocked"
				}
				c.err = errors.New(reason)
				break
			}
			// Rewrites chain: the next hook sees what this one produced.
			if decision.Arguments != "" {
				c.Input = decision.Arguments
			}
		}
	}

	// Closing a call: its span, the after-hooks, its vote. Refused above or
	// finished below, it closes the same way.
	finish := func(c *call) {
		emit(ToolEnd{Turn: a.turnNow(), ID: c.ID, Name: c.Name, Args: c.Input,
			Result: c.result, Err: c.err})

		// Chained: each hook is handed what the one before it produced.
		for _, h := range a.hookSet() {
			if h.PostTool == nil {
				continue
			}
			replacement, err := h.PostTool(ctx, PostToolContext{
				Call: c.ToolCall, Tool: c.tool, Result: c.result, Err: c.err, Messages: messages,
			})
			if err != nil {
				c.err = err
				break
			}
			if replacement != nil {
				c.result = *replacement
			}
		}

		// Every vote is in: the gate's already, the tool's and the hook's now.
		c.stop = c.stop || c.result.Terminate
	}

	var pending []int
	for i := range batch {
		if batch[i].err != nil {
			finish(&batch[i]) // nothing will run; close it now
			continue
		}
		pending = append(pending, i)
	}

	// Run: one sequential tool makes the whole batch sequential.
	if len(pending) > 0 {
		parallel := len(pending) > 1
		for _, i := range pending {
			if isSequential(batch[i].tool) {
				parallel = false
				break
			}
		}

		// One channel back, because reporting is on the turn's goroutine and a
		// tool is on its own. Each owns its element of batch and says which
		// when done; a Report rides the same channel and is dropped rather
		// than stalling the tool.
		type update struct {
			index   int
			partial *Result // non-nil is progress; otherwise the call finished
		}
		ch := make(chan update, len(pending)*2)

		run := func(i int) {
			c := &batch[i]
			// The caller's code on a goroutine this package created — the one
			// place a panic cannot be recovered by whoever wrote it, and
			// unrecovered it takes the process down. A tool already has a way
			// to fail, so a panic becomes that.
			defer func() {
				if p := recover(); p != nil {
					c.result, c.err = Result{}, &PanicError{
						Tool: c.Name, Value: p, Stack: debug.Stack(),
					}
					ch <- update{index: i}
				}
			}()
			rctx := context.WithValue(ctx, reporter{}, func(partial Result) {
				select {
				case ch <- update{index: i, partial: &partial}:
				default:
				}
			})
			c.result, c.err = c.tool.Run(rctx, c.ToolCall)
			ch <- update{index: i}
		}

		// Both modes feed the same channel, so sequential is a pool of one.
		if parallel {
			for _, i := range pending {
				go run(i)
			}
		} else {
			go func() {
				for _, i := range pending {
					run(i)
				}
			}()
		}

		for remaining := len(pending); remaining > 0; {
			u := <-ch
			if u.partial != nil {
				emit(ToolUpdate{Turn: a.turnNow(), ID: batch[u.index].ID,
					Name: batch[u.index].Name, Partial: *u.partial})
				continue
			}
			remaining--
			finish(&batch[u.index])
		}
	}

	// Reply: in the model's order, and the batch stops only on a full vote.
	results := make([]ai.ToolResult, len(batch))
	terminate := true
	for i := range batch {
		c := &batch[i]
		results[i] = ai.ToolResult{
			ToolCallID: c.ID,
			ToolName:   c.Name,
			Content:    ResultText(c.result, c.err),
			IsError:    c.err != nil,
		}
		terminate = terminate && c.stop
	}
	return results, terminate
}

// turn runs one exchange: the input goes in, then reason and act alternate
// until the model stops asking for tools or the step budget runs out.
//
// ctx is this turn's own, so cancelling it is what Interrupt does. Reporting
// does not go through it: an interrupted turn still has a reader.
func (a *Agent) turn(ctx context.Context, emit func(Event), in []ai.Message) (out TurnEnd) {
	// The turn's own context, so Interrupt ends this exchange and not whatever
	// called it. Derived here and not by the callers, because two of them
	// deriving it is two chances to disagree on what Interrupt reaches.
	turnCtx, stopTurn := context.WithCancel(ctx)
	defer func() {
		stopTurn()
		// Put the no-op back. A CancelFunc closes over its context, so
		// keeping the finished turn's would hold the caller's whole context
		// chain alive for as long as the agent is.
		a.mu.Lock()
		a.stopTurn = func() {}
		a.mu.Unlock()
	}()

	a.mu.Lock()
	a.stopTurn = stopTurn
	a.mu.Unlock()

	a.turnCount.Add(1)
	emit(TurnStart{Turn: a.turnNow()})

	// A history replaced between exchanges is announced here, the first place
	// there is anywhere to announce it. Doing it at SetMessages would need a
	// callback; not doing it is what made a compacted session hand back what
	// the agent threw away.
	if replaced, msgs := a.takeReplaced(); replaced {
		emit(MessagesReplaced{Turn: a.turnNow(), Messages: msgs})
	}

	// Read again rather than pinned: only this function advances it, so both
	// ends of the turn carry the same number.
	defer func() {
		out.Turn = a.turnNow()
		emit(out)
	}()

	for _, m := range in {
		a.add(emit, m)
	}

	for step := 0; ; step++ {
		if a.maxSteps > 0 && step >= a.maxSteps {
			return out.stopped(StopMaxSteps)
		}

		if turnCtx.Err() != nil {
			return out.canceled(turnCtx)
		}

		// Anything added while the last tools ran enters here, not mid-stream.
		for _, m := range a.taken() {
			a.add(emit, m)
		}

		resp, spent, err := a.reason(turnCtx, emit)
		out.Usage.Add(spent)
		switch {
		case turnCtx.Err() != nil:
			return out.canceled(turnCtx)
		case err != nil:
			return out.failed(err)
		}

		msg := resp.Message()
		out.Message = msg
		a.add(emit, msg)

		calls := msg.ToolCalls()
		if len(calls) == 0 {
			return out.stopped(endedBecause(resp.StopReason))
		}

		results, terminate := a.act(turnCtx, emit, calls)
		a.add(emit, ai.ToolResultsMessage(results...))
		if terminate {
			return out.stopped(StopTerminated)
		}
	}
}

// StopReason says why an exchange ended.
type StopReason string

const (
	// StopEndTurn is the model answering without asking for another tool.
	StopEndTurn StopReason = "end_turn"
	// StopMaxTokens is the model running out of output room mid-answer. The
	// text that arrived is in the conversation, and it is not a whole reply.
	StopMaxTokens StopReason = "max_tokens"
	// StopRefusal is the model declining to answer, or a provider's content
	// filter stopping it. There may be text; it is not the answer.
	StopRefusal StopReason = "refusal"
	// StopSequence is generation stopping at one of WithStopSequences.
	StopSequence StopReason = "stop_sequence"
	// StopMaxSteps is the step budget running out with the model still working.
	StopMaxSteps StopReason = "max_steps"
	// StopTerminated is every tool in a batch asking the loop not to continue.
	StopTerminated StopReason = "terminated"
	// StopError is an inference that failed past its retry budget.
	StopError StopReason = "error"
	// StopCanceled is the context ending mid-exchange.
	StopCanceled StopReason = "canceled"
)

// pause waits before another attempt: as long as the endpoint asked for, or
// the caller's doubling backoff when it did not say. Reading RetryAfter is the
// difference between retrying and hammering — a 429 that names a delay and is
// replayed immediately is what rate limiting exists to punish.
func pause(ctx context.Context, backoff time.Duration, failure error) error {
	if after := ai.RetryAfter(failure); after > 0 {
		backoff = after
	}
	if backoff <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(backoff)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// endedBecause carries the model's reason for stopping up to the turn. Only a
// model that chose to stop is an end_turn: one that ran out of room, refused,
// or hit a caller's stop sequence produced something that is not a whole
// reply, and reporting any of those as end_turn tells a caller the answer is
// complete when it is cut off.
func endedBecause(reason ai.StopReason) StopReason {
	switch reason {
	case ai.StopMaxTokens:
		return StopMaxTokens
	case ai.StopRefusal:
		return StopRefusal
	case ai.StopSequence:
		return StopSequence
	default:
		return StopEndTurn
	}
}

// stopped, failed and canceled are the three ways a turn ends, one line each at
// the call site — so no exit can forget to say why.

func (e TurnEnd) stopped(reason StopReason) TurnEnd {
	e.StopReason = reason
	return e
}

func (e TurnEnd) failed(err error) TurnEnd {
	e.StopReason = StopError
	e.Err = err
	return e
}

func (e TurnEnd) canceled(ctx context.Context) TurnEnd {
	e.StopReason = StopCanceled
	e.Err = ctx.Err()
	return e
}
