package agent

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math"
	"reflect"
	"slices"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrBusy means a turn is already running on this agent.
var ErrBusy = errors.New("agent: a turn is already running")

// Run advances the conversation one turn: in enters the conversation, then
// the model is called and its tools run until it stops asking for them. The
// last event is TurnEnd, which says how the turn went, error included; the
// iterator's own error is only ErrBusy.
//
//	for e, err := range a.Run(ctx, ai.UserMessage("what changed?")) {
//	    render(e)
//	}
//
// Events arrive on the ranging goroutine. Breaking out of the range ends the
// turn, as Interrupt does, and returns once the running tools have.
func (a *Agent) Run(ctx context.Context, in ...ai.Message) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if !a.running.CompareAndSwap(false, true) {
			yield(nil, ErrBusy)
			return
		}
		defer a.running.Store(false)

		// gone remembers a consumer that broke out of the range: yielding to
		// one again is what the iterator forbids, and stopping reading also
		// ends the turn.
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

		a.turn(ctx, emit, in)
	}
}

// add puts a message into the conversation and reports it, in that order: by
// the time a reader sees MessageAdded, Messages already holds it. The other
// order hands a handler news of a message and a conversation without it.
func (a *Agent) add(emit func(Event), msg ai.Message) {
	if msg.ID == "" && a.mintID != nil {
		msg.ID = a.mintID()
	}

	a.mu.Lock()
	a.messages = append(a.messages, msg)
	a.mu.Unlock()

	emit(MessageAdded{Turn: a.turnNow(), Message: msg})
}

// settle brings the conversation up to date before the loop reads it: a
// replacement first, because anything queued behind it belongs to the
// conversation that replaced and not to the one thrown away.
//
// It runs at every step boundary, which is where a turn opens too — the
// one place it is safe to change what the model is about to see, and the last
// moment a replacement can be announced before messages are appended to it.
func (a *Agent) settle(emit func(Event)) {
	if replaced, msgs := a.takeReplaced(); replaced {
		emit(MessagesReplaced{Turn: a.turnNow(), Messages: msgs})
	}
	for _, m := range a.takePending() {
		a.add(emit, m)
	}
}

// preStep runs the hooks that may replace the conversation, which is the other
// half of the boundary settle opens: what came from outside has landed, and
// this is where the caller says what the step should send instead.
//
// A replacement is applied and announced here, where it was made. Nothing has
// been appended against it in between, so a fold has nothing to get wrong.
func (a *Agent) preStep(ctx context.Context, emit func(Event)) error {
	hooks := a.hookSet()
	if !slices.ContainsFunc(hooks, func(h Hook) bool { return h.PreStep != nil }) {
		// Measuring is cheap but not free, and an agent with no PreStep hook
		// should not pay for a figure nobody reads.
		return nil
	}

	c := a.stepContext(nil)

	return a.compacting(ctx, emit,
		func() ([]ai.Message, int) { return c.Messages, c.Tokens },
		func(hctx context.Context) error {
			changed := false
			for _, h := range hooks {
				if h.PreStep == nil {
					continue
				}
				msgs, err := h.PreStep(hctx, c)
				if err != nil {
					return err
				}
				if msgs == nil {
					continue
				}
				// The next hook is asked about what this one left, priced
				// again: the figure from before a shortening would have it
				// shorten twice.
				c = a.stepContext(msgs)
				changed = true
			}
			if changed {
				a.replace(emit, c.Messages)
			}
			return nil
		})
}

// compacting runs the hooks in run under a context whose Compacting opens a
// compaction span, and closes that span however they return — a consumer that
// drew a spinner has to be told to stop whichever way it went. A span opens at
// most once, however many hooks announce: what is drawn is one wait, not one
// per hook.
//
// subject is what the span carries and is asked for only if one opens, so a
// boundary nobody announces at is never priced.
func (a *Agent) compacting(ctx context.Context, emit func(Event),
	subject func() ([]ai.Message, int), run func(context.Context) error) error {

	var span *CompactionStart
	hctx := context.WithValue(ctx, compactionKey{}, func() {
		if span != nil {
			return
		}
		msgs, tokens := subject()
		span = &CompactionStart{Turn: a.turnNow(), Messages: msgs, Tokens: tokens}
		emit(*span)
	})

	err := run(hctx)
	if span != nil {
		emit(CompactionEnd{
			Turn: span.Turn, Messages: span.Messages, Tokens: span.Tokens, Err: err,
		})
	}
	return err
}

// replace applies a conversation a hook handed back and announces it where the
// change was made. Nothing has been appended against the new one in between,
// so a fold has nothing to get wrong.
func (a *Agent) replace(emit func(Event), msgs []ai.Message) {
	a.SetMessages(msgs)
	if replaced, m := a.takeReplaced(); replaced {
		emit(MessagesReplaced{Turn: a.turnNow(), Messages: m})
	}
}

// onInferError asks the caller what to do about a call that failed, and applies
// the answer. Nil is agreement to give up, which is also what an agent with no
// such hook says.
func (a *Agent) onInferError(ctx context.Context, emit func(Event),
	inf *Inference, err error, attempt int) (*Recovery, error) {

	hooks := a.hookSet()
	if !slices.ContainsFunc(hooks, func(h Hook) bool { return h.OnInferError != nil }) {
		return nil, nil
	}

	// Where the call actually went, so a hook reading a context window reads
	// the window of the model that refused it.
	c := InferErrorContext{
		Inference: inf,
		Client:    a.clientFor(inf),
		Err:       err,
		Attempt:   attempt,
		Messages:  a.Messages(),
	}

	var again *Recovery
	hookErr := a.compacting(ctx, emit,
		// Priced through the same path a step boundary uses, and only if a
		// hook announces: most failures are not about size, and measuring one
		// that is not is a figure nobody reads.
		func() ([]ai.Message, int) {
			priced := a.stepContext(nil)
			return priced.Messages, priced.Tokens
		},
		func(hctx context.Context) error {
			for _, h := range hooks {
				if h.OnInferError == nil {
					continue
				}
				r, err := h.OnInferError(hctx, c)
				if err != nil {
					return err
				}
				// The first answer is taken, as the first refusal is at the
				// gate: two hooks that both want the next attempt would
				// otherwise have the last one silently win.
				if r != nil {
					again = r
					break
				}
			}
			if again != nil && again.Messages != nil {
				a.replace(emit, again.Messages)
			}
			return nil
		})
	if hookErr != nil {
		return nil, hookErr
	}
	return again, nil
}

// stepContext is what a boundary hands the hooks that may change it: the
// conversation, and what sending it would cost — the system prompt and the
// tool schemas included, since a dozen schemas can outweigh the messages.
// Nil prices the conversation the agent holds.
func (a *Agent) stepContext(msgs []ai.Message) PreStepContext {
	a.mu.Lock()
	defer a.mu.Unlock()

	if msgs == nil {
		msgs = a.messages
	}
	msgs = slices.Clone(msgs)
	return PreStepContext{
		Messages: msgs,
		Tokens: (&ai.Request{
			System: a.system, Messages: msgs, Tools: a.definitions(),
		}).EstimateTokens(),
		Client: a.client,
	}
}

// never is longer than any process runs, so a timer set to it is one that does
// not fire — which is what a caller who turned a timeout off asked for.
const never = time.Duration(math.MaxInt64)

// reason asks the model what to do next: one call, retrying a stream that
// failed retryably. It returns the response and not its message, because the
// caller needs both halves — what the model said, and why it stopped.
func (a *Agent) reason(ctx context.Context, emit func(Event)) (*ai.Response, ai.Usage, []Tool, error) {
	var spent ai.Usage
	// lastErr rides to the next attempt on Inference.LastErr, which is how a
	// hook routes away from an endpoint that just failed.
	var lastErr error

	// replays is what WithRetry's budget has spent on the call as it stands.
	// A recovery changes the call, so it starts over — see Recovery.
	replays, wait := 0, a.retryBackoff

	for attempt := 1; ; attempt++ {
		// Rebuilt every attempt, so no hook is handed its own last edit.
		inf, tools := a.inference()
		inf.LastErr = lastErr
		if err := a.preInfer(ctx, inf); err != nil {
			return nil, spent, nil, err
		}
		registered := a.toolset()
		// Bind the final request once; Options can replace the offered tools too.
		inf.Options = append(inf.Options, func(req *ai.Request) {
			var offered []Tool
			for _, definition := range req.Tools {
				tool, ok := toolNamed(registered, definition.Name)
				if !ok || !reflect.DeepEqual(tool.Schema(), definition) {
					// SetTools without editing this call belongs to the next inference.
					tool, ok = toolNamed(tools, definition.Name)
				}
				if ok {
					offered = append(offered, tool)
				}
			}
			tools = offered
		})
		emit(InferenceStart{Turn: a.turnNow(), Attempt: attempt, Inference: inf})

		resp, err := a.stream(ctx, emit, inf)

		// A failed call is paid for too, and so is an abandoned one.
		if resp != nil {
			spent.Add(resp.Usage)
		}

		if ctx.Err() != nil {
			// Abandoned, not failed — but the span still closes: the reader
			// that saw it open has not gone anywhere.
			emit(InferenceEnd{Turn: a.turnNow(), Attempt: attempt, Inference: inf,
				Response: resp, Err: ctx.Err()})
			return nil, spent, nil, ctx.Err()
		}

		// Read before PostInfer can overwrite err: a failed call earns another
		// go, an objection to the answer does not.
		callErr := err
		if err == nil {
			err = a.postInfer(ctx, resp)
		}

		emit(InferenceEnd{Turn: a.turnNow(), Attempt: attempt, Inference: inf,
			Response: resp, Err: err})

		if err == nil {
			return resp, spent, tools, nil
		}
		if callErr == nil {
			// PostInfer objected to an answer that arrived. Neither budget
			// below is for that, and neither is the hook.
			return nil, spent, nil, err
		}
		lastErr = err

		// The loop's own budget answers first, for the failures it knows how
		// to replay unchanged.
		if ai.IsRetryable(err) && replays < a.retries {
			replays++
			if perr := pause(ctx, wait, err); perr != nil {
				return nil, spent, nil, perr
			}
			wait *= 2
			continue
		}

		// Out of the loop's own answers: ask the caller for one, and end with
		// the failure when there is none.
		again, hookErr := a.onInferError(ctx, emit, inf, err, attempt)
		if hookErr != nil {
			return nil, spent, nil, hookErr
		}
		if again == nil {
			return nil, spent, nil, err
		}
		replays, wait = 0, a.retryBackoff
	}
}

// errStalled is a stream that has stopped saying anything: the connection is
// open, nothing is arriving, and nobody has raised an error. It is a network
// failure because that is what it is, and it is one value rather than one
// minted per stall so that what ended a call is compared, not parsed.
var errStalled = &ai.Error{Kind: ai.KindNetwork, Message: "agent: the stream went silent"}

// errNoResponse is a stream that ended without ever finishing. ai.Client
// calls the same failure retryable, and so does this — a bare error here would
// be read as permanent by ai.IsRetryable.
var errNoResponse = &ai.Error{Kind: ai.KindNetwork, Message: "agent: the stream ended without a response"}

// stream makes one model call. Silence is bounded at both ends — streamFirst
// before the endpoint says anything, streamIdle between events once it has —
// and running out cancels the stream, only the stream, with errStalled as the
// cause. Why a call ended is then read off the context that ended it.
func (a *Agent) stream(ctx context.Context, emit func(Event), inf *Inference) (*ai.Response, error) {
	client := a.clientFor(inf)

	streamCtx, stop := context.WithCancelCause(ctx)
	defer stop(nil)

	quiet := time.AfterFunc(a.streamFirst, func() { stop(errStalled) })
	defer quiet.Stop()

	var resp *ai.Response
	var err error
	for evt, streamErr := range client.Stream(streamCtx, inf.Messages, inf.options()...) {
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
			emit(InferenceUpdate{Turn: a.turnNow(), Delta: evt})
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

// errAbandoned is a call that was vetted and then never run, because a hook
// failed elsewhere in the batch and the turn is ending with it.
var errAbandoned = errors.New("agent: the turn ended before this call ran")

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
// close each as it lands, reply. A hook that failed is returned as the third
// value and ends the turn; nothing else here does.
func (a *Agent) act(ctx context.Context, emit func(Event), calls []ai.ToolCall, tools []Tool) ([]ai.ToolResult, bool, error) {
	batch := make([]call, len(calls))
	messages := a.Messages()

	// hookErr is a hook that could not do its job — infrastructure, not the
	// model's business — so the batch stops and the turn ends with it.
	var hookErr error

	// opened is how many calls have a span to close; an unstarted call is not
	// reported at all.
	opened := 0

	// Vet: the batch cannot choose a concurrency until it knows every tool.
	// One at a time here, so a gate never reasons about concurrency.
	for i := range calls {
		batch[i] = call{ToolCall: calls[i]}
		c := &batch[i]
		emit(ToolStart{Turn: a.turnNow(), ID: c.ID, Name: c.Name, Args: c.Input})
		opened = i + 1

		tool, ok := toolNamed(tools, c.Name)
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
				c.err, hookErr = err, err
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
		if hookErr != nil {
			break
		}
		if c.err == nil && c.Input != calls[i].Input {
			c.err = tool.Schema().Validate(c.Input)
		}
	}

	// A failed hook abandons the batch. Spans that opened still close, since a
	// reader saw them start; no after-hook runs, and the model is told nothing.
	if hookErr != nil {
		for i := range opened {
			c := &batch[i]
			if c.err == nil {
				c.err = errAbandoned
			}
			emit(ToolEnd{Turn: a.turnNow(), ID: c.ID, Name: c.Name, Args: c.Input,
				Result: c.result, Err: c.err})
		}
		return nil, false, hookErr
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
				// The batch is still drained — the tools running are writing
				// to a channel this loop reads — and then the turn ends.
				hookErr = err
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
					c.result, c.err = Result{}, fmt.Errorf("tool %s panicked: %v", c.Name, p)
					ch <- update{index: i}
				}
			}()
			rctx := context.WithValue(ctx, toolRunKey{}, func(partial Result) {
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
					// An ended turn stops the queue here; the calls that never
					// ran answer with ctx.Err(), so the batch answers in full.
					if err := ctx.Err(); err != nil {
						c := &batch[i]
						c.result, c.err = Result{}, err
						ch <- update{index: i}
						continue
					}
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

	if hookErr != nil {
		return nil, false, hookErr
	}

	// Reply: in the model's order, and the batch stops only on a full vote.
	results := make([]ai.ToolResult, len(batch))
	terminate := true
	for i := range batch {
		c := &batch[i]
		results[i] = ai.ToolResult{
			ToolCallID: c.ID,
			ToolName:   c.Name,
			Content:    ResultContent(c.result, c.err),
			IsError:    c.err != nil,
		}
		terminate = terminate && c.stop
	}
	return results, terminate, nil
}

// turn runs one turn: the input goes in, then reason and act alternate
// until the model stops asking for tools or the step budget runs out.
//
// ctx is this turn's own, so cancelling it is what Interrupt does. Reporting
// does not go through it: an interrupted turn still has a reader.
func (a *Agent) turn(ctx context.Context, emit func(Event), in []ai.Message) (out TurnEnd) {
	// The turn's own context, so Interrupt ends this turn and not whatever
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

	// Read again rather than pinned: only this function advances it, so both
	// ends of the turn carry the same number.
	//
	// No reason means the stack is unwinding through here — a hook panicked —
	// and a turn that is exploding has no outcome to report.
	defer func() {
		if out.StopReason == "" {
			return
		}
		// Failed and interrupted turns must leave accepted input in the history too.
		a.settle(emit)
		out.Turn = a.turnNow()
		emit(out)
	}()

	// Whatever changed while the agent was idle lands before this turn's
	// own input: that is the order it was said in.
	a.settle(emit)
	for _, m := range in {
		a.add(emit, m)
	}

	continued := 0
	for step := 0; ; step++ {
		if a.maxSteps > 0 && step >= a.maxSteps {
			return out.stopped(StopMaxSteps)
		}

		if turnCtx.Err() != nil {
			return out.canceled(turnCtx)
		}

		// What changed while the last tools ran lands here, not mid-stream:
		// messages added from outside, and a conversation replaced from there.
		a.settle(emit)
		// And then the hooks that may replace it, once everything that was
		// entering has.
		if err := a.preStep(turnCtx, emit); err != nil {
			return out.failed(err)
		}

		resp, spent, tools, err := a.reason(turnCtx, emit)
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
			// A model stopped by the output cap was interrupted, not finished,
			// so a caller who asked for it gets another step rather than half
			// an answer. The prompt goes into the conversation like any other
			// message, which is what makes the next call see it.
			if resp.StopReason == ai.StopMaxTokens && a.continueTries > continued {
				continued++
				a.add(emit, ai.UserMessage(a.continuePrompt))
				continue
			}
			// Something said while the model answered has not been heard: the
			// turn ends only when nothing is waiting, or it would sit
			// unread until whoever next thinks to start one.
			if a.Pending() > 0 {
				continue
			}
			return out.stopped(endedBecause(resp.StopReason))
		}

		results, terminate, err := a.act(turnCtx, emit, calls, tools)
		if err != nil {
			// A hook failed: nothing is answered to the model, and the
			// unanswered call is ai.Repair's to tidy on the next use.
			return out.failed(err)
		}
		a.add(emit, ai.ToolResultsMessage(results...))
		if terminate {
			return out.stopped(StopTerminated)
		}
	}
}

// StopReason says why a turn ended.
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
	// StopCanceled is the turn being interrupted or its context ending.
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
