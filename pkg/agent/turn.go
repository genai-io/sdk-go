package agent

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// A turn is one exchange: someone said something, and the loop runs until the
// model stops asking for tools. It holds as many inferences as the tools
// require — that is the difference between the two words.

// options is what a request contributes to a call: the prompt and the toolset.
// Everything else — temperature, token ceilings, effort — belongs to the
// client, and a second place to configure the same call would only drift.
func options(req *ai.Request) []ai.Option {
	opts := make([]ai.Option, 0, 2)
	if req.System != "" {
		opts = append(opts, ai.WithSystem(req.System))
	}
	if len(req.Tools) > 0 {
		opts = append(opts, ai.WithTools(req.Tools...))
	}
	return opts
}

// reason asks the model what to do next: one call, returning the message and
// what it cost, retrying a stream that failed retryably. The message is either
// an answer or a request to act. opts apply to this call only.
func (a *Agent) reason(ctx context.Context, opts ...ai.Option) (ai.Message, ai.Usage, error) {
	var spent ai.Usage
	var lastErr error

	for attempt := 1; attempt <= a.maxAttempts; attempt++ {
		// Rebuilt every attempt, so no hook is handed its own last edit. A
		// refusal returns before anything is announced.
		req := a.request()
		if err := a.preInfer(ctx, req); err != nil {
			return ai.Message{}, spent, err
		}
		a.emit(ctx, MessageStart{Attempt: attempt, Request: req})

		// The stream gets a context of its own so a stall can end it without
		// ending the turn.
		streamCtx, stopStream := context.WithCancel(ctx)
		quiet := watch(a.streamFirst, a.streamIdle, stopStream)

		var resp *ai.Response
		var err error
		for evt, streamErr := range a.client.Stream(streamCtx, req.Messages, append(options(req), opts...)...) {
			quiet.beat()
			if streamErr != nil {
				// A failed call still spent tokens and may have produced text.
				resp, err = evt.Response, streamErr
				continue
			}
			// Only fragments go out from here. The finished response is a
			// conclusion, announced below once PostInfer has had it.
			switch evt.Type {
			case ai.EventBlockStart, ai.EventBlockDelta, ai.EventBlockEnd:
				a.emit(ctx, MessageUpdate{Delta: evt})
			case ai.EventDone:
				resp = evt.Response
			}
		}
		stopStream()

		// A stall reads as a cancelled stream, which says nothing about why.
		// Naming it makes the attempt retryable, which is what it should be.
		if quiet.fired() {
			err = &ai.Error{Kind: ai.KindNetwork, Message: "agent: the stream went silent"}
		}

		if ctx.Err() != nil {
			return ai.Message{}, spent, ctx.Err()
		}

		// A failed call is paid for too: the tokens are spent either way.
		if resp != nil {
			spent.Add(resp.Usage)
		} else if err == nil {
			err = errors.New("agent: the stream ended without a response")
		}

		// Read before PostInfer can put a different error in err: a failure of
		// the call earns another go, an objection to the answer does not.
		retry := err != nil && ai.IsRetryable(err)
		if err == nil {
			err = a.postInfer(ctx, resp)
		}

		a.emit(ctx, MessageEnd{Response: resp, Err: err})

		switch {
		case retry:
			lastErr = err
		case err != nil:
			return ai.Message{}, spent, err
		default:
			return resp.Message(), spent, nil
		}
	}
	return ai.Message{}, spent, lastErr
}

// call is one tool call and what became of it, kept in the model's order.
type call struct {
	ai.ToolCall // as any PreTool hook rewrote it

	tool   Tool // nil when the model named one that does not exist
	result Result

	// err before the call runs is why it never did; after, how it failed.
	err error
	// stop is the Terminate votes on this call, or'd together — the gate's,
	// the tool's, the hook's. Tallied in finish.
	stop bool
}

// act runs the tools a model asked for: vet the batch, run what survives,
// close each as it lands, reply.
func (a *Agent) act(ctx context.Context, calls []ai.ToolCall) ([]ai.ToolResult, bool) {
	batch := make([]call, len(calls))
	messages := a.Messages()

	// Vet: decided before anything runs, because the batch cannot choose a
	// concurrency until it knows every tool. One at a time on this goroutine,
	// so a gate never has to reason about concurrency.
	for i := range calls {
		batch[i] = call{ToolCall: calls[i]}
		c := &batch[i]
		a.emit(ctx, ToolStart{ID: c.ID, Name: c.Name, Args: c.Input})

		tool, ok := a.toolNamed(c.Name)
		if !ok {
			c.err = fmt.Errorf("no tool named %q is available", c.Name)
			continue
		}
		c.tool = tool

		// Arguments are model output, so they are wrong sometimes. Checking
		// them turns a mistake the model could correct into a tool error it
		// sees, rather than whatever the tool does with them.
		if c.err = tool.Definition().ValidateArgs(c.Input); c.err != nil {
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
	// finished below, a call closes the same way.
	finish := func(c *call) {
		a.emit(ctx, ToolEnd{ID: c.ID, Result: c.result, Err: c.err})

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

		// The tally, where every vote is in: the gate's already, the tool's
		// and the hook's now.
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
			if _, ok := batch[i].tool.(sequential); ok {
				parallel = false
				break
			}
		}

		// Each goroutine owns one element of batch and says which when done.
		// Progress goes straight to the reader — the loop would only forward
		// it, and being droppable it never blocks the tool.
		done := make(chan int, len(pending)) // one each, so no tool waits to report

		run := func(i int) {
			c := &batch[i]
			c.result, c.err = c.tool.Run(ctx, c.ToolCall, func(partial Result) {
				a.emit(ctx, ToolUpdate{ID: c.ID, Partial: partial})
			})
			done <- i
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

		for range pending {
			finish(&batch[<-done])
		}
	}

	// Reply: in the model's order, and the batch stops only on a full vote.
	results := make([]ai.ToolResult, len(batch))
	terminate := true
	for i := range batch {
		c := &batch[i]
		text := c.result.Text()
		if text == "" {
			if c.err != nil {
				text = c.err.Error()
			} else {
				// Several endpoints reject a result with no content.
				text = "(no output)"
			}
		}
		results[i] = ai.ToolResult{
			ToolCallID: c.ID,
			ToolName:   c.Name,
			Content:    text,
			IsError:    c.err != nil,
		}
		terminate = terminate && c.stop
	}
	return results, terminate
}

// watch cancels a stream that goes quiet: first bounds how long it may take to
// say anything, idle how long it may pause once it has started. Either at zero
// turns that half off; both at zero and there is no watchdog at all.
func watch(first, idle time.Duration, cancel context.CancelFunc) *watchdog {
	if first <= 0 && idle <= 0 {
		return nil
	}
	if first <= 0 {
		first = idle
	}
	w := &watchdog{idle: idle}
	w.timer = time.AfterFunc(first, func() {
		w.stalled.Store(true)
		cancel()
	})
	return w
}

// A nil watchdog is one that was never asked for, so every method takes that
// case — the caller has no branch of its own.
type watchdog struct {
	timer   *time.Timer
	idle    time.Duration
	stalled atomic.Bool
}

func (w *watchdog) beat() {
	if w != nil && w.idle > 0 {
		w.timer.Reset(w.idle)
	}
}

func (w *watchdog) fired() bool {
	if w == nil {
		return false
	}
	w.timer.Stop()
	return w.stalled.Load()
}

// turn runs one exchange: the input goes in, then reason and act alternate
// until the model stops asking for tools or the step budget runs out.
// work is the turn's own context: cancelling it ends this exchange and leaves
// the run alive, which is what Interrupt does. ctx is the run's, and only the
// closing report uses it — an interrupted turn still has a reader.
func (a *Agent) turn(ctx, work context.Context, in []ai.Message) (out TurnEnd) {
	a.emit(work, TurnStart{Turn: int(a.turnCount.Load())})

	// A turn the context killed reports nothing further: the reader is gone.
	// The count is read again rather than pinned: only Run advances it, and it
	// is inside this call, so both ends carry the same number.
	defer func() {
		if ctx.Err() == nil {
			out.Turn = int(a.turnCount.Load())
			a.emit(ctx, out)
		}
	}()

	for _, m := range in {
		a.add(work, m)
	}

	for step := 0; ; step++ {
		if a.maxSteps > 0 && step >= a.maxSteps {
			return out.stopped(StopMaxSteps)
		}

		// Anything that arrived while the last tools ran lands here rather
		// than mid-stream: changing what the model is about to see is safe
		// exactly once per inference, at the boundary.
		if work.Err() != nil {
			return out.canceled(work)
		}
		for _, m := range drain(a.in) {
			a.add(work, m)
		}

		msg, spent, err := a.reason(work)
		out.Usage.Add(spent)
		switch {
		case work.Err() != nil:
			return out.canceled(work)
		case err != nil:
			return out.failed(err)
		}

		a.add(ctx, msg)

		calls := msg.ToolCalls()
		if len(calls) == 0 {
			return out.stopped(StopEndTurn)
		}

		results, terminate := a.act(work, calls)
		a.add(work, ai.ToolResultsMessage(results...))
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
	// StopMaxSteps is the step budget running out with the model still working.
	StopMaxSteps StopReason = "max_steps"
	// StopTerminated is every tool in a batch asking the loop not to continue.
	StopTerminated StopReason = "terminated"
	// StopError is an inference that failed past its retry budget.
	StopError StopReason = "error"
	// StopCanceled is the context ending mid-exchange.
	StopCanceled StopReason = "canceled"
)

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
