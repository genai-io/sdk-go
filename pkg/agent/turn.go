package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// never is longer than any process runs, so a timer set to it is one that does
// not fire — which is what a caller who turned a timeout off asked for.
const never = time.Duration(math.MaxInt64)

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

// reason asks the model what to do next: one call, returning what came back
// and what it cost, retrying a stream that failed retryably.
//
// The response rather than its message, because the caller needs both halves:
// what the model said, and why it stopped saying it.
func (a *Agent) reason(ctx context.Context, opts ...ai.Option) (*ai.Response, ai.Usage, error) {
	var spent ai.Usage
	var lastErr error

	// A stream that says nothing is the one failure that looks like work, so a
	// timer ends it: streamFirst before the endpoint says anything at all,
	// streamIdle between events once it has. Zero means never — a duration no
	// timer reaches, so nothing below has a case for it.
	first, idle := never, never
	if a.streamFirst > 0 {
		first = a.streamFirst
	}
	if a.streamIdle > 0 {
		idle = a.streamIdle
	}

	for attempt := 1; attempt <= a.maxAttempts; attempt++ {
		// Rebuilt every attempt, so no hook is handed its own last edit. A
		// refusal returns before anything is announced.
		req := a.request()
		if err := a.preInfer(ctx, req); err != nil {
			return nil, spent, err
		}
		a.emit(MessageStart{Attempt: attempt, Request: req})

		// The stream gets a context of its own so a stall can end it without
		// ending the turn.
		streamCtx, stopStream := context.WithCancel(ctx)
		var silent atomic.Bool
		quiet := time.AfterFunc(first, func() {
			silent.Store(true)
			stopStream()
		})

		var resp *ai.Response
		var err error
		for evt, streamErr := range a.client.Stream(streamCtx, req.Messages, append(options(req), opts...)...) {
			quiet.Reset(idle)
			if streamErr != nil {
				// A failed call still spent tokens and may have produced text.
				resp, err = evt.Response, streamErr
				continue
			}
			// Only fragments go out from here. The finished response is a
			// conclusion, announced below once PostInfer has had it.
			switch evt.Type {
			case ai.EventBlockStart, ai.EventBlockDelta, ai.EventBlockEnd:
				a.emit(MessageUpdate{Delta: evt})
			case ai.EventDone:
				resp = evt.Response
			}
		}
		quiet.Stop()
		stopStream()

		// A stall reads as a cancelled stream, which says nothing about why.
		// Naming it makes the attempt retryable, which is what it should be.
		if silent.Load() {
			err = &ai.Error{Kind: ai.KindNetwork, Message: "agent: the stream went silent"}
		}

		if ctx.Err() != nil {
			// Abandoned, not failed — but the span still closes, because the
			// reader that saw it open has not gone anywhere.
			a.emit(MessageEnd{Response: resp, Err: ctx.Err()})
			return nil, spent, ctx.Err()
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

		a.emit(MessageEnd{Response: resp, Err: err})

		switch {
		case retry:
			lastErr = err
		case err != nil:
			return nil, spent, err
		default:
			return resp, spent, nil
		}
	}
	return nil, spent, lastErr
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
		a.emit(ToolStart{ID: c.ID, Name: c.Name, Args: c.Input})

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
		a.emit(ToolEnd{ID: c.ID, Result: c.result, Err: c.err})

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

		// Everything comes back through one channel, because reporting happens
		// on the turn's goroutine and a tool runs on its own. Each goroutine
		// owns its element of batch and says which when done; progress rides
		// the same channel, and is dropped rather than stalling a tool for it.
		type update struct {
			index   int
			partial *Result // non-nil is progress; otherwise the call finished
		}
		ch := make(chan update, len(pending)*2)

		run := func(i int) {
			c := &batch[i]
			c.result, c.err = c.tool.Run(ctx, c.ToolCall, func(partial Result) {
				select {
				case ch <- update{index: i, partial: &partial}:
				default:
				}
			})
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
				a.emit(ToolUpdate{ID: batch[u.index].ID, Partial: *u.partial})
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

// turn runs one exchange: the input goes in, then reason and act alternate
// until the model stops asking for tools or the step budget runs out.
// ctx is this turn's own: cancelling it ends the exchange and leaves the run
// alive, which is what Interrupt does. Reporting does not go through it —
// emitting asks the run whether anyone is listening, and an interrupted turn
// still has a reader.
func (a *Agent) turn(ctx context.Context, in []ai.Message) (out TurnEnd) {
	// The turn's own context, so Interrupt can end this exchange without
	// ending whatever called it. Derived here rather than by the callers,
	// because two of them deriving it separately is two chances to disagree
	// about what Interrupt reaches.
	turnCtx, stopTurn := context.WithCancel(ctx)
	defer stopTurn()

	a.mu.Lock()
	a.stopTurn = stopTurn
	a.mu.Unlock()

	a.turnCount.Add(1)
	a.emit(TurnStart{Turn: int(a.turnCount.Load())})

	// The count is read again rather than pinned: only this function advances
	// it, so both ends of the turn carry the same number.
	defer func() {
		out.Turn = int(a.turnCount.Load())
		a.emit(out)
	}()

	for _, m := range in {
		a.add(m)
	}

	for step := 0; ; step++ {
		if a.maxSteps > 0 && step >= a.maxSteps {
			return out.stopped(StopMaxSteps)
		}

		// Anything that arrived while the last tools ran lands here rather
		// than mid-stream: changing what the model is about to see is safe
		// exactly once per inference, at the boundary.
		if turnCtx.Err() != nil {
			return out.canceled(turnCtx)
		}
		for _, m := range a.injected() {
			a.add(m)
		}

		resp, spent, err := a.reason(turnCtx)
		out.Usage.Add(spent)
		switch {
		case turnCtx.Err() != nil:
			return out.canceled(turnCtx)
		case err != nil:
			return out.failed(err)
		}

		msg := resp.Message()
		out.Message = msg
		a.add(msg)

		calls := msg.ToolCalls()
		if len(calls) == 0 {
			// A model that ran out of room did not answer, whatever the text
			// says. Reporting that as end_turn would tell a caller the reply
			// is whole when it is cut off mid-sentence.
			if resp.StopReason == ai.StopMaxTokens {
				return out.stopped(StopMaxTokens)
			}
			return out.stopped(StopEndTurn)
		}

		results, terminate := a.act(turnCtx, calls)
		a.add(ai.ToolResultsMessage(results...))
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
