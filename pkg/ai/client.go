package ai

import (
	"context"
	"fmt"
	"iter"
	"slices"
)

// Client turns a Driver into a usable model handle.
//
// It owns everything that is the same for every protocol: applying per-client
// and per-model defaults to a request, and accumulating the driver's deltas
// into a Response. A Client is safe for concurrent use; it holds no per-call
// state.
type Client struct {
	driver   Driver
	model    Model
	defaults []Option
}

// New wraps a Driver and the Model it was configured for. Any options given
// here become this client's defaults, which a per-call option overrides.
func New(d Driver, m Model, defaults ...Option) *Client {
	return &Client{driver: d, model: cloneModel(m), defaults: slices.Clone(defaults)}
}

// Model describes the model this client talks to. The value is a detached
// snapshot, so editing it changes nothing here — a client is bound to one
// model for its lifetime, and this hands back a description of that binding,
// not a handle on it.
//
// The copy is not ceremony: Model carries slices and maps, so returning it
// directly would let one caller's edit reach into what every other caller
// reads. What to do instead depends on what you meant:
//
//	client.Complete(ctx, msgs, ai.WithEffort(ai.EffortHigh)) // change a setting for a call
//	other := ai.New(driver, tweaked)                         // talk to a different model
func (c *Client) Model() Model { return cloneModel(c.model) }

// Driver returns the underlying driver.
func (c *Client) Driver() Driver { return c.driver }

// ContextWindow returns the model's maximum input tokens, or 0 when unknown.
// Callers sizing a conversation against it must treat 0 as "cannot tell" — a
// substituted guess is acted on silently and is wrong in both directions.
func (c *Client) ContextWindow() int { return c.model.ContextWindow }

// unsupportedBy reports that a driver does not implement an optional
// capability, naming both so the message says what to do about it.
func unsupportedBy(d Driver, capability string) error {
	return &Error{
		Driver: d.Name(), Kind: KindUnsupported,
		Message: fmt.Sprintf("ai: driver %q does not support %s", d.Name(), capability),
	}
}

// Models lists the models the underlying endpoint currently serves.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	lister, ok := c.driver.(ModelLister)
	if !ok {
		return nil, unsupportedBy(c.driver, "model listing")
	}
	models, err := lister.Models(ctx)
	if err != nil {
		return nil, err
	}
	return cloneModels(models), nil
}

// Stream runs one inference call and yields events as they arrive.
//
// The iterator ends after EventDone, or immediately after yielding a non-nil
// error. Abandoning it (break, return) cancels the underlying request on the
// next driver send. Every event but EventDone is a view of work in progress;
// EventDone carries the aggregated Response, which is also what Complete
// returns.
//
// Passing no options runs on the client's and model's defaults alone.
func (c *Client) Stream(ctx context.Context, messages []Message, opts ...Option) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		resp := &Response{Model: c.model.ID}

		blocks := blockTracker{yield: yield, content: &resp.Content}

		// finish stamps whatever arrived so far onto the response. A failed
		// turn still streamed text and still burned tokens; handing back only
		// an error throws away both the partial answer and the accounting for
		// it.
		finish := func(err error) *Response {
			resp.Err = err
			switch {
			case err == nil:
			case IsKind(err, KindCanceled), ctx.Err() != nil:
				resp.StopReason = StopAborted
			default:
				resp.StopReason = StopError
			}
			if resp.StopReason == "" {
				// A protocol that reports no finish reason still told us what
				// happened: pending tool calls mean the turn is waiting on us.
				if resp.Content.Has(BlockToolCall) {
					resp.StopReason = StopToolUse
				} else {
					resp.StopReason = StopEndTurn
				}
			}
			return resp
		}

		req, err := c.prepare(ctx, messages, opts)
		if err != nil {
			yield(Event{Type: EventDone, Response: finish(err)}, err)
			return
		}

		for delta, err := range c.driver.Stream(ctx, req) {
			if err != nil {
				// One yield carries both: the error fires the caller's error
				// branch, and the event carries what was produced before it.
				blocks.close()
				yield(Event{Type: EventDone, Response: finish(err)}, err)
				return
			}
			if delta.Block.Type != "" {
				if !blocks.write(delta.Block) {
					return
				}
			}
			if delta.EndBlock && !blocks.close() {
				return
			}
			if delta.Usage != nil {
				mergeUsage(&resp.Usage, *delta.Usage)
			}
			if delta.StopReason != "" {
				resp.StopReason = delta.StopReason
			}
			if delta.Model != "" {
				resp.Model = delta.Model
			}
			if delta.ID != "" {
				resp.ID = delta.ID
			}
		}

		if !blocks.close() {
			return
		}
		yield(Event{Type: EventDone, Response: finish(nil)}, nil)
	}
}

// Complete runs one inference call and returns the aggregated response,
// discarding the intermediate events.
//
// On failure it returns a non-nil error *and* a non-nil response carrying
// everything that arrived first: the text already streamed and the tokens
// already billed. Check the error as usual; read the response when you want to
// show the partial answer or account for the spend. Its StopReason is
// StopError or StopAborted, and Response.Err is the same error.
func (c *Client) Complete(ctx context.Context, messages []Message, opts ...Option) (*Response, error) {
	return Collect(c.Stream(ctx, messages, opts...))
}

// Collect drains an event stream and returns its final Response. It is what
// Complete is built from, exposed so a caller who already holds a stream can
// aggregate it without re-running the request.
//
// Like Complete, it returns both a response and an error when the turn failed
// partway.
func Collect(events iter.Seq2[Event, error]) (*Response, error) {
	var resp *Response
	var failure error
	for event, err := range events {
		if event.Type == EventDone && event.Response != nil {
			resp = event.Response
		}
		if err != nil {
			failure = err
			break
		}
	}
	if failure != nil {
		return resp, failure
	}
	if resp == nil {
		// The driver closed its iterator without a terminal error and without
		// completing. Reporting that plainly beats handing back a Response
		// that looks like an empty answer.
		return nil, &Error{Kind: KindNetwork, Message: "stream ended without a completed response"}
	}
	return resp, nil
}

// prepare builds the request for one call, repairs the conversation and
// validates it. Counting and generation both go through it, so a prompt cannot
// be measured differently from how it is sent.
//
// The caller's messages slice is never written to: history repair returns new
// slices rather than editing the old ones, so the conversation the caller
// still holds is left exactly as it was. The ordinary Go contract covers the
// rest — do not mutate what you passed while the call is in flight.
func (c *Client) prepare(ctx context.Context, messages []Message, opts []Option) (*Request, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	req := newRequest(c.model, messages, c.defaults, opts)
	if err := c.model.validateStructure(req); err != nil {
		return nil, err
	}
	// History repair is semantic preparation, not wire translation. Doing it
	// once here makes exact counting, estimated counting, middleware and
	// generation observe the same conversation.
	req.Messages = RepairHistory(req.Messages)
	if err := c.model.validate(req); err != nil {
		return nil, err
	}
	return req, nil
}

// Handler runs one model call: the seam every middleware wraps.
//
// It is the same shape as Driver.Stream, so a driver is already a Handler
// and a middleware chain collapses back to one.
type Handler func(ctx context.Context, req *Request) iter.Seq2[Delta, error]

// Middleware wraps a Handler.
//
// This is where retry, caching, request logging and cost metering belong, and
// they belong to the caller: only the application knows the budget for a turn,
// what may be cached, and what must not be logged.
//
// One rule is not the application's to discover. A retry may only replay a
// call that failed *before producing any delta* — once output has reached the
// caller the answer has begun, and resending would either duplicate the text
// already shown or silently discard it. Check IsRetryable and RetryAfter, and
// give up once anything has streamed.
//
// Middleware runs outermost-first: the first one given sees the request first
// and the deltas last.
type Middleware func(Handler) Handler

// Wrap returns a Driver that runs mw around d, outermost first.
//
// Middleware decorates the driver rather than the client because that is what
// it actually is: a Handler wrapping a Handler. Composing it here keeps it
// visible at the point of construction —
//
//	client := ai.New(ai.Wrap(driver, retry, logging), model)
//
// — instead of hiding it in a client setting, and it means a decorated driver
// can be handed anywhere an undecorated one can.
func Wrap(d Driver, mw ...Middleware) Driver {
	h := Handler(d.Stream)
	for i := len(mw) - 1; i >= 0; i-- {
		h = mw[i](h)
	}
	return wrapped{Driver: d, handler: h}
}

// wrapped forwards the optional driver capabilities as well as Stream: a
// decorated driver that silently stopped being a ModelLister would change what
// the client can do depending on whether middleware happened to be attached.
type wrapped struct {
	Driver
	handler Handler
}

func (w wrapped) Stream(ctx context.Context, req *Request) iter.Seq2[Delta, error] {
	return w.handler(ctx, req)
}

func (w wrapped) Models(ctx context.Context) ([]Model, error) {
	lister, ok := w.Driver.(ModelLister)
	if !ok {
		return nil, unsupportedBy(w.Driver, "model listing")
	}
	return lister.Models(ctx)
}

func (w wrapped) CountTokens(ctx context.Context, req *Request) (int, error) {
	counter, ok := w.Driver.(TokenCounter)
	if !ok {
		return 0, unsupportedBy(w.Driver, "token counting")
	}
	return counter.CountTokens(ctx, req)
}

// EventType identifies one stage in a streamed block's lifecycle.
type EventType string

const (
	// EventBlockStart opens Event.Index with Block.Type set.
	EventBlockStart EventType = "block_start"
	// EventBlockDelta carries an incremental Block fragment.
	EventBlockDelta EventType = "block_delta"
	// EventBlockEnd closes a block and carries its complete value.
	EventBlockEnd EventType = "block_end"
	// EventDone is final and carries the aggregated Response.
	EventDone EventType = "done"
)

// Event is one element of a streamed response. All content kinds share the
// same start/delta/end lifecycle, so consumers need one state machine and can
// switch on Block.Type only when rendering or acting on a complete block.
type Event struct {
	Type  EventType
	Index int
	Block Block

	// Response is set only on EventDone.
	Response *Response
}

// blockTracker assembles the raw Delta fragments a driver emits into canonical
// Content, and exposes the same ordered block lifecycle to stream consumers.
// Client.Stream owns one per call.
type blockTracker struct {
	yield   func(Event, error) bool
	content *Content
	open    *Block
	index   int
	count   int
}

func (b *blockTracker) write(fragment Block) bool {
	if fragment.Type == "" {
		return true
	}
	if fragment.Type != BlockText && fragment.Type != BlockThinking {
		return b.complete(fragment)
	}
	if b.open != nil && b.open.Type != fragment.Type {
		if !b.close() {
			return false
		}
	}
	if b.open == nil {
		b.index = b.count
		b.count++
		b.open = &Block{Type: fragment.Type}
		if !b.yield(Event{Type: EventBlockStart, Index: b.index, Block: *b.open}, nil) {
			return false
		}
	}
	b.open.Text += fragment.Text
	b.open.Signature += fragment.Signature
	return b.yield(Event{Type: EventBlockDelta, Index: b.index, Block: cloneBlock(fragment)}, nil)
}

func (b *blockTracker) complete(block Block) bool {
	if !b.close() {
		return false
	}
	index := b.count
	b.count++
	block = cloneBlock(block)
	if !b.yield(Event{Type: EventBlockStart, Index: index, Block: Block{Type: block.Type}}, nil) {
		return false
	}
	*b.content = append(*b.content, block)
	return b.yield(Event{Type: EventBlockEnd, Index: index, Block: cloneBlock(block)}, nil)
}

func (b *blockTracker) close() bool {
	if b.open == nil {
		return true
	}
	block := cloneBlock(*b.open)
	b.open = nil
	*b.content = append(*b.content, block)
	return b.yield(Event{Type: EventBlockEnd, Index: b.index, Block: cloneBlock(block)}, nil)
}
