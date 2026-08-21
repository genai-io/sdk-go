package llm

import (
	"context"
	"iter"
	"strings"
)

// Client turns a Driver into a usable model handle.
//
// It owns everything that is the same for every protocol: applying per-client
// and per-model defaults to a request, and accumulating the driver's deltas
// into a Response. A Client is safe for concurrent use; it holds no per-call
// state.
type Client struct {
	driver     Driver
	model      Model
	defaults   Options
	middleware []Middleware
}

// Option configures a Client.
type Option func(*Client)

// WithDefaults sets the options applied to any request that leaves a field
// unset. A per-request value always wins, and the model's own limits fill
// whatever neither states.
func WithDefaults(o Options) Option { return func(c *Client) { c.defaults = o } }

// WithMaxTokens sets the default output cap. Zero restores "use the model's
// advertised output limit".
func WithMaxTokens(n int) Option { return func(c *Client) { c.defaults.MaxTokens = n } }

// WithTemperature sets the default temperature.
func WithTemperature(t float64) Option { return func(c *Client) { c.defaults.Temperature = t } }

// WithEffort sets the default reasoning rung.
func WithEffort(e Effort) Option { return func(c *Client) { c.defaults.Effort = e } }

// New wraps a Driver and the Model it was configured for.
func New(d Driver, m Model, opts ...Option) *Client {
	c := &Client{driver: d, model: m}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Model returns the model this client talks to.
func (c *Client) Model() Model { return c.model }

// Driver returns the underlying driver.
func (c *Client) Driver() Driver { return c.driver }

// ContextWindow returns the model's maximum input tokens, or 0 when unknown.
// Callers sizing a conversation against it must treat 0 as "cannot tell" — a
// substituted guess is acted on silently and is wrong in both directions.
func (c *Client) ContextWindow() int { return c.model.ContextWindow }

// Models lists the models the underlying endpoint currently serves.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	return c.driver.Models(ctx)
}

// Stream runs one inference call and yields events as they arrive.
//
// The iterator ends after EventDone, or immediately after yielding a non-nil
// error. Abandoning it (break, return) cancels the underlying request on the
// next driver send. Every event but EventDone is a view of work in progress;
// EventDone carries the aggregated Response, which is also what Complete
// returns.
//
// opts may be nil, meaning the client's and model's defaults alone.
func (c *Client) Stream(ctx context.Context, p *Prompt, opts *Options) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if p == nil {
			p = &Prompt{}
		}
		merged := opts.merged(c.model, c.defaults)

		// Validation sits innermost, next to the driver, so middleware gets to
		// fix a request before it is judged — SimulateSystemPrompt folds a
		// system prompt that would otherwise be rejected here.
		handler := chain(c.validated(c.driver.Generate), c.middleware)

		resp := &Response{Model: c.model.ID}
		var content, thinking strings.Builder

		// Text arrives a block at a time. Tracking the open one is what lets a
		// consumer distinguish a new paragraph from a continuation.
		blocks := blockTracker{yield: yield}

		// finish stamps whatever arrived so far onto the response. A failed
		// turn still streamed text and still burned tokens; handing back only
		// an error throws away both the partial answer and the accounting for
		// it.
		finish := func(err error) *Response {
			resp.Content = content.String()
			resp.Thinking = thinking.String()
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
				if len(resp.ToolCalls) > 0 {
					resp.StopReason = StopToolUse
				} else {
					resp.StopReason = StopEndTurn
				}
			}
			return resp
		}

		for delta, err := range handler(ctx, p, merged) {
			if err != nil {
				// One yield carries both: the error fires the caller's error
				// branch, and the event carries what was produced before it.
				blocks.close()
				yield(Event{Type: EventDone, Response: finish(err)}, err)
				return
			}
			if delta.Text != "" {
				content.WriteString(delta.Text)
				if !blocks.write(EventTextDelta, delta.Text) {
					return
				}
			}
			if delta.Thinking != "" {
				thinking.WriteString(delta.Thinking)
				if !blocks.write(EventThinkingDelta, delta.Thinking) {
					return
				}
			}
			if delta.EndBlock && !blocks.close() {
				return
			}
			if delta.ThinkingSignature != "" {
				resp.ThinkingSignature += delta.ThinkingSignature
			}
			if delta.ToolCall != nil {
				// A tool call ends whatever text was being written.
				if !blocks.close() {
					return
				}
				call := *delta.ToolCall
				resp.ToolCalls = append(resp.ToolCalls, call)
				if !yield(Event{Type: EventToolCall, Index: blocks.next(), ToolCall: &call}, nil) {
					return
				}
			}
			if delta.Reasoning != nil {
				resp.Reasoning = delta.Reasoning
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

// validated refuses a request the model cannot serve before it reaches the
// network, so nothing is spent on a call that was never going to work.
func (c *Client) validated(next Handler) Handler {
	return func(ctx context.Context, p *Prompt, opts Options) iter.Seq2[Delta, error] {
		if err := c.model.Validate(p, opts); err != nil {
			return func(yield func(Delta, error) bool) { yield(Delta{}, err) }
		}
		return next(ctx, p, opts)
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
func (c *Client) Complete(ctx context.Context, p *Prompt, opts *Options) (*Response, error) {
	return Collect(c.Stream(ctx, p, opts))
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

// mergeUsage applies an update field by field, ignoring zeros. Protocols
// report usage in pieces — Anthropic sends input tokens at message_start and
// output tokens at message_delta — so a whole-struct replace would erase the
// half that arrived first.
func mergeUsage(dst *Usage, src Usage) {
	if src.Input > 0 {
		dst.Input = src.Input
	}
	if src.Output > 0 {
		dst.Output = src.Output
	}
	if src.CacheWrite > 0 {
		dst.CacheWrite = src.CacheWrite
	}
	if src.CacheWrite1h > 0 {
		dst.CacheWrite1h = src.CacheWrite1h
	}
	if src.CacheRead > 0 {
		dst.CacheRead = src.CacheRead
	}
	if src.Reasoning > 0 {
		dst.Reasoning = src.Reasoning
	}
}

// SplitPromptTokens separates a combined prompt count into its fresh and
// cached halves.
//
// Usage.Input is defined as fresh tokens only, with the cached prefix under
// CacheRead — the Anthropic convention. OpenAI-family protocols instead report
// the whole prompt in one figure and expose the cached slice under
// *_tokens_details.cached_tokens; their drivers call this to convert.
//
// fresh + cached always equals the (non-negative) prompt, so TotalInput stays
// exactly what the API reported. The split is defensive about malformed wire
// data: a cached count that is negative, or larger than the prompt, is clamped
// so fresh never goes negative.
func SplitPromptTokens(promptTokens, cachedTokens int) (fresh, cached int) {
	promptTokens = max(promptTokens, 0)
	cached = min(max(cachedTokens, 0), promptTokens)
	return promptTokens - cached, cached
}
