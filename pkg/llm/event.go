package llm

import (
	"context"
	"iter"
	"strings"
)

// EventType identifies what a streamed Event carries.
type EventType string

const (
	// EventTextStart opens a block of answer text. Event.Index identifies it.
	EventTextStart EventType = "text_start"
	// EventTextDelta carries an incremental slice of the answer.
	EventTextDelta EventType = "text_delta"
	// EventTextEnd closes a block of answer text, carrying the whole block in
	// Event.Text.
	EventTextEnd EventType = "text_end"
	// EventThinkingStart opens a block of reasoning text.
	EventThinkingStart EventType = "thinking_start"
	// EventThinkingDelta carries an incremental slice of the reasoning text.
	EventThinkingDelta EventType = "thinking_delta"
	// EventThinkingEnd closes a block of reasoning text.
	EventThinkingEnd EventType = "thinking_end"
	// EventToolCall reports one fully-assembled tool call. It arrives as soon
	// as its arguments are complete, ahead of EventDone, so a caller can start
	// slow work without waiting for the turn to finish.
	EventToolCall EventType = "tool_call"
	// EventDone is the final event and carries the aggregated Response.
	EventDone EventType = "done"
)

// Event is one element of a streamed response.
//
// Text arrives as a block at a time: a Start, some Deltas, an End. Without
// those boundaries a consumer cannot tell "the model started a second
// paragraph" from "it kept typing", which is the difference between opening a
// new bubble and appending to the last one — and it cannot know when a block
// is finished enough to render as markdown rather than as a growing string.
type Event struct {
	Type EventType

	// Index identifies the content block a Start, Delta or End belongs to. It
	// counts from zero and rises across the response, so a consumer can key
	// its own state by it.
	Index int

	// Text is the increment on a Delta, and the whole block on an End.
	Text string
	// ToolCall is set on EventToolCall.
	ToolCall *ToolCall
	// Response is set on EventDone.
	Response *Response
}

// Delta is what a Driver emits: raw increments, with no accumulation.
//
// Keeping drivers delta-only is deliberate. Assembling the final Response —
// joining text, ordering tool calls, reconciling usage — is identical across
// protocols, so Client does it once for all of them instead of each driver
// re-implementing it slightly differently.
//
// A Delta may set several fields at once. Usage and StopReason are replacing
// rather than additive: a driver sends the best figures it has so far, and the
// last non-zero value wins.
type Delta struct {
	Text     string
	Thinking string
	// ThinkingSignature is appended, not replaced — Anthropic streams it in
	// fragments.
	ThinkingSignature string
	// ToolCall is a complete call, emitted once its arguments have finished
	// streaming.
	ToolCall *ToolCall
	// Reasoning replaces the response's reasoning items.
	Reasoning []ReasoningItem
	// Usage, when non-nil, replaces the running usage figures field by field,
	// ignoring zeros.
	Usage *Usage
	// StopReason, when non-empty, replaces the recorded stop reason.
	StopReason StopReason
	// EndBlock closes the current content block. A protocol that reports block
	// boundaries sets it; where one does not, Client infers a boundary from a
	// change of kind, which is right except for two adjacent blocks of the
	// same kind — something only a protocol that marks them can distinguish.
	EndBlock bool

	// Model, when non-empty, records the model ID the provider actually served.
	// A gateway routing "auto" to a concrete model reports it here.
	Model string
	// ID, when non-empty, is the provider's own identifier for this response.
	ID string
}

// Driver speaks one wire protocol.
//
// Implementations are the only place that knows about an HTTP API's request
// and response shapes. They translate a Request into that protocol, stream the
// result back as Deltas, and classify their SDK's errors into *Error. They do
// not accumulate state across a call, retry, or read credentials.
type Driver interface {
	// Name identifies the driver for diagnostics, e.g. "anthropic-messages".
	Name() string

	// Generate runs one inference call. The returned iterator yields Deltas
	// until the response is complete; a non-nil error terminates it and must
	// be the last element. Iteration is what drives the request, so an
	// abandoned iterator must release the underlying connection.
	//
	// Options arrives fully merged — Client has already applied its own
	// defaults and the model's — so an implementation reads it directly and
	// never has to handle a nil or an unset field it would have to guess at.
	Generate(ctx context.Context, p *Prompt, opts Options) iter.Seq2[Delta, error]

	// Models lists the models this driver's endpoint currently serves.
	// Implementations may fall back to a static catalog when the endpoint
	// offers no listing.
	Models(ctx context.Context) ([]Model, error)
}

// blockTracker turns a flat run of deltas into bounded content blocks.
//
// Only some protocols report where a block begins and ends. Rather than leave
// consumers to guess per provider, boundaries are synthesised here from a
// change of kind — text to thinking, or either to a tool call — and a driver
// whose protocol does mark them refines that with Delta.EndBlock.
type blockTracker struct {
	yield func(Event, error) bool
	open  bool
	kind  EventType
	index int
	buf   strings.Builder
	// counted rises for every block, closed ones included, so an index is
	// never reused.
	counted int
}

// write emits a delta of the given kind, opening a block first and closing any
// block of a different kind. It reports whether the consumer is still reading.
func (b *blockTracker) write(kind EventType, text string) bool {
	if b.open && b.kind != kind {
		if !b.close() {
			return false
		}
	}
	if !b.open {
		b.open, b.kind, b.index = true, kind, b.counted
		b.counted++
		b.buf.Reset()
		if !b.yield(Event{Type: startOf(kind), Index: b.index}, nil) {
			return false
		}
	}
	b.buf.WriteString(text)
	return b.yield(Event{Type: kind, Index: b.index, Text: text}, nil)
}

// close ends the open block, if any, handing back its whole text.
func (b *blockTracker) close() bool {
	if !b.open {
		return true
	}
	b.open = false
	return b.yield(Event{Type: endOf(b.kind), Index: b.index, Text: b.buf.String()}, nil)
}

// next allocates an index for a non-text block, such as a tool call.
func (b *blockTracker) next() int {
	i := b.counted
	b.counted++
	return i
}

func startOf(kind EventType) EventType {
	if kind == EventThinkingDelta {
		return EventThinkingStart
	}
	return EventTextStart
}

func endOf(kind EventType) EventType {
	if kind == EventThinkingDelta {
		return EventThinkingEnd
	}
	return EventTextEnd
}
