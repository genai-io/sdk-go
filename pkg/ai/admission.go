package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
)

// AdmissionError means dispatch was refused, not that the provider failed.
// Callers must not automatically retry it as a provider/network error, even
// when its cause is retryable. A successful admission does not prove dispatch
// happened: cancellation or process death can still intervene.
type AdmissionError struct{ Err error }

func (e *AdmissionError) Error() string { return fmt.Sprintf("ai: inference admission: %v", e.Err) }
func (e *AdmissionError) Unwrap() error { return e.Err }

// StreamAdmitted is Stream with a synchronous boundary after request preparation
// and validation, before driver dispatch. admit receives a detached, JSON-normalized
// snapshot of the SDK request (not provider wire bytes). Editing it cannot change
// the dispatched request. It may contain sensitive input; persistence/redaction
// policy belongs to the caller. A nil callback is equivalent to Stream.
//
// Admission errors stop dispatch and are wrapped in AdmissionError. The callback
// runs once per consumed iterator; replaying an iterator is a new invocation.
// Streaming events remain speculative, not acceptance of durable output.
func (c *Client) StreamAdmitted(ctx context.Context, messages []Message, admit func(context.Context, *Request) error, opts ...Option) iter.Seq2[Event, error] {
	return c.stream(ctx, messages, admit, opts...)
}

func admitRequest(ctx context.Context, req *Request, admit func(context.Context, *Request) error) (*Request, error) {
	// Freeze both sides before invoking application code. Options may have left
	// aliases to caller-owned schemas, messages or sampling parameters.
	data, err := json.Marshal(req)
	if err != nil {
		return nil, &AdmissionError{Err: fmt.Errorf("snapshot request: %w", err)}
	}
	decode := func() (*Request, error) {
		var out Request
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber()
		if err := decoder.Decode(&out); err != nil {
			return nil, err
		}
		// ToolChoice deliberately has no exported fields or JSON representation.
		out.ToolChoice = req.ToolChoice
		return &out, nil
	}
	frozen, err := decode()
	if err != nil {
		return nil, &AdmissionError{Err: fmt.Errorf("snapshot request: %w", err)}
	}
	snapshot, err := decode()
	if err != nil {
		return nil, &AdmissionError{Err: fmt.Errorf("snapshot request: %w", err)}
	}
	if err := ctx.Err(); err != nil {
		return nil, &AdmissionError{Err: err}
	}
	if err := admit(ctx, snapshot); err != nil {
		return nil, &AdmissionError{Err: err}
	}
	return frozen, nil
}
