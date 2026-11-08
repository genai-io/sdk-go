package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
)

// AdmissionError means the callback stopped the request before it was sent.
// It is not a provider failure and must not trigger a provider retry.
type AdmissionError struct{ Err error }

func (e *AdmissionError) Error() string { return fmt.Sprintf("ai: inference admission: %v", e.Err) }
func (e *AdmissionError) Unwrap() error { return e.Err }

// StreamAdmitted calls admit with the prepared request before calling the model.
// For example, admit can save a start record; an error stops the call and returns
// AdmissionError. Nil admit behaves like Stream.
//
// The callback gets a JSON-normalized copy, not wire bytes; edits cannot change
// the request. Remove secrets before saving it. Each iteration makes a new call.
// Cancellation can still stop a request after admit succeeds.
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
