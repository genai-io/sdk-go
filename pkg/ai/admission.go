package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"math"
	"slices"
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
// The callback gets a detached copy, not wire bytes; edits cannot change the
// request. Schema definitions and sampling parameters are JSON-normalized, with
// numbers preserved as json.Number. Remove secrets before saving it. Each
// iteration makes a new call.
// Cancellation can still stop a request after admit succeeds.
func (c *Client) StreamAdmitted(ctx context.Context, messages []Message, admit func(context.Context, *Request) error, opts ...Option) iter.Seq2[Event, error] {
	return c.stream(ctx, messages, admit, opts...)
}

// CompleteAdmitted runs one inference call with the admission behavior of
// StreamAdmitted and returns the aggregated response, discarding intermediate
// events. Nil admit behaves like Complete.
func (c *Client) CompleteAdmitted(ctx context.Context, messages []Message, admit func(context.Context, *Request) error, opts ...Option) (*Response, error) {
	return collect(c.StreamAdmitted(ctx, messages, admit, opts...))
}

func admitRequest(ctx context.Context, req *Request, admit func(context.Context, *Request) error) (*Request, error) {
	frozen, snapshot, err := admissionCopies(req)
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

// admissionCopies freezes both sides before application code runs. Native
// payloads use SDK clones, so immutable strings (including base64 images) need
// no encoding or copying. Only open-ended JSON data goes through the codec;
// this also rejects unsupported values without a reflection-based clone.
func admissionCopies(req *Request) (*Request, *Request, error) {
	if req.Temperature != nil && (math.IsNaN(*req.Temperature) || math.IsInf(*req.Temperature, 0)) {
		return nil, nil, fmt.Errorf("non-finite temperature")
	}
	for _, message := range req.Messages {
		if err := admissionContentCheck(message.Content, nil); err != nil {
			return nil, nil, err
		}
	}
	type dynamic struct {
		Tools    []any
		Schema   any
		Sampling map[string]any
	}
	values := dynamic{Sampling: req.SamplingParams}
	for _, tool := range req.Tools {
		values.Tools = append(values.Tools, tool.Definition)
	}
	if req.Schema != nil {
		values.Schema = req.Schema.Definition
	}
	var data []byte
	if len(values.Tools) != 0 || values.Schema != nil || values.Sampling != nil {
		var err error
		data, err = json.Marshal(values)
		if err != nil {
			return nil, nil, err
		}
	}
	clone := func() (*Request, error) {
		var normalized dynamic
		if data != nil {
			decoder := json.NewDecoder(bytes.NewReader(data))
			decoder.UseNumber()
			if err := decoder.Decode(&normalized); err != nil {
				return nil, err
			}
		}
		out := *req
		out.Messages = slices.Clone(req.Messages)
		for i := range out.Messages {
			out.Messages[i].Content = out.Messages[i].Content.Clone()
		}
		out.StopSequences = slices.Clone(req.StopSequences)
		if req.Temperature != nil {
			v := *req.Temperature
			out.Temperature = &v
		}
		out.Tools = slices.Clone(req.Tools)
		for i := range out.Tools {
			out.Tools[i].Definition = normalized.Tools[i]
		}
		if req.Schema != nil {
			schema := *req.Schema
			schema.Definition = normalized.Schema
			out.Schema = &schema
		}
		out.SamplingParams = normalized.Sampling
		return &out, nil
	}
	frozen, err := clone()
	if err != nil {
		return nil, nil, err
	}
	snapshot, err := clone()
	return frozen, snapshot, err
}

// Reject cycles before Content.Clone recurses, as JSON encoding did. Repeated
// aliases are fine; only a tool result already on the active path is a cycle.
func admissionContentCheck(content Content, active map[*ToolResult]bool) error {
	for _, block := range content {
		if result := block.ToolResult; result != nil {
			if active[result] {
				return fmt.Errorf("cyclic tool result content")
			}
			if active == nil {
				active = make(map[*ToolResult]bool)
			}
			active[result] = true
			if err := admissionContentCheck(result.Content, active); err != nil {
				return err
			}
			delete(active, result)
		}
	}
	return nil
}
