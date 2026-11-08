package agent

import (
	"context"
	"fmt"
	"slices"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// InferenceAdmission describes one finalized provider request, after client
// defaults, PreInfer options, history repair and validation. Request is detached:
// editing or retaining it does not change dispatch. Model identifies the selected
// client, including routing changes made by PreInfer. Attempt starts at one per
// step; neither it nor Turn is a durable globally unique execution identity.
type InferenceAdmission struct {
	Turn    int
	Attempt int
	Model   ai.Model
	Request *ai.Request
}

// WithInferenceAdmission installs an owner-supplied boundary before each provider
// dispatch (including retries). Failure is fatal, bypassing retry/recovery hooks.
// InferenceStart is a notification, not evidence of admission or dispatch. A
// successful admission cannot guarantee dispatch or exactly-once remote effects.
func WithInferenceAdmission(admit func(context.Context, InferenceAdmission) error) Option {
	return func(a *Agent) { a.admitInference = admit }
}

// ToolAdmission is a finalized, validated call after all permission gates and
// argument rewrites. Call is a detached snapshot; editing it cannot change execution.
// Turn and the model-supplied Call.ID are correlation values, not globally unique IDs.
type ToolAdmission struct {
	Turn int
	Call ai.ToolCall
}

// WithToolAdmission installs an owner-supplied execution admission boundary.
// Returning an error ends the turn, without running this tool or dispatching any
// more tools in the batch. Already dispatched parallel tools are drained, not
// rolled back. Admission runs serially on Run's goroutine, after gates and before
// dispatch, independently of ToolStart (which still precedes gates).
// Successful admission is not proof that execution occurred: cancellation or a
// process failure can intervene. The owner supplies persistence and identity policy.
func WithToolAdmission(admit func(context.Context, ToolAdmission) error) Option {
	return func(a *Agent) { a.admitTool = admit }
}

// OutputAdmission is a candidate model-visible output after post-hooks and after
// assigning its message ID when WithMessageIDs is configured. Message is detached
// from the conversation. Response,
// when non-nil, is a detached successful inference response; otherwise Message
// contains the complete tool-result batch in model order. Terminate is the batch's
// final termination vote. These values let an owner commit output and completion
// together before the SDK appends the message.
type OutputAdmission struct {
	Turn      int
	Message   ai.Message
	Response  *ai.Response
	Terminate bool
}

// WithOutputAdmission installs an owner-supplied accepted-output boundary.
// An error is fatal: the candidate is not appended, MessageAdded is not emitted,
// and it is not sent to a later inference. Streaming fragments, InferenceEnd and
// ToolEnd remain speculative notifications and may already have been delivered.
// This is not a transaction over provider/tool effects, input messages, history
// replacement, or TurnEnd. SetMessages and replay do not invoke it.
func WithOutputAdmission(admit func(context.Context, OutputAdmission) error) Option {
	return func(a *Agent) { a.admitOutput = admit }
}

// AdmissionError identifies an owner boundary failure. It is never a model-facing
// tool error and is never offered to inference retry/recovery hooks.
type AdmissionError struct {
	Boundary string
	Err      error
}

func (e *AdmissionError) Error() string {
	return fmt.Sprintf("agent: %s admission: %v", e.Boundary, e.Err)
}
func (e *AdmissionError) Unwrap() error { return e.Err }

func (a *Agent) admitToolCall(ctx context.Context, c ai.ToolCall) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if a.admitTool != nil {
		c.Signature = slices.Clone(c.Signature)
		if err := a.admitTool(ctx, ToolAdmission{Turn: a.turnNow(), Call: c}); err != nil {
			return &AdmissionError{Boundary: "tool", Err: err}
		}
	}
	return ctx.Err()
}

func (a *Agent) acceptOutput(ctx context.Context, emit func(Event), msg ai.Message, resp *ai.Response, terminate bool) (ai.Message, error) {
	if a.admitOutput != nil {
		if err := ctx.Err(); err != nil {
			return ai.Message{}, err
		}
		if msg.ID == "" && a.mintID != nil {
			msg.ID = a.mintID()
		}
		snapshot := msg
		snapshot.Content = msg.Content.Clone()
		var response *ai.Response
		if resp != nil {
			copy := *resp
			copy.Content = resp.Content.Clone()
			response = &copy
		}
		if err := a.admitOutput(ctx, OutputAdmission{Turn: a.turnNow(), Message: snapshot, Response: response, Terminate: terminate}); err != nil {
			return ai.Message{}, &AdmissionError{Boundary: "output", Err: err}
		}
		// A successful durable commit remains accepted even if cancellation races
		// with its return; do not leave the owner committed and the SDK unpublished.
	}
	a.add(emit, msg)
	return msg, nil
}
