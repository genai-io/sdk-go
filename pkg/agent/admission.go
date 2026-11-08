package agent

import (
	"context"
	"fmt"
	"slices"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// InferenceAdmission contains the selected model and prepared request, after
// options, hooks and history repair. Request is a copy; edits do not affect the
// call. Attempt restarts at one each step; Turn/Attempt are not durable IDs.
type InferenceAdmission struct {
	Turn    int
	Attempt int
	Model   ai.Model
	Request *ai.Request
}

// WithInferenceAdmission calls admit before each model call, including retries.
// An error stops the turn without retry or recovery. InferenceStart is only a
// notification; it does not mean the request was sent.
func WithInferenceAdmission(admit func(context.Context, InferenceAdmission) error) Option {
	return func(a *Agent) { a.admitInference = admit }
}

// ToolAdmission holds a copy of the checked tool call, after permissions and
// argument rewrites. Edits do not affect execution; Call.ID is not a global ID.
type ToolAdmission struct {
	Turn int
	Call ai.ToolCall
}

// WithToolAdmission calls admit before running each allowed tool. An error stops
// this and later calls in the batch; already-started tools finish, not roll back.
// Callbacks run one at a time on Run's goroutine. ToolStart still precedes
// permission checks; neither it nor a successful callback proves the tool ran.
func WithToolAdmission(admit func(context.Context, ToolAdmission) error) Option {
	return func(a *Agent) { a.admitTool = admit }
}

// OutputAdmission holds copies of output after hooks, before it joins history.
// WithMessageIDs assigns Message.ID first. Response accompanies model output;
// otherwise Message contains the tool results in call order, and Terminate says
// whether the batch ends the turn.
type OutputAdmission struct {
	Turn      int
	Message   ai.Message
	Response  *ai.Response
	Terminate bool
}

// WithOutputAdmission calls admit before adding output to history. An error stops
// the turn: no MessageAdded, and later model calls will not see that output.
// Stream fragments and end notifications may already have reached the UI.
// It does not cover inputs, history replacement or replay, or undo executed work.
func WithOutputAdmission(admit func(context.Context, OutputAdmission) error) Option {
	return func(a *Agent) { a.admitOutput = admit }
}

// AdmissionError means an admission callback failed. It stops the turn instead
// of becoming a tool result or triggering inference retry/recovery.
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
		// Once saved, add it to history even if cancellation just arrived.
	}
	a.add(emit, msg)
	return msg, nil
}
