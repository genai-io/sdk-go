package agent

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Hook is where a caller gets between the loop and the model. An agent holds
// several, and they run in the order they were added.
//
//	           in                                     out
//	PreInfer   the call, about to go                  edits it in place
//	PostInfer  the response, on a call that worked    edits it in place
//	PreTool    the call, its tool, the conversation   Decision
//	PostTool   the call, its tool, what it produced   *Result (nil keeps it)
//
// All but PreTool chain: each sees what the one before it left. PreTool is
// asked in order and the first refusal is final. Every hook runs on the loop's
// goroutine one at a time, so none needs locking of its own, and an error from
// any of them ends the exchange.
type Hook struct {
	// PreInfer edits one call in place: prune the history, narrow the toolset,
	// add a line to the prompt that is only true right now. To change the
	// agent itself, use SetMessages, SetTools, SetSystem.
	PreInfer func(ctx context.Context, inf *Inference) error

	// PostInfer edits what came back — redact, annotate, normalise. An error
	// ends the turn without a retry.
	PostInfer func(ctx context.Context, resp *ai.Response) error

	// PreTool runs after the arguments validate and before the tool does. It
	// can refuse the call, rewrite what the model sent, or vote to end the
	// turn. This is where a permission system lives.
	PreTool func(ctx context.Context, c PreToolContext) (Decision, error)

	// PostTool runs after the tool returns. A non-nil result replaces what it
	// produced; nil keeps it.
	PostTool func(ctx context.Context, c PostToolContext) (*Result, error)
}

// PreToolContext is what the gate is told about a call it may refuse.
type PreToolContext struct {
	Call     ai.ToolCall
	Tool     Tool
	Messages []ai.Message
}

// PostToolContext is what the post-hook is told about a call that finished.
type PostToolContext struct {
	Call     ai.ToolCall
	Tool     Tool
	Result   Result
	Err      error
	Messages []ai.Message
}

// Decision is the gate's answer.
type Decision struct {
	// Block refuses the call. Reason is reported to the model as the tool
	// error, so write it for the model to act on — it is the only thing the
	// model learns about why nothing happened.
	Block  bool
	Reason string

	// Arguments, when non-empty, replaces what the model sent.
	Arguments string

	// Terminate votes to end the turn after this batch — see Result.Terminate.
	// A gate's vote stands even when it blocks the call.
	Terminate bool
}

// hookSet returns the hooks to run, in order. No copy: hooks are only
// appended, so a concurrent append either writes past this length or
// reallocates, and neither disturbs a reader.
func (a *Agent) hookSet() []Hook {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hooks
}

// preInfer chains the hooks over the call, each seeing the last one's edits.
func (a *Agent) preInfer(ctx context.Context, inf *Inference) error {
	for _, h := range a.hookSet() {
		if h.PreInfer == nil {
			continue
		}
		if err := h.PreInfer(ctx, inf); err != nil {
			return err
		}
	}
	return nil
}

// postInfer chains the hooks over a response.
func (a *Agent) postInfer(ctx context.Context, resp *ai.Response) error {
	for _, h := range a.hookSet() {
		if h.PostInfer == nil {
			continue
		}
		if err := h.PostInfer(ctx, resp); err != nil {
			return err
		}
	}
	return nil
}

// Inference is one model call as the agent is about to make it: what PreInfer
// is handed, and what MessageStart and MessageEnd carry.
//
// System, Messages and Tools are what the agent contributes. Everything else —
// a forced tool for this step, a schema for this answer, a cap on these tokens
// — is reached by appending to Options, and not by a field, because a
// half-filled ai.Request cannot say which of its fields were meant: "left
// alone" and "set to zero" are the same bytes. An appended option is there or
// it is not.
//
//	PreInfer: func(_ context.Context, inf *agent.Inference) error {
//	    if len(inf.Messages) > 200 {
//	        inf.Messages = inf.Messages[len(inf.Messages)-200:]
//	    }
//	    inf.Options = append(inf.Options, ai.WithForceTool("search"))
//	    return nil
//	},
type Inference struct {
	System   string
	Messages []ai.Message
	// Tools is what the model is offered. Empty means none and says so: the
	// agent's toolset is the toolset, not a suggestion the client may override.
	Tools []ai.Tool
	// Options are layered onto this call last, over the client's defaults.
	Options []ai.Option
}

// options renders the call as the layer pkg/ai composes from.
func (inf *Inference) options() []ai.Option {
	opts := make([]ai.Option, 0, len(inf.Options)+2)
	opts = append(opts, ai.WithSystem(inf.System), ai.WithTools(inf.Tools...))
	return append(opts, inf.Options...)
}
