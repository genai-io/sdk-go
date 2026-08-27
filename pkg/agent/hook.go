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
	// PreInfer runs before every model call, on the call this agent is about
	// to make. Edit it in place, for that one call: prune the history, narrow
	// the toolset for one question, add a line to the prompt that is only true
	// right now. To change the agent itself, use SetMessages, SetTools,
	// SetSystem.
	PreInfer func(ctx context.Context, inf *Inference) error

	// PostInfer runs after every model call that succeeded, on what came back.
	// Edit the response in place — redact, annotate, normalise. An error ends
	// the turn without a retry.
	PostInfer func(ctx context.Context, resp *ai.Response) error

	// PreTool runs after the arguments validate and before the tool runs.
	// It can refuse the call, rewrite what the model sent, or ask the loop to
	// stop after this batch. This is where a permission system lives.
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
// appended and their elements are function values nobody edits, so an append
// either writes past this length or reallocates — neither disturbs a reader.
func (a *Agent) hookSet() []Hook {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hooks
}

// preInfer runs the hooks on the call, chaining them: each sees what the one
// before it edited.
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

// postInfer runs the hooks on a response, chaining them: each sees what the
// one before it edited.
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

// Inference is one model call, as the agent is about to make it. It is what
// PreInfer is handed, and what MessageStart and MessageEnd carry.
//
// The three fields are what an agent owns and a hook may rewrite for one call.
// Everything else a model call can carry — a forced tool for this step, a
// schema for this answer, a cap on these tokens, a protocol's own setting —
// is reached by appending to Options.
//
// That split is not arbitrary. An ai.Request handed over half-filled cannot
// say which fields were meant: for every value type on it, "left alone" and
// "deliberately set to zero" are the same bytes, which is the ambiguity
// ai.Request.Temperature is a pointer to avoid. An appended option has no such
// problem — it is there or it is not — and it is how pkg/ai composes a call in
// the first place, in layers, the last of which wins.
//
//	PreInfer: func(_ context.Context, inf *agent.Inference) error {
//	    if len(inf.Messages) > 200 {
//	        inf.Messages = inf.Messages[len(inf.Messages)-200:]
//	    }
//	    inf.Options = append(inf.Options, ai.WithForceTool("search"))
//	    return nil
//	},
type Inference struct {
	// System is the prompt this call goes out with. The agent's, unless a hook
	// changed it.
	System string
	// Messages is the conversation as this call will see it.
	Messages []ai.Message
	// Tools is what the model is offered. Empty means none, and says so — the
	// agent's toolset is the toolset, not a suggestion the client may override.
	Tools []ai.Tool

	// Options are layered onto this call last, over the client's own defaults.
	// Anything ai.Option can express belongs here.
	Options []ai.Option
}

// options renders the call as the layer pkg/ai composes from. System and Tools
// go unconditionally: an agent with no tools is an agent that offers none, and
// an empty ai.WithTools is documented to say exactly that.
func (inf *Inference) options() []ai.Option {
	opts := make([]ai.Option, 0, len(inf.Options)+2)
	opts = append(opts, ai.WithSystem(inf.System), ai.WithTools(inf.Tools...))
	return append(opts, inf.Options...)
}
