package agent

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Hook is where a caller gets between the loop and the model.
//
// All but PreTool chain: each sees what the one before it left. PreTool is
// asked in order and the first refusal is final. All run on the loop's
// goroutine, one at a time, so none needs locking of its own, and an error
// from any ends the exchange.
type Hook struct {
	// PreInfer edits one call in place: prune the history, narrow the toolset,
	// add a line to the prompt that is only true right now. To change the
	// agent itself, use SetMessages, SetTools, SetSystem.
	PreInfer func(ctx context.Context, inf *Inference) error

	// PostInfer edits what came back. An error ends the turn without a retry.
	PostInfer func(ctx context.Context, resp *ai.Response) error

	// PreTool runs after the arguments validate and before the tool does.
	// Where a permission system lives.
	PreTool func(ctx context.Context, c PreToolContext) (Decision, error)

	// PostTool runs after the tool returns; a non-nil result replaces its own.
	PostTool func(ctx context.Context, c PostToolContext) (*Result, error)
}

// PreToolContext is what the gate is told about a call it may refuse.
//
// "Context" here is the English word, not context.Context, and the two do sit
// in one signature. Every alternative tried was worse: PreToolCall makes the
// field below PreToolCall.Call, and naming it for the moment rather than the
// subject loses what it is. A name that reads oddly beside one other name
// beats a name that stutters against its own field.
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
	// Block refuses the call. Reason reaches the model as the tool error and
	// is all it learns about why nothing happened, so write it for the model.
	Block  bool
	Reason string

	// Arguments, when non-empty, replaces what the model sent.
	Arguments string

	// Terminate votes to end the turn after this batch, blocked or not — see
	// Result.Terminate.
	Terminate bool
}

// hookSet returns the hooks in order. No copy: hooks are only appended, so a
// concurrent append writes past this length or reallocates, and neither
// disturbs a reader.
func (a *Agent) hookSet() []Hook {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hooks
}

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
// Anything not named below is reached by appending to Options, rather than by
// a field, because a half-filled request cannot say which of its fields were
// meant — "left alone" and "set to zero" are the same bytes.
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
	// Tools is what the model is offered. Empty means none and says so,
	// overriding whatever the client was built with.
	Tools []ai.Tool
	// Options are layered on last, over the client's defaults.
	Options []ai.Option
}

func (inf *Inference) options() []ai.Option {
	opts := make([]ai.Option, 0, len(inf.Options)+2)
	opts = append(opts, ai.WithSystem(inf.System), ai.WithTools(inf.Tools...))
	return append(opts, inf.Options...)
}
