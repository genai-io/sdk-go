package agent

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Hook is where a caller gets between the loop and the model.
//
// PreInfer, PostInfer, PostTool and PreStep chain: each sees what the one
// before it left. The two that answer a question rather than edit a value are
// asked in order and the first answer is taken — PreTool's first refusal is
// final, and so is OnInferError's first Retry. All run on the loop's
// goroutine, one at a time, so none needs locking of its own.
//
// An error from any of them ends the exchange with StopError: the hook could
// not do its job. A Decision that blocks is the other answer — a refusal the
// model is told about as a tool error and may work around.
//
// A hook that panics is not recovered, unlike a tool: it runs on the goroutine
// ranging over Run, and the exchange unwinds with it, reporting no outcome.
type Hook struct {
	// PreInfer edits one call in place: prune the history, narrow the toolset,
	// add a line to the prompt that is only true right now. To change the
	// agent itself, use SetMessages, SetTools, SetSystem.
	PreInfer func(ctx context.Context, inf *Inference) error

	// PostInfer edits what came back. An error ends the turn without a retry.
	PostInfer func(ctx context.Context, resp *ai.Response) error

	// PreTool runs after the arguments validate and before the tool does.
	// Where a permission system lives, and where it refuses with a Decision.
	PreTool func(ctx context.Context, c PreToolContext) (Decision, error)

	// PostTool runs after the tool returns; a non-nil result replaces its own.
	// The batch still finishes, and then the exchange ends.
	PostTool func(ctx context.Context, c PostToolContext) (*Result, error)

	// PreStep changes the conversation itself, at the step boundary where the
	// loop is free to. A returned slice replaces it and is announced there and
	// then, as MessagesReplaced; nil leaves it alone.
	//
	// This is where compaction lives: the loop measures and applies the
	// answer, and what a shorter conversation should be is yours.
	//
	//	PreStep: func(ctx context.Context, c agent.PreStepContext) ([]ai.Message, error) {
	//	    if c.Tokens < c.Client.Model().ContextWindow*8/10 {
	//	        return nil, nil
	//	    }
	//	    return summarise(ctx, c.Messages)
	//	},
	//
	// PreInfer is the other half of the pair and edits one call, leaving the
	// conversation alone. Reach for this one when the change is meant to last.
	PreStep func(ctx context.Context, c PreStepContext) ([]ai.Message, error)

	// OnInferError is asked what to do about a call that failed, once the loop
	// is out of its own answers: the error was not retryable, or WithRetry's
	// budget is spent. Returning nil agrees to give up, and the exchange ends
	// with StopError carrying the failure.
	//
	// This is where a conversation the provider called too long is shortened
	// and sent again — a failure the caller can fix and the loop cannot, since
	// replaying the same oversized prompt only fails the same way:
	//
	//	OnInferError: func(ctx context.Context, c agent.InferErrorContext) (*agent.Retry, error) {
	//	    if !ai.IsContextExceeded(c.Err) || c.Attempt > 2 {
	//	        return nil, nil
	//	    }
	//	    agent.Compacting(ctx)
	//	    short, err := summarise(ctx, c.Messages)
	//	    if err != nil {
	//	        return nil, nil // nothing left to shorten; give up as the loop would
	//	    }
	//	    return &agent.Retry{Messages: short}, nil
	//	},
	//
	// Where the next attempt goes is not settled here but in PreInfer, which
	// is handed Inference.LastErr: one place decides routing and this one
	// decides whether there is a call left to route.
	//
	// It is not asked about a turn that was cancelled, which is a caller
	// stopping rather than a failure to answer, nor about an error PostInfer
	// returned — an objection to an answer that arrived does not earn another
	// go at getting one.
	OnInferError func(ctx context.Context, c InferErrorContext) (*Retry, error)
}

// InferErrorContext is what OnInferError is told about the call that failed.
type InferErrorContext struct {
	// Inference is that call as it went out, PreInfer's edits included. Its
	// Messages are what it carried, which is not necessarily the conversation:
	// a PreInfer that prunes sends less than the agent holds.
	Inference *Inference

	// Client is where the call actually went — the agent's own, resolved in
	// when the Inference named none. Model().ContextWindow is what a prompt
	// that was too long was too long for.
	Client *ai.Client

	// Err is why it failed. ai.IsContextExceeded, ai.IsAuth and ai.IsRetryable
	// are how one kind is told from another.
	Err error

	// Attempt is which call of this step this was, from one. It rises across
	// recoveries as well as retries, so it is the number a caller counts
	// against its own budget — see Retry.
	Attempt int

	// Messages is the conversation the agent holds, which is what a Retry
	// replaces.
	Messages []ai.Message
}

// Retry is OnInferError asking for another attempt at the same step.
//
// It does not spend WithRetry's budget, and a recovered attempt starts that
// budget over: WithRetry is the loop replaying a call it knows how to replay,
// while this is a call that has been changed — a shorter conversation, another
// endpoint — and so is not a replay of the one that failed.
//
// Nothing here bounds how many times a hook may ask for one. The budget is the
// caller's, as the policy is: only the caller knows what its recovery costs and
// whether it can still make progress. A summariser with one message left to
// summarise cannot, and says so by returning nil.
type Retry struct {
	// Messages, when non-nil, replaces the conversation before the next
	// attempt and is announced as MessagesReplaced there and then. Nil sends
	// what the agent already holds, which is what a caller that only wanted
	// the call routed elsewhere means.
	Messages []ai.Message
}

// PreStepContext is the conversation as it stands at a step boundary, and what
// sending it would cost.
type PreStepContext struct {
	// Messages is what the next call will carry.
	Messages []ai.Message

	// Tokens estimates the whole prompt — the conversation, the system prompt
	// and the tool schemas, which are easy to forget and can outweigh it.
	// Measured at every boundary rather than remembered, so a conversation
	// that was just replaced does not still read as full. Client.CountTokens
	// is the provider's own count, for a caller who will pay a round trip.
	Tokens int

	// Client is the handle this step will call. Model().ContextWindow is what
	// Tokens is measured against, and is zero for a model that states none.
	Client *ai.Client
}

// PreToolContext is what the gate is told about a call it may refuse. "Context"
// is the English word here, not context.Context: naming the type for the moment
// rather than for its subject would lose what it is.
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

// compactionKey marks a context under which Compacting announces — the run of
// a hook the loop will take a replaced conversation from — and carries the span
// that opens. It is named for that scope rather than for either hook, because
// there are two of them; a context without it is neither.
type compactionKey struct{}

// Compacting says the work a PreStep or OnInferError hook is about to do will
// take a while: it puts CompactionStart on the stream, and has the loop close
// it with CompactionEnd however the hook returns. Call it once the decision to
// shorten is made, before the model call that does it:
//
//	if c.Tokens < budget {
//	    return nil, nil
//	}
//	agent.Compacting(ctx)
//	return summarise(ctx, c.Messages)
//
// It goes through the context, as Report does, so that a hook with nothing to
// announce declares nothing and the summariser it calls can announce for it.
// Outside one of those two hooks it does nothing.
//
// A hook that shortens the conversation without a model call should not call
// it: the span exists for the wait, and one that opens and closes in the same
// instant is noise on every step.
func Compacting(ctx context.Context) {
	if open, ok := ctx.Value(compactionKey{}).(func()); ok {
		open()
	}
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
	// Client is where this call goes: the agent's own, unless a hook points it
	// elsewhere — a cheaper model for a summarising step, a second endpoint
	// after the first ran out of quota. Nil is the agent's own. It is rebuilt
	// for every attempt, so a retry can be sent where the attempt before it
	// was not; SetClient moves every later call instead of this one.
	Client *ai.Client

	// LastErr is what ended the attempt before this one, nil on the first —
	// what a hook routing away from an endpoint that just failed needs to
	// know. It is told, not asked: writing it changes nothing.
	//
	//	PreInfer: func(_ context.Context, inf *agent.Inference) error {
	//	    if ai.IsAuth(inf.LastErr) {
	//	        inf.Client = fallback
	//	    }
	//	    return nil
	//	},
	LastErr error

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
