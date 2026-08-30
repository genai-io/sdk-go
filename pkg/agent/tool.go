package agent

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Result is what running a tool produced, for the two audiences a tool has.
type Result struct {
	// Content is what the model is told. It is the only field that reaches it.
	Content ai.Content

	// Details is for your interface and goes nowhere else — not to the model,
	// not into a session. Put the structured form of the answer here: the rows
	// behind a count, the paths behind a total.
	//
	// It is any and not a type parameter because that parameter would land on
	// Tool, and an agent's tools are a heterogeneous list. Producer and
	// consumer agree out of band, the way they do with a context value.
	Details any

	// Terminate ends the turn after this batch, with StopTerminated. It is a
	// vote: every call in the batch must ask, so one tool cannot cut short a
	// turn whose others are still working. Set it on a failing result too, if
	// the failure is the reason to stop.
	Terminate bool
}

// TextResult returns a result carrying one line of text for the model.
func TextResult(text string) Result { return Result{Content: ai.TextContent(text)} }

// Text returns the result's text in block order.
func (r Result) Text() string { return r.Content.Text() }

// ResultText is what a tool call comes to say: its text, or the error when it
// failed, or a placeholder when it produced neither, since several endpoints
// reject a tool result with no content. One function, because the model and
// the session must be told the same thing.
func ResultText(r Result, err error) string {
	if text := r.Text(); text != "" {
		return text
	}
	if err != nil {
		return err.Error()
	}
	return "(no output)"
}

// Tool is one thing an agent can do: what the model is told, and what answers
// a call. The same two halves as ai.Tool, as an interface.
type Tool interface {
	// Schema is the tool's name, what it is for, and the shape of its arguments.
	Schema() ai.Schema
	// Run executes one call. An error is how a tool fails: the loop turns it
	// into a tool error the model can see and correct, not a failed turn.
	Run(ctx context.Context, call ai.ToolCall) (Result, error)
}

// reporter carries a running tool's partial results, put in its context by the
// exchange running it.
type reporter struct{}

// Report sends a partial result from inside a tool, arriving as ToolUpdate:
// how a tool that takes a while shows its work. It goes through the context so
// that a tool with nothing to report declares nothing, and it does nothing
// outside a tool or when nobody is listening.
//
// What is reported is handed over: it leaves the tool's goroutine, so do not
// keep writing to anything inside it. Build the next report, do not edit the
// last one. A report is also droppable rather than stalling a tool nobody is
// listening to — ToolEnd carries the finished result either way.
//
//	agent.Report(ctx, agent.TextResult(line))
func Report(ctx context.Context, partial Result) {
	if to, ok := ctx.Value(reporter{}).(func(Result)); ok {
		to(partial)
	}
}

// Sequential marks a tool that must not run beside others: one of them makes
// its whole batch run one at a time, because a batch is only safe to
// parallelize if every member is. A tool that mutates shared state wants this.
//
// It must be the outermost wrapper — the mark is on the value the agent holds,
// so your own decorator placed outside it hides it:
//
//	agent.Sequential(logged{writeFile})   // marked
//	logged{agent.Sequential(writeFile)}   // not
func Sequential(t Tool) Tool { return sequential{t} }

type sequential struct{ Tool }

// isSequential reports whether a tool was marked. A type assertion on purpose:
// a capability method would not survive an embedding decorator either, so the
// rule documented on Sequential is the honest fix, not machinery that looks
// like it lifts the rule.
func isSequential(t Tool) bool {
	_, ok := t.(sequential)
	return ok
}

// ToolFunc builds a tool from a Go argument type, the way ai.ToolFunc does:
// the schema and the decode target are the same struct, so they cannot come to
// describe different things.
func ToolFunc[T any](name, description string,
	run func(ctx context.Context, args T) (Result, error)) Tool {
	return &funcTool{
		schema: ai.ToolSchema[T](name, description),
		run: func(ctx context.Context, call ai.ToolCall) (Result, error) {
			var args T
			if err := call.UnmarshalArgs(&args); err != nil {
				return Result{}, err
			}
			return run(ctx, args)
		},
	}
}

// FromAI lifts a plain ai.Tool, so anything already written against pkg/ai
// works here without being rewritten.
func FromAI(t ai.Tool) Tool {
	return &funcTool{
		schema: t.Schema,
		run: func(ctx context.Context, call ai.ToolCall) (Result, error) {
			if t.Run == nil {
				return Result{}, &ai.Error{Kind: ai.KindInvalidRequest, Message: "agent: tool " + t.Schema.Name + " has no Run"}
			}
			out, err := t.Run(ctx, call.Input)
			if err != nil {
				return Result{}, err
			}
			return TextResult(out), nil
		},
	}
}

type funcTool struct {
	schema ai.Schema
	run    func(context.Context, ai.ToolCall) (Result, error)
}

func (t *funcTool) Schema() ai.Schema { return t.schema }

func (t *funcTool) Run(ctx context.Context, call ai.ToolCall) (Result, error) {
	return t.run(ctx, call)
}
