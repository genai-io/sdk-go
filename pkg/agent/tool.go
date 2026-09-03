package agent

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Result is what running a tool produced, for the two audiences a tool has.
type Result struct {
	// Content is the only field the model is told.
	Content ai.Content

	// Details is for your interface and goes nowhere else — not to the model,
	// not into a session. The structured form of the answer: the rows behind a
	// count, the paths behind a total.
	//
	// Not a type parameter, because that would land on Tool and an agent's
	// tools are a heterogeneous list. Producer and consumer agree out of band.
	Details any

	// Terminate votes to end the turn after this batch, with StopTerminated.
	// Every call in the batch must ask, so one tool cannot cut short a turn
	// whose others are still working.
	Terminate bool
}

// TextResult is the answer most tools have: some text, for the model, and
// nothing for the interface.
func TextResult(text string) Result { return Result{Content: ai.TextContent(text)} }

// Text is the part of the result the model is told, as a string.
func (r Result) Text() string { return r.Content.Text() }

// ResultContent is what a tool call comes to say: what it produced, the error
// when it failed, or a placeholder when it produced neither, since several
// endpoints reject an empty tool result. This is what the model is told.
func ResultContent(r Result, err error) ai.Content {
	switch {
	case r.Content.Text() != "" || r.Content.HasImages():
		return r.Content
	case err != nil:
		return ai.TextContent(err.Error())
	}
	return ai.TextContent("(no output)")
}

// ResultText is the same answer as text, for a log or a session record. A
// picture is not text and says so rather than reading as nothing, which is
// what a record of a tool that returned only an image would otherwise be.
func ResultText(r Result, err error) string {
	content := ResultContent(r, err)
	if text := content.Text(); text != "" {
		return text
	}
	if content.HasImages() {
		return "(image)"
	}
	return "(no output)"
}

// Tool is one thing an agent can do: what the model is told, and what answers
// a call. The same two halves as ai.Tool, as an interface.
type Tool interface {
	Schema() ai.Schema
	// Run executes one call. An error is how a tool fails: the loop turns it
	// into a tool error the model can correct, not a failed turn.
	Run(ctx context.Context, call ai.ToolCall) (Result, error)
}

// reporter is the context key the exchange puts a tool's Report channel under.
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
// rule on Sequential is the honest fix, not machinery that looks like it lifts
// the rule.
func isSequential(t Tool) bool {
	_, ok := t.(sequential)
	return ok
}

// ToolFunc builds a tool from a Go argument type: the schema and the decode
// target are the same struct, so they cannot describe different things.
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

// FromAI lifts a plain ai.Tool, so anything written against pkg/ai runs here
// unchanged.
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
