package agent

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Result is what running a tool produced, for the two audiences a tool has.
type Result struct {
	// Content is what the model is told. It is the only field that reaches it.
	Content ai.Content

	// Details is for your interface, and goes nowhere else: not to the model,
	// and not into a session, which drops it as belonging to a program that is
	// no longer running. Put the structured form of the answer here — the rows
	// behind a count, the paths behind a total — so that formatting it for a
	// person is not paid for on every turn thereafter.
	//
	// It is any rather than a type parameter because a type parameter would
	// have to be on Tool, and an agent's tools are a heterogeneous list: one
	// Result type for all of them is no type at all, and one per tool is not a
	// list. The producer and the consumer of a Details agree out of band, the
	// way they do with a context value.
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

// ResultText is what a tool call comes to say: the result's text, or the error
// when it failed, or a placeholder when it produced neither — several endpoints
// reject a tool result with no content at all. It is one function because the
// model and the session must be told the same thing, and two implementations
// of the same fallback had already come to disagree.
func ResultText(r Result, err error) string {
	if text := r.Text(); text != "" {
		return text
	}
	if err != nil {
		return err.Error()
	}
	return "(no output)"
}

// Tool is one thing an agent can do. Like ai.Tool it is two halves — what the
// model is told, and what answers a call — as an interface, so a tool can be
// anything with those two.
type Tool interface {
	// Schema is what the model is told: the tool's name, what it is for, and
	// the shape of its arguments.
	Schema() ai.Schema
	// Run executes one call. Returning an error is how a tool fails: the loop
	// turns it into a tool error the model can see and correct, rather than
	// failing the turn.
	Run(ctx context.Context, call ai.ToolCall) (Result, error)
}

// reporter is where a running tool's partial results go, put in its context by
// the exchange running it.
type reporter struct{}

// Report sends a partial result from inside a tool, reaching whoever is ranging
// over the exchange as ToolUpdate. It is how a tool that takes a while shows
// its work while it works.
//
// It is reached through the context rather than a parameter so that a tool with
// nothing to report pays nothing for it — no argument to declare, no interface
// to satisfy. Outside a tool, or when nobody is listening, it does nothing.
//
// What is reported is handed over. It leaves the tool's goroutine for whoever
// is ranging over the exchange, so a tool must not keep writing to anything
// inside it — the slice behind a Content block, the value in Details — after
// passing it here. Build the next report; do not edit the last one.
//
// A report is also droppable, and dropped rather than stalling a tool that has
// nobody listening. ToolEnd carries the finished result either way.
//
//	agent.Report(ctx, agent.TextResult(line))
func Report(ctx context.Context, partial Result) {
	if to, ok := ctx.Value(reporter{}).(func(Result)); ok {
		to(partial)
	}
}

// Sequential marks a tool that must not run beside others. One of them in a
// batch makes the whole batch run one at a time, because a batch is only safe
// to parallelize if every member of it is. A tool that mutates shared state
// wants this.
//
// It must be the outermost wrapper. The mark is on the value the agent holds,
// so a decorator of your own placed outside it hides it and the batch goes
// back to running in parallel — wrap the other way round:
//
//	agent.Sequential(logged{writeFile})   // marked
//	logged{agent.Sequential(writeFile)}   // not
func Sequential(t Tool) Tool { return sequential{t} }

type sequential struct{ Tool }

// isSequential reports whether a tool was marked. Deliberately a type
// assertion: a capability method would not survive a decorator that embeds the
// Tool interface either, since the interface is the whole method set a
// decorator forwards, so the honest fix is the documented rule on Sequential
// rather than machinery that looks like it lifts it.
func isSequential(t Tool) bool {
	_, ok := t.(sequential)
	return ok
}

// ToolFunc builds a tool from a Go argument type, the way ai.ToolFunc does:
// the schema the model is sent is derived from the same struct the arguments
// are decoded into, so the two cannot come to describe different things.
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
