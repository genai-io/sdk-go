package agent

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// Result is what running a tool produced.
type Result struct {
	Content ai.Content
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

// Tool is one thing an agent can do.
type Tool interface {
	// Definition is what the model is told: name, description, argument schema.
	Definition() ai.Tool
	// Run executes one call. Returning an error is how a tool fails: the loop
	// turns it into a tool error the model can see and correct, rather than
	// failing the turn.
	Run(ctx context.Context, call ai.ToolCall) (Result, error)
}

// Sequential marks a tool that must not run beside others. One of them in a
// batch makes the whole batch run one at a time, because a batch is only safe
// to parallelize if every member of it is. A tool that mutates shared state
// wants this.
func Sequential(t Tool) Tool { return sequential{t} }

type sequential struct{ Tool }

// ToolFunc builds a tool from a Go argument type, the way ai.ToolFunc does:
// the schema the model is sent is derived from the same struct the arguments
// are decoded into, so the two cannot come to describe different things.
func ToolFunc[T any](name, description string,
	run func(ctx context.Context, args T) (Result, error)) Tool {
	return &funcTool{
		def: ai.Tool{
			Name:        name,
			Description: description,
			Parameters:  jsonschema.For[T](),
		},
		run: func(ctx context.Context, call ai.ToolCall) (Result, error) {
			var args T
			input := strings.TrimSpace(call.Input)
			if input != "" && input != "null" {
				if err := json.Unmarshal([]byte(input), &args); err != nil {
					return Result{}, err
				}
			}
			return run(ctx, args)
		},
	}
}

// FromAI lifts a plain ai.Tool, so anything already written against pkg/ai
// works here without being rewritten.
func FromAI(t ai.Tool) Tool {
	return &funcTool{
		def: t,
		run: func(ctx context.Context, call ai.ToolCall) (Result, error) {
			if t.Run == nil {
				return Result{}, &ai.Error{Kind: ai.KindInvalidRequest, Message: "agent: tool " + t.Name + " has no Run"}
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
	def ai.Tool
	run func(context.Context, ai.ToolCall) (Result, error)
}

func (t *funcTool) Definition() ai.Tool { return t.def }

func (t *funcTool) Run(ctx context.Context, call ai.ToolCall) (Result, error) {
	return t.run(ctx, call)
}
