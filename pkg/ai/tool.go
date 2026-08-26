package ai

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// Tool is one thing the model can ask for. It is two halves: the Schema is
// everything the model is told, and Run is what answers a call.
type Tool struct {
	// Schema names the tool, says what it is for, and describes its arguments.
	// It is the same Schema an answer's shape uses, because it is the same
	// problem — a JSON object the model has to get right — so both are
	// described and checked by the same code.
	Schema Schema `json:"schema"`

	// Run answers this tool's calls, taking the model's arguments as the JSON
	// it sent. ToolFunc fills it in from a Go type, which is what most callers
	// want; set it directly for a tool whose shape is not known until run time
	// — one loaded from configuration, or proxied from somewhere else.
	Run func(ctx context.Context, arguments string) (string, error) `json:"-"`
}

// FindTool returns the tool with the given name.
func FindTool(tools []Tool, name string) (Tool, bool) {
	for _, t := range tools {
		if t.Schema.Name == name {
			return t, true
		}
	}
	return Tool{}, false
}

// ─── Running the tools a model asked for ───

// ToolFunc builds a tool from a name, a description and one function.
//
//	type SearchArgs struct {
//		Query string `json:"query" description:"what to look for, in plain words"`
//		Limit int    `json:"limit,omitempty" description:"how many passages" maximum:"10"`
//	}
//
//	search := ai.ToolFunc("search", "Search the docs and return matching passages.",
//		func(ctx context.Context, a SearchArgs) (string, error) {
//			return docs.Search(ctx, a.Query, a.Limit)
//		})
//
//	client.Run(ctx, messages, []ai.Tool{search, fetch})
func ToolFunc[T any](name, description string, run func(ctx context.Context, arguments T) (string, error)) Tool {
	return Tool{
		Schema: ToolSchema[T](name, description),
		Run: func(ctx context.Context, arguments string) (string, error) {
			var args T
			if err := (ToolCall{Name: name, Input: arguments}).UnmarshalArgs(&args); err != nil {
				return "", err
			}
			return run(ctx, args)
		},
	}
}

// ToolSchema derives a tool's schema from its Go argument type. ToolFunc uses
// it; call it directly when you are filling in Tool.Run yourself.
func ToolSchema[T any](name, description string) Schema {
	t := reflect.TypeFor[T]()
	if t == nil || t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("ai: a tool's arguments must be a struct, not %v", t))
	}
	if name == "" {
		panic(fmt.Sprintf("ai: the tool taking %s has no name; "+
			"that is the string the model calls", t))
	}
	return Schema{Name: name, Description: description, Definition: jsonschema.ForType(t)}
}

// RunTools answers every call in a turn, in order, and returns the results as
// one user turn's worth of answers:
//
//	messages = append(messages, response.Message())
//	messages = append(messages, ai.ToolResultsMessage(
//		ai.RunTools(ctx, tools, response.ToolCalls())...))
func RunTools(ctx context.Context, tools []Tool, calls []ToolCall) []ToolResult {
	out := make([]ToolResult, 0, len(calls))
	for _, call := range calls {
		out = append(out, runOne(ctx, tools, call))
	}
	return out
}

func runOne(ctx context.Context, tools []Tool, call ToolCall) ToolResult {
	result := ToolResult{ToolCallID: call.ID, ToolName: call.Name}
	failed := func(format string, args ...any) ToolResult {
		result.Content, result.IsError = fmt.Sprintf(format, args...), true
		return result
	}

	tool, found := FindTool(tools, call.Name)
	if !found {
		// Naming what does exist matters: a model that invented a tool can
		// pick a real one, where "unknown tool" leaves it guessing again.
		return failed("no tool named %q; the tools available are %s", call.Name, toolNames(tools))
	}
	if tool.Run == nil {
		return failed("tool %q was offered without anything to run it", call.Name)
	}
	if err := tool.Schema.Validate(call.Input); err != nil {
		return failed("%v", err)
	}

	output, err := tool.Run(ctx, call.Input)
	if err != nil {
		return failed("%v", err)
	}
	result.Content = output
	return result
}

func toolNames(tools []Tool) string {
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Schema.Name
	}
	switch len(names) {
	case 0:
		return "none"
	case 1:
		return names[0]
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}

// maxToolTurns bounds Run against a model that keeps calling tools. It is
// deliberately generous — a real conversation ends in a handful of turns — and
// hitting it means something is wrong, not that the budget was too small.
const maxToolTurns = 32

// Run holds a conversation to the end, answering tool calls as they arrive.
//
//	response, history, err := client.Run(ctx, messages,
//		[]ai.Tool{search, fetch})
//
//	fmt.Println(response.Text())
//
//	response, history, err = client.Run(ctx, append(history, ai.UserMessage(next)), tools)
func (c *Client) Run(ctx context.Context, messages []Message, tools []Tool, opts ...Option) (*Response, []Message, error) {
	opts = append([]Option{WithTools(tools...)}, opts...)

	history := slices.Clone(messages)
	for turn := 1; turn <= maxToolTurns; turn++ {
		response, err := c.Complete(ctx, history, opts...)
		if err != nil {
			return response, history, err
		}
		calls := response.ToolCalls()
		if len(calls) == 0 {
			return response, append(history, response.Message()), nil
		}
		history = append(history,
			response.Message(),
			ToolResultsMessage(RunTools(ctx, tools, calls)...))
	}
	return nil, history, &Error{Kind: KindInvalidRequest, Message: fmt.Sprintf(
		"ai: the model was still calling tools after %d turns; "+
			"write the loop yourself if a conversation should run longer", maxToolTurns)}
}
