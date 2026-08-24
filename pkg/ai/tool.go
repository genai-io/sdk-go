package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// Tool is a tool definition offered to the model.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`

	// Parameters is a JSON Schema object describing the tool's arguments,
	// typically a map[string]any. Drivers translate it into their wire shape.
	Parameters any `json:"parameters,omitempty"`

	// Run answers this tool's calls, taking the model's arguments as the JSON
	// it sent. ToolFunc fills it in from a Go type, which is what most callers
	// want; set it directly for a tool whose shape is not known until run time
	// — one loaded from configuration, or proxied from somewhere else.
	Run func(ctx context.Context, arguments string) (string, error) `json:"-"`
}

// ParameterSchema returns an independent JSON-object representation of the
// tool's parameters. A nil result means that Parameters was omitted or is not
// representable as a JSON object.
func (t Tool) ParameterSchema() map[string]any {
	return jsonSchemaObject(t.Parameters)
}

// ValidateArgs checks a tool call's arguments against the tool's own schema.
func (t Tool) ValidateArgs(input string) error {
	schema := t.ParameterSchema()
	if len(schema) == 0 {
		return nil
	}
	var value any
	trimmed := bytes.TrimSpace([]byte(input))
	if len(trimmed) == 0 {
		// An empty argument string means an empty object, which every protocol
		// sends for a no-argument call.
		value = map[string]any{}
	} else if err := json.Unmarshal(trimmed, &value); err != nil {
		return fmt.Errorf("arguments for %s are not valid JSON: %w", t.Name, err)
	}
	if err := jsonschema.Check(schema, value); err != nil {
		return fmt.Errorf("arguments for %s: %w", t.Name, err)
	}
	return nil
}

// decodeArgs writes a model's arguments over whatever the target already
// holds, so a field the model did not send keeps the value it was given.
func decodeArgs(arguments string, into any) error {
	trimmed := bytes.TrimSpace([]byte(arguments))
	if len(trimmed) == 0 {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	return dec.Decode(into)
}

// FindTool returns the tool with the given name.
func FindTool(tools []Tool, name string) (Tool, bool) {
	for _, t := range tools {
		if t.Name == name {
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
	var zero T
	t := reflect.TypeOf(zero)
	if t == nil || t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("ai: a tool's arguments must be a struct, not %v", t))
	}
	if name == "" {
		panic(fmt.Sprintf("ai: the tool taking %s has no name; "+
			"that is the string the model calls", t))
	}
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  jsonschema.ForType(t),
		Run: func(ctx context.Context, arguments string) (string, error) {
			var args T
			if err := decodeArgs(arguments, &args); err != nil {
				return "", fmt.Errorf("arguments for %s: %w", name, err)
			}
			return run(ctx, args)
		},
	}
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
	if err := tool.ValidateArgs(call.Input); err != nil {
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
		names[i] = tool.Name
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
