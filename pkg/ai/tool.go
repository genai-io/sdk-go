package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

	// run answers this tool's calls, and is set by Handle. It is unexported
	// because it is behaviour rather than definition: a Tool decoded from JSON
	// carries what the model is told and nothing that could be executed.
	run func(ctx context.Context, arguments string) (string, error)
}

// ToolFor builds a tool whose parameters are derived from a Go type, so the
// schema the model sees and the struct the arguments decode into cannot drift
// apart.
//
// It defines the tool without saying what runs it, which leaves you to match
// call.Name back to the right argument type yourself. Handle does both at once
// and is what most callers want.
func ToolFor[T any](name, description string) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  jsonschema.For[T](),
	}
}

// ParameterSchema returns an independent JSON-object representation of the
// tool's parameters. A nil result means that Parameters was omitted or is not
// representable as a JSON object.
//
// It is the one way to read a tool's schema. Drivers, validation and argument
// checking all go through it, so a schema that one of them accepts cannot be
// silently dropped by another — and the copy means none of them can corrupt
// the caller's definition.
func (t Tool) ParameterSchema() map[string]any {
	return jsonSchemaObject(t.Parameters)
}

// ValidateArgs checks a tool call's arguments against the tool's own schema.
//
// A model produces those arguments, so they are model output and wrong
// sometimes: a missing required field, a string where a number belongs, an
// invented property, a value outside its enum. Running the tool anyway turns a
// mistake the model could have corrected into whatever the tool does with
// nonsense — a deletion with an empty path, a query with a null filter.
// Checking first turns it back into a tool error the model sees and retries.
//
// A tool that declares no schema is not checked, because there is nothing to
// check against.
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

// UnmarshalArgs decodes a tool call's arguments into T.
//
// Unknown fields are rejected rather than dropped: a model inventing an
// argument is a mistake worth surfacing, and silently ignoring it hides both
// the model's error and, sometimes, the caller's own tag typo.
func UnmarshalArgs[T any](call ToolCall) (T, error) {
	var out T
	trimmed := bytes.TrimSpace([]byte(call.Input))
	if len(trimmed) == 0 {
		return out, nil
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&out); err != nil {
		return out, fmt.Errorf("arguments for %s: %w", call.Name, err)
	}
	return out, nil
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

// ToolFor derives a tool's schema from a Go type, which keeps the shape the
// model is told about from drifting away from the shape you parse. It leaves
// one join unguarded: when the model calls, only call.Name says which tool it
// meant, and matching that name back to the right argument type is the
// caller's to remember.
//
//	switch call.Name {
//	case "search":
//		args, _ := ai.UnmarshalArgs[SearchArgs](call)      // nothing checks
//	case "population":                                     // that these agree
//		args, _ := ai.UnmarshalArgs[PopulationArgs](call)
//	}
//
// Rename the string, or decode into the neighbouring type, and it still
// compiles. Handle closes that join the way CompleteAs closes the same one for
// structured output: the name, the type and the code that runs it are written
// once, together.

// Handle builds a tool that knows how to run itself.
//
// The argument type is inferred from run, so it is stated once — as run's own
// parameter — and the schema, the decode and the call cannot disagree:
//
//	func population(ctx context.Context, args PopulationArgs) (string, error) {
//		return fmt.Sprintf("%.1f million", census[args.City][args.Year]), nil
//	}
//
//	tools := []ai.Tool{
//		ai.Handle("population", "Population of a city, in millions.", population),
//		ai.Handle("area", "Area of a city, in square kilometres.", area),
//	}
//
// RunTools is what dispatches to it. A tool built with ToolFor instead is
// offered to the model exactly the same way; it simply has nothing to run, and
// RunTools says so rather than guessing.
func Handle[T any](name, description string, run func(context.Context, T) (string, error)) Tool {
	tool := ToolFor[T](name, description)
	tool.run = func(ctx context.Context, arguments string) (string, error) {
		args, err := UnmarshalArgs[T](ToolCall{Name: name, Input: arguments})
		if err != nil {
			return "", err
		}
		return run(ctx, args)
	}
	return tool
}

// Runnable reports whether this tool was built with Handle and can answer its
// own calls.
func (t Tool) Runnable() bool { return t.run != nil }

// RunTools answers every call in a turn, in order, and returns the results as
// one user turn's worth of answers:
//
//	messages = append(messages, response.Message())
//	messages = append(messages, ai.ToolResultsMessage(
//		ai.RunTools(ctx, tools, response.ToolCalls())...))
//
// Nothing here returns an error, because none of what can go wrong is the
// caller's to handle: a mistake in the arguments, an unknown tool name, a tool
// that failed. Every one of those comes back as a result marked IsError, which
// the model sees and can act on. Failing the turn instead would throw away a
// conversation over something the model could have fixed by trying again.
//
// Arguments are checked against the tool's own schema before it runs, so a
// model's mistake stays a mistake the model can correct rather than becoming
// whatever the tool does with a missing field.
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
	if !tool.Runnable() {
		return failed("tool %q was offered without anything to run it", call.Name)
	}
	if err := tool.ValidateArgs(call.Input); err != nil {
		return failed("%v", err)
	}

	output, err := tool.run(ctx, call.Input)
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
