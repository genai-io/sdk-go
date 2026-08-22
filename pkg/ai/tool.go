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

	// run answers this tool's calls, and is set by Handle. It is unexported
	// because it is behaviour rather than definition: a Tool decoded from JSON
	// carries what the model is told and nothing that could be executed.
	run func(ctx context.Context, arguments string) (string, error)
}

// ToolInfo is what a model is told about a tool besides its arguments: the
// name it calls, and what the tool is for.
type ToolInfo struct {
	// Name is what the model puts in ToolCall.Name. It has to be stable —
	// renaming it mid-conversation orphans the calls already in the history.
	Name string
	// Description is prompt text. It is the model's only guide to when this
	// tool applies rather than the one next to it.
	Description string
}

// ToolDescriber is implemented by a tool's argument type, and is what keeps a
// tool from being three things in three places.
//
// The type already carries the arguments and their descriptions. Making it
// carry the name and the purpose too means everything the model is told about
// the tool is in one declaration, and the string the model calls sits next to
// the fields it will fill in:
//
//	type Search struct {
//		Query string `json:"query" description:"what to look for"`
//		Limit int    `json:"limit,omitempty" description:"how many" maximum:"20"`
//	}
//
//	func (Search) Tool() ai.ToolInfo {
//		return ai.ToolInfo{Name: "search", Description: "Search the knowledge base."}
//	}
type ToolDescriber interface {
	Tool() ToolInfo
}

// ToolFor builds a tool from its argument type: the schema from the fields,
// the name and purpose from the type's own Tool method.
//
// It defines the tool without saying what runs it, which leaves you to match
// call.Name back to the right argument type yourself. Handle does both at once
// and is what most callers want.
func ToolFor[T ToolDescriber]() Tool {
	var zero T
	info := zero.Tool()
	if info.Name == "" {
		panic(fmt.Sprintf("ai: %T returns a ToolInfo with no Name; "+
			"that is the string the model calls", zero))
	}
	return Tool{
		Name:        info.Name,
		Description: info.Description,
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
	if err := decodeArgs(call.Input, &out); err != nil {
		return out, fmt.Errorf("arguments for %s: %w", call.Name, err)
	}
	return out, nil
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

// ToolRunner is a tool that is one Go type: its fields are the arguments, its
// tags say what each one means, and its two methods say what it is called and
// what it does.
type ToolRunner interface {
	ToolDescriber
	// Run answers one call. The receiver is this tool's own value with the
	// model's arguments filled in.
	Run(ctx context.Context) (string, error)
}

// ToolOf builds a tool from a value of the type that implements it.
//
// Everything is in one declaration — the arguments, what they mean, the name
// the model calls, and the code that answers:
//
//	type Search struct {
//		Query string `json:"query" description:"what to look for"`
//		Limit int    `json:"limit,omitempty" description:"how many" maximum:"20"`
//
//		index *Index // unexported: yours, not the model's
//	}
//
//	func (Search) Tool() ai.ToolInfo {
//		return ai.ToolInfo{Name: "search", Description: "Search the knowledge base."}
//	}
//
//	func (s Search) Run(ctx context.Context) (string, error) {
//		return s.index.Query(ctx, s.Query, s.Limit)
//	}
//
//	tools := []ai.Tool{ai.ToolOf(Search{index: idx}), ai.ToolOf(Fetch{store: db})}
//
// The value you pass is the tool's dependencies. Each call runs against a copy
// of it with the model's arguments decoded over the top, so what the model
// sends fills the exported fields and everything unexported stays as you set
// it — the same split encoding/json already draws, and the same one the schema
// draws, since an unexported field is never described to the model.
//
// A copy per call, so two calls in one turn cannot see each other's arguments.
// T must be a struct rather than a pointer to one, for that reason: a pointer
// would be shared.
func ToolOf[T ToolRunner](prototype T) Tool {
	if kind := reflect.TypeFor[T]().Kind(); kind != reflect.Struct {
		panic(fmt.Sprintf("ai: ToolOf needs a struct, not %s: a copy per call is what "+
			"keeps two calls in one turn from sharing arguments", kind))
	}
	tool := ToolFor[T]()
	tool.run = func(ctx context.Context, arguments string) (string, error) {
		args := prototype // the dependencies; the model fills in the rest
		if err := decodeArgs(arguments, &args); err != nil {
			return "", fmt.Errorf("arguments for %s: %w", tool.Name, err)
		}
		return args.Run(ctx)
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

// maxToolTurns bounds Run against a model that keeps calling tools. It is
// deliberately generous — a real conversation ends in a handful of turns — and
// hitting it means something is wrong, not that the budget was too small.
const maxToolTurns = 32

// Run holds a conversation to the end, answering tool calls as they arrive.
//
// The loop it replaces is the same in every application — complete, run the
// calls, append the results, repeat — so it is written here once:
//
//	response, history, err := client.Run(ctx, messages, ai.WithTools(tools...))
//	fmt.Println(response.Text())
//
// It returns the final response, and the whole conversation including every
// turn it added, so the next question continues from where this one finished:
//
//	response, history, err = client.Run(ctx, append(history, ai.UserMessage(next)), ai.WithTools(tools...))
//
// Tools come from WithTools like any other setting, and a tool built with
// ToolFor rather than Handle has nothing to run — Run says so in the result
// rather than guessing.
//
// Write the loop yourself when the turns are your business: to stream text as
// it arrives, to stop on a condition, to log or bill each turn, or to decide
// per-turn whether to continue. Run stops when the model stops asking, when
// ctx is done, or after maxToolTurns, which is a runaway guard rather than a
// budget — cost control is Middleware's job.
func (c *Client) Run(ctx context.Context, messages []Message, opts ...Option) (*Response, []Message, error) {
	// The tools are resolved the same way the request resolves them, so what
	// Run dispatches to is exactly what the model was offered.
	tools := newRequest(c.model, nil, c.defaults, opts).Tools

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
