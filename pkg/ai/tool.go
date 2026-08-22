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
	//
	// It is skipped by encoding/json: what a Tool marshals to is what the
	// model is told, which never includes anything executable.
	Run func(ctx context.Context, arguments string) (string, error) `json:"-"`
}

// Doc is how a tool says what it is called and what it does — the two things a
// parameter schema does not say, written in the same tags the parameters use.
//
//	type Search struct {
//		_ ai.Doc `name:"search" description:"Search the docs and return matching passages."`
//
//		Query string `json:"query" description:"what to look for, in plain words"`
//	}
//
// Go attaches tags to fields, so a field is how a type says something about
// itself. This one is blank and zero-sized: encoding/json never sees it, the
// schema never describes it, and no code can read or set it. It is not data.
type Doc struct{}

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

// ToolFunc builds a tool from one function.
//
//	search := ai.ToolFunc(func(ctx context.Context, a Search) (string, error) {
//		return docs.Search(ctx, a.Query, a.Limit)
//	})
//
//	client.Run(ctx, messages, []ai.Tool{search, fetch})
//
// Everything the model is told comes from T — its name and description from
// the ai.Doc field, its parameters from the fields themselves. The schema that
// goes out and the struct the arguments arrive in are therefore the same
// declaration, and cannot come to describe different things.
//
// Every word of that declaration is prompt text: describe each parameter, and
// use enum where a field has a fixed set of answers. Arguments are checked
// against the schema before the function is called.
//
// Dependencies are whatever the function closes over. That is the ordinary Go
// answer, and it needs nothing from this package.
//
// The result is an ordinary Tool. Assign to its Parameters afterwards to send
// a hand-written schema in place of the derived one.
func ToolFunc[T any](run func(ctx context.Context, arguments T) (string, error)) Tool {
	var zero T
	t := reflect.TypeOf(zero)
	name, description := toolDoc(t)
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

// toolDoc reads what a tool says about itself.
//
// Everything it can find wrong is a mistake in a declaration rather than a
// condition to handle, so it panics: a tool with no name is one the model
// cannot call, and the panic happens where the tool is built rather than in
// the middle of a conversation.
func toolDoc(t reflect.Type) (name, description string) {
	if t == nil || t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("ai: a tool's arguments must be a struct, not %v", t))
	}
	doc := reflect.TypeOf(Doc{})
	for i := range t.NumField() {
		field := t.Field(i)
		if field.Type != doc {
			continue
		}
		if field.Name != "_" {
			panic(fmt.Sprintf("ai: %s.%s must be blank — write `_ ai.Doc`, or the "+
				"model is told about it as a parameter", t, field.Name))
		}
		name, description = field.Tag.Get("name"), field.Tag.Get("description")
		if name == "" {
			panic(fmt.Sprintf("ai: the ai.Doc on %s has no name tag; "+
				"that is the string the model calls", t))
		}
		return name, description
	}
	panic(fmt.Sprintf("ai: %s does not say what it is called; give it\n"+
		"\t_ ai.Doc `name:\"…\" description:\"…\"`", t))
}

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
// The loop it replaces is the same in every application — complete, run the
// calls, append the results, repeat — so it is written here once:
//
//	response, history, err := client.Run(ctx, messages,
//		[]ai.Tool{search, fetch})
//
//	fmt.Println(response.Text())
//
// It returns the final response, and the whole conversation including every
// turn it added, so the next question continues from where this one finished:
//
//	response, history, err = client.Run(ctx, append(history, ai.UserMessage(next)), tools)
//
// Tools are a parameter rather than an option because a Run without them is a
// Complete. A Tool with no Run — one defined but not implemented — is reported
// in the result rather than guessed at.
//
// Write the loop yourself when the turns are your business: to stream text as
// it arrives, to stop on a condition, to log or bill each turn, or to decide
// per-turn whether to continue. Run stops when the model stops asking, when
// ctx is done, or after maxToolTurns, which is a runaway guard rather than a
// budget — cost control is Middleware's job.
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
