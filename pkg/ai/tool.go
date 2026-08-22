package ai

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/genai-io/sdk-go/pkg/ai/schema"
)

// Tool is a tool definition offered to the model.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`

	// Parameters is a JSON Schema object describing the tool's arguments,
	// typically a map[string]any. Drivers translate it into their wire shape.
	Parameters any `json:"parameters,omitempty"`
}

// ToolFor builds a tool whose parameters are derived from a Go type, so the
// schema the model sees and the struct the arguments decode into cannot drift
// apart.
func ToolFor[T any](name, description string) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  schema.For[T](),
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
	definition := t.ParameterSchema()
	if len(definition) == 0 {
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
	if err := schema.Check(definition, value); err != nil {
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
