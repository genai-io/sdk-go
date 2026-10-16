package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// Schema describes a tool: its name, what it is for, and the JSON arguments it
// takes.
type Schema struct {
	// Name is the string the model calls the tool by.
	Name string `json:"name,omitempty"`

	// Description tells the model what the tool is for. Providers pass it to
	// the model, so it is prompt text, not documentation.
	Description string `json:"description,omitempty"`

	// Definition is the JSON Schema of the arguments, typically a
	// map[string]any.
	Definition any `json:"definition,omitempty"`

	// Strict asks for exact conformance where the protocol distinguishes it
	// from best-effort. Strict mode accepts a narrower subset of JSON Schema —
	// no open-ended objects, every property required — so a schema that works
	// without it may be rejected with it.
	Strict bool `json:"strict,omitempty"`
}

// ToolSchema derives a tool's schema from its Go argument type.
//
//	type SearchArgs struct {
//		Query string `json:"query" description:"what to look for, in plain words"`
//		Limit int    `json:"limit,omitempty" description:"how many passages" maximum:"10"`
//	}
//
//	search := ai.ToolSchema[SearchArgs]("search", "Search the docs and return matching passages.")
func ToolSchema[T any](name, description string) Schema {
	t := reflect.TypeFor[T]()
	if t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("ai: a tool's arguments must be a struct, not %v", t))
	}
	if name == "" {
		panic(fmt.Sprintf("ai: the tool taking %s has no name; "+
			"that is the string the model calls", t))
	}
	return Schema{Name: name, Description: description, Definition: jsonschema.ForType(t)}
}

// DefinitionMap returns an independent JSON-object representation of the
// schema, or nil when Definition cannot be represented as a JSON object.
func (s Schema) DefinitionMap() map[string]any {
	switch def := s.Definition.(type) {
	case map[string]any:
		return maps.Clone(def)
	case nil:
		return nil
	default:
		// A caller who built the schema from a struct or a typed value gets it
		// re-encoded rather than dropped.
		raw, err := json.Marshal(def)
		if err != nil {
			return nil
		}
		var out map[string]any
		if json.Unmarshal(raw, &out) != nil {
			return nil
		}
		return out
	}
}

// Validate checks a JSON document against the schema. An empty document is
// read as an empty object, which is what every protocol sends for a call that
// takes no arguments. A schema with no definition accepts anything.
func (s Schema) Validate(input string) error {
	definition := s.DefinitionMap()
	if len(definition) == 0 {
		return nil
	}
	name := s.Name
	if name == "" {
		name = "the schema"
	}
	var value any
	trimmed := bytes.TrimSpace([]byte(input))
	if len(trimmed) == 0 {
		value = map[string]any{}
	} else if err := json.Unmarshal(trimmed, &value); err != nil {
		return fmt.Errorf("arguments for %s are not valid JSON: %w", name, err)
	}
	if err := jsonschema.Check(definition, value); err != nil {
		return fmt.Errorf("arguments for %s: %w", name, err)
	}
	return nil
}
