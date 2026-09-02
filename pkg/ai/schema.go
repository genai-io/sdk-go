package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// Schema constrains an answer to a JSON shape.
type Schema struct {
	// Name identifies the shape. Some protocols require one, and it is what
	// appears in an error when the answer does not match.
	Name string `json:"name,omitempty"`

	// Description tells the model what the shape is for. Providers pass it to
	// the model, so it is prompt text, not documentation.
	Description string `json:"description,omitempty"`

	// Definition is the JSON Schema itself, typically a map[string]any. It is
	// the same shape a Tool takes for its parameters.
	Definition any `json:"definition,omitempty"`

	// Strict asks for exact conformance where the protocol distinguishes it
	// from best-effort. Strict mode accepts a narrower subset of JSON Schema —
	// no open-ended objects, every property required — so a schema that works
	// without it may be rejected with it.
	Strict bool `json:"strict,omitempty"`
}

// SchemaOf builds a Schema from a Go type, ready for WithSchema. The schema is
// named after T; description is prompt text the model reads, and may be empty
// when the type name already says what the shape is for.
//
//	type Order struct {
//		Name     string `json:"name" description:"full name, family name last"`
//		Priority string `json:"priority" enum:"low|medium|high"`
//		Quantity int    `json:"quantity" description:"how many" minimum:"1" maximum:"99"`
//	}
func SchemaOf[T any](description string) *Schema {
	return &Schema{
		Name:        reflect.TypeFor[T]().Name(),
		Description: description,
		Definition:  jsonschema.For[T](),
		Strict:      true,
	}
}

// CompleteAs asks for an answer shaped like T and decodes it into one.
//
//	ai.Parse[Company](client.Complete(ctx, msgs,
//		ai.WithSchema(ai.SchemaOf[Person]("…"))))   // compiles, fails at runtime
//
//	company, err := ai.CompleteAs[Company](ctx, client, messages)
func CompleteAs[T any](ctx context.Context, c *Client, messages []Message, opts ...Option) (T, error) {
	withDerived := append([]Option{WithSchema(SchemaOf[T](""))}, opts...)
	return Parse[T](c.Complete(ctx, messages, withDerived...))
}

// DefinitionMap returns an independent JSON-object representation of the
// schema, or nil when Definition cannot be represented as a JSON object.
func (s Schema) DefinitionMap() map[string]any {
	return jsonSchemaObject(s.Definition)
}

// Validate checks a JSON document against the schema. An empty document is
// read as an empty object, which is what every protocol sends for a call that
// takes no arguments. A schema with no definition accepts anything.
func (s Schema) Validate(input string) error {
	definition := jsonSchemaObject(s.Definition)
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

// WireName is the identifier to send for a protocol that requires a schema to
// be named. It is the schema's own Name, or "response" when it states none.
func (s Schema) WireName() string {
	if s.Name == "" {
		return "response"
	}
	return s.Name
}

// jsonSchemaObject renders a schema definition as an independent JSON object.
func jsonSchemaObject(value any) map[string]any {
	switch def := value.(type) {
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

func cloneSchema(schema *Schema) *Schema {
	if schema == nil {
		return nil
	}
	out := *schema
	// A map is the one definition shape a caller can still be holding and edit
	// after handing it over; anything else is copied by the assignment above.
	if def, ok := schema.Definition.(map[string]any); ok {
		out.Definition = maps.Clone(def)
	}
	return &out
}
