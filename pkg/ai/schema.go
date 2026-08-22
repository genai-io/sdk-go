package ai

import (
	"context"
	"encoding/json"
	"maps"
	"reflect"

	"github.com/genai-io/sdk-go/pkg/ai/schema"
)

// Asking a model for a shape, and the JSON Schema machinery behind it.
//
// Two different things are called a schema here, and they are worth keeping
// apart. A Schema is this SDK's request: a name, a description the model
// reads, and the shape itself. The shape is a JSON Schema document — an
// ordinary map — built and checked by the second half of this file.
//
// Building one from a Go type, and checking a value against one, live in
// ai/schema. That package targets what the providers accept rather than JSON
// Schema in general, which is a different and stricter target.
//
// Reading the answer back is in response.go.

// Schema constrains an answer to a JSON shape.
//
// Without it, getting structured data out of a model means asking for JSON in
// the prompt and scraping the reply — which fails in a long tail of ways that
// all look like the model misbehaving: a markdown fence, a "Sure, here you
// go!" preamble, a trailing paragraph of commentary, a truncated object. Every
// protocol here can constrain generation properly, so none of that is
// necessary.
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
// Per-field constraints come from an ai struct tag, and the model reads them.
// Field names alone are often ambiguous to it in ways they are not to you, and
// a field with a fixed set of answers should say so rather than hope:
//
//	type Order struct {
//		Name     string `json:"name" description:"full name, family name last"`
//		Priority string `json:"priority" enum:"low|medium|high"`
//		Quantity int    `json:"quantity" description:"how many" minimum:"1" maximum:"99"`
//	}
//
// A tag key is the JSON Schema keyword it sets; see ai/schema for the full
// list and what it refuses.
func SchemaOf[T any](description string) *Schema {
	return &Schema{
		Name:        reflect.TypeFor[T]().Name(),
		Description: description,
		Definition:  schema.For[T](),
		Strict:      true,
	}
}

// CompleteAs asks for an answer shaped like T and decodes it into one.
//
// It is the whole round trip in a single call — derive the schema from T,
// constrain generation to it, unmarshal the answer — and T is named once.
// Spelling SchemaOf and Parse separately lets them disagree:
//
//	ai.Parse[Company](client.Complete(ctx, msgs,
//		ai.WithSchema(ai.SchemaOf[Person]("…"))))   // compiles, fails at runtime
//
// This cannot:
//
//	company, err := ai.CompleteAs[Company](ctx, client, messages)
//
// It is a function rather than a method because Go has no generic methods.
// Pass WithSchema to describe the shape to the model, or to constrain to one T
// does not capture exactly; options apply in order, so a caller's schema wins
// over the derived one.
func CompleteAs[T any](ctx context.Context, c *Client, messages []Message, opts ...Option) (T, error) {
	withDerived := append([]Option{WithSchema(SchemaOf[T](""))}, opts...)
	return Parse[T](c.Complete(ctx, messages, withDerived...))
}

// DefinitionMap returns an independent JSON-object representation of the
// schema, or nil when Definition cannot be represented as a JSON object.
//
// Definition may be a map or a typed Go value. Keeping the conversion here
// gives every driver the same behavior and prevents one protocol from silently
// dropping a schema another protocol accepts.
func (s *Schema) DefinitionMap() map[string]any {
	if s == nil {
		return nil
	}
	return jsonSchemaObject(s.Definition)
}

// WireName is the identifier to send for a protocol that requires a schema to
// be named. It is the schema's own Name, or "response" when it states none.
//
// Every protocol that demands a name gets the same one from here, so a schema
// that moves between providers keeps its identity — and a driver does not each
// invent its own fallback.
func (s *Schema) WireName() string {
	if s == nil || s.Name == "" {
		return "response"
	}
	return s.Name
}

// jsonSchemaObject renders a schema definition as an independent JSON object.
// It is shared by Schema.DefinitionMap and Tool.ParameterSchema so that the
// two places a JSON Schema enters this package accept exactly the same values.
func jsonSchemaObject(value any) map[string]any {
	switch def := value.(type) {
	case map[string]any:
		return cloneStringMap(def)
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
	out.Definition = cloneShallow(schema.Definition)
	return &out
}

func cloneShallow(value any) any {
	if m, ok := value.(map[string]any); ok {
		return maps.Clone(m)
	}
	return value
}
