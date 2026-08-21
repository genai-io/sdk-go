package ai

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// Asking a model for a shape, and the JSON Schema machinery behind it.
//
// Two different things are called a schema here, and they are worth keeping
// apart. A Schema is this SDK's request: a name, a description the model
// reads, and the shape itself. The shape is a JSON Schema document — an
// ordinary map — built and checked by the second half of this file.
//
// That half is the only place in the SDK that touches
// github.com/google/jsonschema-go. Both jobs are deep enough to be worth
// borrowing rather than writing: deriving is not the mechanical type walk it
// looks like, because a type with its own MarshalJSON has to be described by
// what it marshals to, and validating is a specification with a long tail.
// Nothing the library returns escapes this file — callers see map[string]any
// and plain errors — so replacing it means rewriting the bottom of one file.
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

// SchemaOf builds a Schema from a Go type, ready for WithSchema.
func SchemaOf[T any](name, description string) *Schema {
	return &Schema{
		Name:        name,
		Description: description,
		Definition:  deriveSchema[T](),
		Strict:      true,
	}
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

// ─── JSON Schema documents: deriving one, and checking a value against one ───

// byteSchema corrects the one translation upstream gets wrong for this SDK's
// purposes. encoding/json writes a []byte as a base64 string; upstream
// describes it as an array of 0-255 integers, which its own validator then
// rejects when handed real JSON.
var byteSchema = map[reflect.Type]*jsonschema.Schema{
	reflect.TypeFor[[]byte](): {Type: "string", ContentEncoding: "base64"},
}

// deriveSchema builds a JSON Schema from a Go type.
//
// Field names, optionality and nesting follow encoding/json: a json tag names
// the property, omitempty makes it optional, an embedded struct is flattened,
// and a type with its own MarshalJSON is described by what it marshals to. A
// jsonschema struct tag becomes the property's description, which is what the
// model reads.
//
// A type that cannot be described — a channel, a function, a map with
// non-string keys, a malformed struct tag — panics, the way a bad pattern
// panics in regexp.MustCompile. It is a mistake in the caller's own type, it is
// found the moment the tool is constructed rather than mid-conversation, and
// the alternative is worse: a nil schema silently means "no parameters", so the
// model would be offered a tool it cannot call and its arguments would never be
// checked.
func deriveSchema[T any]() map[string]any {
	schema, err := jsonschema.For[T](&jsonschema.ForOptions{TypeSchemas: byteSchema})
	if err != nil {
		panic(fmt.Sprintf("ai: cannot describe %s as a JSON Schema: %v", reflect.TypeFor[T](), err))
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		panic(fmt.Sprintf("ai: cannot encode the schema for %s: %v", reflect.TypeFor[T](), err))
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		panic(fmt.Sprintf("ai: cannot read back the schema for %s: %v", reflect.TypeFor[T](), err))
	}
	return out
}

// validateAgainst checks a decoded JSON value against a schema.
//
// The schema may have been derived here, hand-written as a map, or reloaded
// from a session file, so it arrives as a map and is parsed rather than
// assumed. An empty schema constrains nothing, which is what a tool that
// declares no parameters wants.
func validateAgainst(schema map[string]any, value any) error {
	if len(schema) == 0 {
		return nil
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("schema is not encodable: %w", err)
	}
	var parsed jsonschema.Schema
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return fmt.Errorf("not a usable JSON Schema: %w", err)
	}
	resolved, err := parsed.Resolve(nil)
	if err != nil {
		return fmt.Errorf("schema cannot be applied: %w", err)
	}
	if err := resolved.Validate(value); err != nil {
		return schemaFailure(err)
	}
	return nil
}

// schemaFailure rewrites a validation failure into the sentence a model is handed
// back as a tool result.
//
// Upstream reports a schema location — "validating root: validating
// /properties/mode: type: …" — which is right for a person debugging a schema
// and wrong for a model that has to work out which argument to fix. The
// property path is what it needs, and nothing else.
func schemaFailure(err error) error {
	msg := strings.TrimPrefix(err.Error(), "validating root: ")
	msg = strings.TrimPrefix(msg, "validating ")

	// A property path prefixes the reason; render it the way the caller wrote
	// the argument rather than as a schema location.
	if location, rest, ok := strings.Cut(msg, ": "); ok && strings.HasPrefix(location, "/") {
		if name := propertyPath(location); name != "" {
			return fmt.Errorf("%s %s", name, strings.TrimPrefix(rest, "type: "))
		}
	}
	// The two mistakes a model makes most often are reported against the whole
	// object rather than one property, and read as schema vocabulary. Say them
	// as the instruction they are.
	if names, ok := bracketed(msg, "required: missing properties: "); ok {
		return fmt.Errorf("missing required %s: %s", plural("property", names), strings.Join(names, ", "))
	}
	if names, ok := bracketed(msg, "unexpected additional properties "); ok {
		return fmt.Errorf("unknown %s: %s", plural("property", names), strings.Join(names, ", "))
	}
	return errors.New(msg)
}

// bracketed reads the JSON list of property names that upstream appends to its
// object-level failures.
func bracketed(msg, prefix string) ([]string, bool) {
	rest, found := strings.CutPrefix(msg, prefix)
	if !found {
		return nil, false
	}
	var names []string
	if json.Unmarshal([]byte(rest), &names) != nil || len(names) == 0 {
		return nil, false
	}
	return names, true
}

func plural(word string, of []string) string {
	if len(of) == 1 {
		return word
	}
	return word[:len(word)-1] + "ies"
}

// propertyPath turns a JSON Schema location into a dotted property name,
// dropping the keywords that structure the schema rather than the value.
func propertyPath(location string) string {
	var parts []string
	for segment := range strings.SplitSeq(strings.TrimPrefix(location, "/"), "/") {
		switch segment {
		case "properties", "items", "additionalProperties", "":
		default:
			parts = append(parts, segment)
		}
	}
	return strings.Join(parts, ".")
}
