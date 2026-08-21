// Package jsonschema derives JSON Schemas from Go types and checks values
// against them.
//
// It is a leaf: it imports nothing outside the standard library, not even the
// rest of this SDK. That is deliberate. Deriving a schema from a struct and
// checking a decoded value against one are not language-model problems — they
// are the two halves of keeping a wire description and a Go type from drifting
// apart, and they are worth being able to test, replace, or reuse without
// dragging an inference client along.
//
// Package llm wraps both: llm.SchemaOf and llm.ToolFor build the llm types
// from Of, and Tool.ValidateArgs runs Validate.
package jsonschema

import (
	"fmt"
	"reflect"
	"strings"
)

// Of builds a JSON Schema from a Go type.
//
// Hand-writing a schema as a map[string]any is verbose and, worse, drifts:
// the map and the struct the arguments are decoded into are two descriptions
// of one shape, and nothing keeps them in step. Deriving one from the other
// removes the second description.
//
//	jsonschema.Of[SearchArgs]()
//
// Struct tags:
//
//	json:"name"                 the property name; json:"-" omits the field
//	json:",omitempty"           makes the property optional
//	jsonschema:"description=…"  documents it for the model
//	jsonschema:"enum=a|b|c"     restricts it to a set
//	jsonschema:"required"       requires it despite omitempty
//
// The output is a JSON Schema object suitable for a Tool's Parameters or a
// Schema's Definition. It covers what tool arguments and structured answers
// are made of — objects, arrays, maps, strings, numbers, booleans, pointers,
// and nesting of those. A type it cannot describe precisely is described
// loosely rather than wrongly: an interface or a func field becomes an
// unconstrained value, because claiming a shape the type does not have is the
// worse failure.
func Of[T any]() map[string]any {
	var zero T
	return schemaOfType(reflect.TypeOf(&zero).Elem(), map[reflect.Type]bool{})
}

// schemaOfType renders one type. visiting guards against a self-referential
// struct, which would otherwise recurse forever.
func schemaOfType(t reflect.Type, visiting map[reflect.Type]bool) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.String:
		return map[string]any{"type": "string"}

	case reflect.Slice, reflect.Array:
		// A []byte is base64 text on the wire, not an array of numbers.
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "contentEncoding": "base64"}
		}
		return map[string]any{"type": "array", "items": schemaOfType(t.Elem(), visiting)}

	case reflect.Map:
		return map[string]any{
			"type":                 "object",
			"additionalProperties": schemaOfType(t.Elem(), visiting),
		}

	case reflect.Struct:
		if visiting[t] {
			// A cycle. Describing it loosely beats recursing forever, and
			// beats claiming a shape that would be wrong.
			return map[string]any{"type": "object"}
		}
		visiting[t] = true
		defer delete(visiting, t)
		return structSchema(t, visiting)

	default:
		// Interfaces, funcs, channels: nothing truthful to say about the
		// shape, so say nothing rather than something wrong.
		return map[string]any{}
	}
}

func structSchema(t reflect.Type, visiting map[reflect.Type]bool) map[string]any {
	properties := map[string]any{}
	var required []string

	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		name, optional, skip := fieldName(field)
		if skip {
			continue
		}
		// An embedded struct with no json name contributes its own fields, the
		// way encoding/json flattens it.
		if field.Anonymous && field.Tag.Get("json") == "" {
			embedded := schemaOfType(field.Type, visiting)
			mergeProperties(properties, &required, embedded)
			continue
		}

		prop := schemaOfType(field.Type, visiting)
		forceRequired := applyTags(prop, field.Tag.Get("jsonschema"))
		properties[name] = prop
		if !optional || forceRequired {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
		// Strict mode on every provider rejects unknown properties, and it is
		// what a caller wants regardless: a model inventing an argument is a
		// mistake to surface, not to ignore.
		"additionalProperties": false,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// fieldName reads the json tag the way encoding/json does.
func fieldName(f reflect.StructField) (name string, optional, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	name = f.Name
	parts := strings.Split(tag, ",")
	if parts[0] != "" {
		name = parts[0]
	}
	for _, opt := range parts[1:] {
		if opt == "omitempty" {
			optional = true
		}
	}
	return name, optional, false
}

// applyTags reads the jsonschema tag onto a property, reporting whether it
// forced the field to be required.
func applyTags(prop map[string]any, tag string) (forceRequired bool) {
	if tag == "" {
		return false
	}
	for _, part := range strings.Split(tag, ",") {
		part = strings.TrimSpace(part)
		switch {
		case part == "required":
			forceRequired = true
		case strings.HasPrefix(part, "description="):
			prop["description"] = strings.TrimPrefix(part, "description=")
		case strings.HasPrefix(part, "enum="):
			values := strings.Split(strings.TrimPrefix(part, "enum="), "|")
			enum := make([]any, len(values))
			for i, v := range values {
				enum[i] = v
			}
			prop["enum"] = enum
		}
	}
	return forceRequired
}

// mergeProperties folds an embedded struct's schema into its parent.
func mergeProperties(into map[string]any, required *[]string, embedded map[string]any) {
	if props, ok := embedded["properties"].(map[string]any); ok {
		for k, v := range props {
			into[k] = v
		}
	}
	if req, ok := embedded["required"].([]string); ok {
		*required = append(*required, req...)
	}
}

// Validate checks a decoded JSON value against a schema.
//
// It is not a complete JSON Schema implementation, and does not pretend to be
// one: it checks the keywords a tool call or a structured answer actually gets
// wrong — a missing required property, a value of the wrong type, a string
// outside its enum, an invented property — and ignores the rest. A validator
// that silently accepted those would be worse than none, and one that
// implemented every keyword would be a library of its own.
func Validate(schema map[string]any, value any) error {
	return validateValue(schema, value, "")
}

func validateValue(schema map[string]any, value any, path string) error {
	if len(schema) == 0 {
		return nil
	}
	if enum, ok := schema["enum"].([]any); ok {
		if !containsValue(enum, value) {
			return schemaErr(path, "must be one of %v, got %v", enum, value)
		}
	}

	declared, _ := schema["type"].(string)
	switch declared {
	case "":
		return nil
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			return schemaErr(path, "must be an object, got %s", jsonKind(value))
		}
		return validateObject(schema, obj, path)
	case "array":
		arr, ok := value.([]any)
		if !ok {
			return schemaErr(path, "must be an array, got %s", jsonKind(value))
		}
		items, _ := schema["items"].(map[string]any)
		for i, item := range arr {
			if err := validateValue(items, item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	case "string":
		if _, ok := value.(string); !ok {
			return schemaErr(path, "must be a string, got %s", jsonKind(value))
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return schemaErr(path, "must be a boolean, got %s", jsonKind(value))
		}
	case "integer":
		f, ok := value.(float64)
		if !ok {
			return schemaErr(path, "must be an integer, got %s", jsonKind(value))
		}
		if f != float64(int64(f)) {
			return schemaErr(path, "must be an integer, got %v", f)
		}
	case "number":
		if _, ok := value.(float64); !ok {
			return schemaErr(path, "must be a number, got %s", jsonKind(value))
		}
	}
	return nil
}

func validateObject(schema map[string]any, obj map[string]any, path string) error {
	properties, _ := schema["properties"].(map[string]any)

	for _, name := range requiredNames(schema) {
		if _, present := obj[name]; !present {
			return schemaErr(join(path, name), "is required but missing")
		}
	}
	// additionalProperties: false is what catches a model inventing an
	// argument, which is otherwise silently passed to the tool.
	if allowed, ok := schema["additionalProperties"].(bool); ok && !allowed {
		for name := range obj {
			if _, declared := properties[name]; !declared {
				return schemaErr(join(path, name), "is not a known property")
			}
		}
	}
	for name, raw := range obj {
		prop, ok := properties[name].(map[string]any)
		if !ok {
			continue
		}
		if err := validateValue(prop, raw, join(path, name)); err != nil {
			return err
		}
	}
	return nil
}

// requiredNames reads the required list, which arrives as []string when built
// in Go and []any when decoded from JSON.
func requiredNames(schema map[string]any) []string {
	switch req := schema["required"].(type) {
	case []string:
		return req
	case []any:
		out := make([]string, 0, len(req))
		for _, v := range req {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func containsValue(enum []any, value any) bool {
	for _, allowed := range enum {
		if allowed == value {
			return true
		}
	}
	return false
}

func jsonKind(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "a boolean"
	case float64:
		return "a number"
	case string:
		return "a string"
	case []any:
		return "an array"
	case map[string]any:
		return "an object"
	default:
		return fmt.Sprintf("%T", v)
	}
}

func join(path, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}

func schemaErr(path, format string, args ...any) error {
	where := path
	if where == "" {
		where = "value"
	}
	return fmt.Errorf("%s %s", where, fmt.Sprintf(format, args...))
}
