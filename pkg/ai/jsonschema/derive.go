package jsonschema

import (
	"fmt"
	"maps"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// The tag key is the JSON Schema keyword it sets.
//
//	type Order struct {
//		Item     string `json:"item" description:"what to order, one line"`
//		Priority string `json:"priority" enum:"low|medium|high"`
//		Quantity int    `json:"quantity" description:"how many" minimum:"1" maximum:"99"`
//	}

// keywords maps a tag key to how its value is written on the wire.
var keywords = map[string]struct{ numeric bool }{
	"description": {},
	"format":      {},
	"pattern":     {},
	"enum":        {},
	"minimum":     {numeric: true},
	"maximum":     {numeric: true},
	"multipleOf":  {numeric: true},
	"minLength":   {numeric: true},
	"maxLength":   {numeric: true},
	"minItems":    {numeric: true},
	"maxItems":    {numeric: true},
}

// applyTags writes a field's keywords onto its schema.
func applyTags(field reflect.StructField, schema map[string]any, t reflect.Type, where string) {
	rejectNearMisses(field, where)
	for key, spec := range keywords {
		value, ok := field.Tag.Lookup(key)
		if !ok {
			continue
		}
		if strings.TrimSpace(value) == "" {
			panic(fmt.Sprintf("jsonschema: %s has an empty %s tag", where, key))
		}
		switch {
		case key == "enum":
			schema["enum"] = parseEnum(value, t, where)
		case spec.numeric:
			n, err := strconv.ParseFloat(value, 64)
			if err != nil {
				panic(fmt.Sprintf("jsonschema: %s sets %s:%q, which is not a number", where, key, value))
			}
			schema[key] = n
		default:
			schema[key] = value
		}
	}
}

// rejectNearMisses catches a keyword that was misspelled or pluralised.
func rejectNearMisses(field reflect.StructField, where string) {
	for _, key := range tagKeys(string(field.Tag)) {
		if _, known := keywords[key]; known || key == "json" {
			continue
		}
		if key == "jsonschema" || key == "ai" {
			panic(fmt.Sprintf("jsonschema: %s carries a %q tag. Keywords go in tags of their own: "+
				"`description:\"…\" enum:\"a|b\"`", where, key))
		}
		for candidate := range keywords {
			if nearMiss(key, candidate) {
				panic(fmt.Sprintf("jsonschema: %s has a %s tag; did you mean %s?", where, key, candidate))
			}
		}
	}
}

// tagKeys lists the keys present in a raw struct tag. reflect can look one up
// but cannot enumerate them, and enumerating is what makes a typo findable.
func tagKeys(tag string) []string {
	var out []string
	for tag != "" {
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		i = 0
		for i < len(tag) && tag[i] > ' ' && tag[i] != ':' && tag[i] != '"' {
			i++
		}
		if i == 0 || i+1 >= len(tag) || tag[i] != ':' || tag[i+1] != '"' {
			return out
		}
		key := tag[:i]
		tag = tag[i+1:]
		i = 1
		for i < len(tag) && tag[i] != '"' {
			if tag[i] == '\\' {
				i++
			}
			i++
		}
		if i >= len(tag) {
			return out
		}
		out = append(out, key)
		tag = tag[i+1:]
	}
	return out
}

// nearMiss reports keys within one edit of each other, plus the plural of a
// keyword — "enums" for "enum" is the mistake people actually make.
func nearMiss(got, want string) bool {
	if got == want+"s" {
		return true
	}
	if abs(len(got)-len(want)) > 1 {
		return false
	}
	lower, upper := strings.ToLower(got), strings.ToLower(want)
	if lower == upper {
		return true
	}
	edits, i, j := 0, 0, 0
	for i < len(lower) && j < len(upper) {
		if lower[i] == upper[j] {
			i, j = i+1, j+1
			continue
		}
		edits++
		if edits > 1 {
			return false
		}
		switch {
		case len(lower) > len(upper):
			i++
		case len(lower) < len(upper):
			j++
		default:
			i, j = i+1, j+1
		}
	}
	return edits+abs(len(lower)-i)+abs(len(upper)-j) <= 1
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// parseEnum types the members by the field they constrain, so a numeric field
// gets numbers rather than the strings they were written as.
func parseEnum(value string, t reflect.Type, where string) []any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	members := strings.Split(value, "|")
	out := make([]any, 0, len(members))
	for _, m := range members {
		m = strings.TrimSpace(m)
		switch t.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			n, err := strconv.ParseInt(m, 10, 64)
			if err != nil {
				panic(fmt.Sprintf("jsonschema: %s has enum member %q, which is not an integer", where, m))
			}
			out = append(out, n)
		case reflect.Float32, reflect.Float64:
			n, err := strconv.ParseFloat(m, 64)
			if err != nil {
				panic(fmt.Sprintf("jsonschema: %s has enum member %q, which is not a number", where, m))
			}
			out = append(out, n)
		default:
			out = append(out, m)
		}
	}
	if len(out) == 0 {
		panic(fmt.Sprintf("jsonschema: %s has an empty enum", where))
	}
	return out
}

// marshalsAs are the types that do not marshal to their own fields. Describing
// one by reflection would produce a schema that rejects the JSON the type
// itself writes.
var marshalsAs = map[reflect.Type]map[string]any{
	reflect.TypeFor[time.Time]():     {"type": "string", "format": "date-time"},
	reflect.TypeFor[time.Duration](): {"type": "integer"},
	reflect.TypeFor[[]byte]():        {"type": "string"},
}

// deriveSchema builds a JSON Schema from a Go type.
func For[T any]() map[string]any { return ForType(reflect.TypeFor[T]()) }

// ForType is For for a type only known at run time, which is what a set of
// tools written as different Go types arrives as.
func ForType(t reflect.Type) map[string]any {
	return schemaForType(t, map[reflect.Type]bool{}, t.String())
}

func schemaForType(t reflect.Type, seen map[reflect.Type]bool, where string) map[string]any {
	// A pointer is the same shape, allowed to be null.
	if t.Kind() == reflect.Pointer {
		return nullable(schemaForType(t.Elem(), seen, where))
	}
	if fixed, ok := marshalsAs[t]; ok {
		return maps.Clone(fixed)
	}

	var out map[string]any
	switch t.Kind() {
	case reflect.String:
		out = map[string]any{"type": "string"}
	case reflect.Bool:
		out = map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		out = map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		out = map[string]any{"type": "number"}

	case reflect.Slice, reflect.Array:
		out = map[string]any{
			"type":  "array",
			"items": schemaForType(t.Elem(), seen, where+" element"),
		}
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			panic(fmt.Sprintf("jsonschema: %s is a map keyed by %s; JSON object keys are strings, "+
				"so this cannot be described to a model", where, t.Key().Kind()))
		}
		out = map[string]any{
			"type":                 "object",
			"additionalProperties": schemaForType(t.Elem(), seen, where+" value"),
		}
	case reflect.Struct:
		out = structSchema(t, seen, where)

	case reflect.Interface:
		panic(fmt.Sprintf("jsonschema: %s is %s, which describes nothing a model can fill in. "+
			"Strict structured output rejects an open schema outright — give the field a "+
			"concrete type, or write the schema by hand", where, kindName(t)))
	default:
		panic(fmt.Sprintf("jsonschema: %s is %s, which has no JSON representation", where, kindName(t)))
	}

	return out
}

func kindName(t reflect.Type) string {
	if t.Name() != "" {
		return t.String()
	}
	return t.Kind().String()
}

// nullable widens a schema's type to admit null, which is how strict
// structured output expresses "this may be absent" — the field stays required
// and the model answers null.
func nullable(schema map[string]any) map[string]any {
	switch t := schema["type"].(type) {
	case string:
		if t != "null" {
			schema["type"] = []any{t, "null"}
		}
	case []any:
		for _, existing := range t {
			if existing == "null" {
				return schema
			}
		}
		schema["type"] = append(t, "null")
	}
	return schema
}

func structSchema(t reflect.Type, seen map[reflect.Type]bool, where string) map[string]any {
	if seen[t] {
		panic(fmt.Sprintf("jsonschema: %s refers to itself; a recursive type has no finite schema", where))
	}
	seen[t] = true
	defer delete(seen, t)

	properties := map[string]any{}
	var required []any
	collectFields(t, seen, where, properties, &required)

	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

// collectFields walks one struct level, following embedded structs so their
// fields land in the parent — which is what encoding/json does, and what a
// hand-rolled walk most often gets wrong.
func collectFields(t reflect.Type, seen map[reflect.Type]bool, where string,
	properties map[string]any, required *[]any) {
	for i := range t.NumField() {
		field := t.Field(i)

		// Promotion is decided first. encoding/json promotes the exported
		// fields of an embedded *unexported* type, so a walk that reaches the
		// exported check before this one drops them without a word — the bug
		// this file's predecessor shipped.
		if field.Anonymous && !namedByTag(field) && !droppedByTag(field) {
			embedded := field.Type
			for embedded.Kind() == reflect.Pointer {
				embedded = embedded.Elem()
			}
			if embedded.Kind() == reflect.Struct {
				collectFields(embedded, seen, where, properties, required)
				continue
			}
		}

		name, optional, skip := jsonFieldName(field)
		if skip || !field.IsExported() {
			continue
		}

		at := where + "." + field.Name
		schema := schemaForType(field.Type, seen, at)
		applyTags(field, schema, field.Type, at)
		if optional {
			schema = nullable(schema)
		}
		properties[name] = schema
		// Every field is required, optional ones included. Strict structured
		// output demands it, and "may be null" is what optional means there.
		*required = append(*required, name)
	}
}

// jsonFieldName applies encoding/json's naming rules: the tag's first segment
// names the property, "-" drops the field, and omitempty or omitzero makes it
// optional.
// namedByTag reports whether a json tag gave the field an explicit name, which
// is what stops an embedded struct from being flattened.
func namedByTag(field reflect.StructField) bool {
	tag, ok := field.Tag.Lookup("json")
	if !ok {
		return false
	}
	name, _, _ := strings.Cut(tag, ",")
	return name != ""
}

// droppedByTag reports a json:"-" field, which encoding/json omits entirely —
// embedded or not.
func droppedByTag(field reflect.StructField) bool {
	tag, ok := field.Tag.Lookup("json")
	return ok && tag == "-"
}

func jsonFieldName(field reflect.StructField) (name string, optional, skip bool) {
	tag, ok := field.Tag.Lookup("json")
	if !ok {
		if !field.IsExported() {
			return "", false, true
		}
		return field.Name, field.Type.Kind() == reflect.Pointer, false
	}
	name, rest, _ := strings.Cut(tag, ",")
	if name == "-" && rest == "" {
		return "", false, true
	}
	for _, opt := range strings.Split(rest, ",") {
		if opt == "omitempty" || opt == "omitzero" {
			optional = true
		}
	}
	if field.Type.Kind() == reflect.Pointer {
		optional = true
	}
	return name, optional, false
}
