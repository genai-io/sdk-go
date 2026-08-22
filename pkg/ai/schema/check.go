package schema

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// Checking a decoded value against a schema.
//
// This is not a JSON Schema conformance engine and does not try to be. It
// exists for one job: a model produced these arguments, so they are model
// output and wrong sometimes, and the mistake should come back as something
// the model can correct rather than as whatever the tool does with nonsense.
//
// The mistakes a model actually makes are a short list — a missing field, a
// string where a number belongs, an invented property, a value outside the set
// it was given — and every one of them is checked here. What is deliberately
// absent is the long tail a model never trips over and a schema derived by this
// package never contains: $ref, allOf/oneOf, patternProperties, dependencies.
// A schema using those is checked as far as this understands it and passes the
// rest, which is the honest failure: it lets a valid call through rather than
// blocking one it cannot judge.
//
// Every message names the property by the path the caller wrote it at, because
// "mode must be one of on, off" is actionable where "validating /properties/
// mode: enum" is not.

// validateAgainst checks a decoded JSON value against a schema. An empty
// schema constrains nothing, which is what a tool declaring no parameters
// wants.
func Check(schema map[string]any, value any) error {
	if len(schema) == 0 {
		return nil
	}
	return checkValue(schema, value, "")
}

func checkValue(schema map[string]any, value any, path string) error {
	if err := checkType(schema, value, path); err != nil {
		return err
	}
	if value == nil {
		return nil
	}
	if allowed, ok := schema["enum"].([]any); ok && !inEnum(allowed, value) {
		return fmt.Errorf("%s must be one of %s", at(path), joinValues(allowed))
	}
	switch v := value.(type) {
	case map[string]any:
		return checkObject(schema, v, path)
	case []any:
		return checkArray(schema, v, path)
	case string:
		return checkString(schema, v, path)
	case float64:
		return checkNumber(schema, v, path)
	}
	return nil
}

// checkType reports a value whose JSON type is not one the schema admits. A
// schema may list several, which is how an optional field is written.
func checkType(schema map[string]any, value any, path string) error {
	wanted := typeNames(schema["type"])
	if len(wanted) == 0 {
		return nil
	}
	got := jsonTypeOf(value)
	for _, want := range wanted {
		if want == got || (want == "number" && got == "integer") {
			return nil
		}
	}
	return fmt.Errorf("%s must be %s, not %s", at(path), joinWords(wanted, "or"), got)
}

func checkObject(schema map[string]any, value map[string]any, path string) error {
	properties, _ := schema["properties"].(map[string]any)

	var missing []string
	for _, name := range stringList(schema["required"]) {
		if _, ok := value[name]; !ok {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("%s is missing %s: %s", atObject(path),
			plural("property", "properties", len(missing)), strings.Join(missing, ", "))
	}

	if open, ok := schema["additionalProperties"].(bool); ok && !open {
		var unknown []string
		for name := range value {
			if _, declared := properties[name]; !declared {
				unknown = append(unknown, name)
			}
		}
		if len(unknown) > 0 {
			sort.Strings(unknown)
			return fmt.Errorf("%s has unknown %s: %s", atObject(path),
				plural("property", "properties", len(unknown)), strings.Join(unknown, ", "))
		}
	}

	// Sorted, so the same bad arguments always come back with the same
	// complaint rather than whichever the map happened to yield first.
	names := make([]string, 0, len(value))
	for name := range value {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		if sub, ok := properties[name].(map[string]any); ok {
			if err := checkValue(sub, value[name], join(path, name)); err != nil {
				return err
			}
			continue
		}
		if sub, ok := schema["additionalProperties"].(map[string]any); ok {
			if err := checkValue(sub, value[name], join(path, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

func checkArray(schema map[string]any, value []any, path string) error {
	if n, ok := numberOf(schema["minItems"]); ok && float64(len(value)) < n {
		return fmt.Errorf("%s needs at least %s items, and has %d", at(path), trim(n), len(value))
	}
	if n, ok := numberOf(schema["maxItems"]); ok && float64(len(value)) > n {
		return fmt.Errorf("%s takes at most %s items, and has %d", at(path), trim(n), len(value))
	}
	items, ok := schema["items"].(map[string]any)
	if !ok {
		return nil
	}
	for i, element := range value {
		if err := checkValue(items, element, fmt.Sprintf("%s[%d]", path, i)); err != nil {
			return err
		}
	}
	return nil
}

func checkString(schema map[string]any, value string, path string) error {
	if n, ok := numberOf(schema["minLength"]); ok && float64(len([]rune(value))) < n {
		return fmt.Errorf("%s must be at least %s characters", at(path), trim(n))
	}
	if n, ok := numberOf(schema["maxLength"]); ok && float64(len([]rune(value))) > n {
		return fmt.Errorf("%s must be at most %s characters", at(path), trim(n))
	}
	if pattern, ok := schema["pattern"].(string); ok {
		re, err := regexp.Compile(pattern)
		if err != nil {
			// The schema's own mistake, not the model's. Saying so keeps a
			// caller from hunting for a bad argument that is not there.
			return fmt.Errorf("%s has an unusable pattern %q: %w", at(path), pattern, err)
		}
		if !re.MatchString(value) {
			return fmt.Errorf("%s must match %s", at(path), pattern)
		}
	}
	return nil
}

func checkNumber(schema map[string]any, value float64, path string) error {
	if n, ok := numberOf(schema["minimum"]); ok && value < n {
		return fmt.Errorf("%s must be at least %s", at(path), trim(n))
	}
	if n, ok := numberOf(schema["maximum"]); ok && value > n {
		return fmt.Errorf("%s must be at most %s", at(path), trim(n))
	}
	if n, ok := numberOf(schema["multipleOf"]); ok && n != 0 {
		if math.Abs(math.Mod(value, n)) > 1e-9 {
			return fmt.Errorf("%s must be a multiple of %s", at(path), trim(n))
		}
	}
	return nil
}

// jsonTypeOf names a decoded value the way a schema does. Integer is reported
// separately from number so "must be integer, not number" can be said.
func jsonTypeOf(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case string:
		return "string"
	case float64:
		if v == math.Trunc(v) {
			return "integer"
		}
		return "number"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return fmt.Sprintf("%T", value)
}

func typeNames(value any) []string {
	switch t := value.(type) {
	case string:
		return []string{t}
	case []any:
		out := make([]string, 0, len(t))
		for _, name := range t {
			if s, ok := name.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func stringList(value any) []string {
	list, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func numberOf(value any) (float64, bool) {
	switch n := value.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

func inEnum(allowed []any, value any) bool {
	for _, candidate := range allowed {
		if candidate == value {
			return true
		}
		// A schema decoded from JSON carries numbers as float64, one written
		// in Go may carry int64; 3 and 3.0 are the same member either way.
		a, aok := numberOf(candidate)
		b, bok := numberOf(value)
		if aok && bok && a == b {
			return true
		}
		if n, ok := candidate.(int64); ok && bok && float64(n) == b {
			return true
		}
	}
	return false
}

// at names the property under complaint. The root has no name, so a complaint
// about it is phrased about the value as a whole.
func at(path string) string {
	if path == "" {
		return "the value"
	}
	return path
}

func atObject(path string) string {
	if path == "" {
		return "the arguments"
	}
	return path
}

func join(path, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}

func joinValues(values []any) string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = fmt.Sprint(v)
	}
	return joinWords(out, "or")
}

func joinWords(words []string, conjunction string) string {
	switch len(words) {
	case 0:
		return ""
	case 1:
		return words[0]
	case 2:
		return words[0] + " " + conjunction + " " + words[1]
	}
	return strings.Join(words[:len(words)-1], ", ") + " " + conjunction + " " + words[len(words)-1]
}

func plural(one, many string, n int) string {
	if n == 1 {
		return one
	}
	return many
}

// trim writes a bound the way it was written in the schema: 1 rather than 1.
func trim(n float64) string {
	if n == math.Trunc(n) {
		return fmt.Sprintf("%d", int64(n))
	}
	return fmt.Sprintf("%g", n)
}
