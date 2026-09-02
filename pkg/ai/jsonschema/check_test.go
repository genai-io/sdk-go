package jsonschema_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// searchArgs carries one of every constraint the checker enforces, so the
// cases below run against a schema a caller would really have derived.
type searchArgs struct {
	Query    string         `json:"query" description:"what to look for" minLength:"2" maxLength:"40" pattern:"^[a-z ]+$"`
	Priority string         `json:"priority" enum:"low|medium|high"`
	Limit    int            `json:"limit,omitempty" minimum:"5" maximum:"50" multipleOf:"5"`
	Tags     []string       `json:"tags,omitempty" minItems:"1" maxItems:"3"`
	Filters  map[string]int `json:"filters,omitempty"`
	Nested   *nestedArgs    `json:"nested"`
}

type nestedArgs struct {
	Name string `json:"name"`
	Note string `json:"note,omitempty"`
}

func TestCheckAccepts(t *testing.T) {
	schema := jsonschema.For[searchArgs]()

	for name, input := range map[string]string{
		"every argument, filled in": `{"query":"go rocks","priority":"low","limit":5,
			"tags":["a"],"filters":{"x":1},"nested":{"name":"n","note":"hi"}}`,

		// A model on a provider that is not strict omits the arguments it has
		// nothing to say about, though strict mode had them listed as required.
		"the optional arguments left out": `{"query":"go","priority":"low"}`,
		"an optional argument left out one level down": `{"query":"go","priority":"low",
			"nested":{"name":"n"}}`,

		// The other half of the same union: a strict model answers null.
		"the optional arguments answered null": `{"query":"go","priority":"low","limit":null,
			"tags":null,"filters":null,"nested":null}`,

		"an empty collection where the bounds allow one": `{"query":"go","priority":"low",
			"tags":["a","b","c"],"filters":{}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := jsonschema.Check(schema, decode(t, input)); err != nil {
				t.Errorf("a legitimate call was rejected: %v", err)
			}
		})
	}
}

func TestCheckRejects(t *testing.T) {
	schema := jsonschema.For[searchArgs]()

	for name, tc := range map[string]struct{ input, want string }{
		"an argument with nothing optional about it": {
			`{"priority":"low"}`, "missing required property: query",
		},
		"two of them, named in a stable order": {
			`{}`, "missing required properties: priority, query",
		},
		"a missing argument one level down": {
			`{"query":"go","priority":"low","nested":{}}`, "nested: missing required property: name",
		},
		"a value outside the enum": {
			`{"query":"go","priority":"urgent"}`, "priority must be one of low, medium or high",
		},
		"a string where a number belongs": {
			`{"query":"go","priority":"low","limit":"five"}`, "limit must be integer or null, not string",
		},
		"a number out of range": {
			`{"query":"go","priority":"low","limit":900}`, "limit must be at most 50",
		},
		"a number that does not step": {
			`{"query":"go","priority":"low","limit":7}`, "limit must be a multiple of 5",
		},
		"a string the pattern refuses": {
			`{"query":"GO","priority":"low"}`, "query must match ^[a-z ]+$",
		},
		"a string too short": {
			`{"query":"g","priority":"low"}`, "query must be at least 2 characters",
		},
		"too few items": {
			`{"query":"go","priority":"low","tags":[]}`, "tags needs at least 1 items, and has 0",
		},
		"too many items": {
			`{"query":"go","priority":"low","tags":["a","b","c","d"]}`,
			"tags takes at most 3 items, and has 4",
		},
		"an item of the wrong type": {
			`{"query":"go","priority":"low","tags":[1]}`, "tags[0] must be string, not integer",
		},
		"a map value of the wrong type": {
			`{"query":"go","priority":"low","filters":{"x":"one"}}`, "filters.x must be integer, not string",
		},
		"an invented argument": {
			`{"query":"go","priority":"low","sort":"asc"}`, "unknown property: sort",
		},
		"an invented argument one level down": {
			`{"query":"go","priority":"low","nested":{"name":"n","extra":1}}`,
			"nested: unknown property: extra",
		},
		"arguments that are not an object at all": {
			`["go"]`, "the value must be object, not array",
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := jsonschema.Check(schema, decode(t, tc.input))
			if err == nil {
				t.Fatalf("%s was accepted", tc.input)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %q\nwant it to contain %q", err, tc.want)
			}
		})
	}
}

// A property is missing only when its own schema refuses null. Written by hand,
// because that is the difference the derived schema hides behind omitempty.
func TestCheckReadsRequiredByWhatTheTypeAdmits(t *testing.T) {
	for name, tc := range map[string]struct {
		property map[string]any
		wantErr  bool
	}{
		"a plain type is genuinely required":     {map[string]any{"type": "string"}, true},
		"a union with null stands in for absent": {map[string]any{"type": []any{"string", "null"}}, false},
		"a null-only type admits nothing else":   {map[string]any{"type": "null"}, false},
		"an unconstrained property is required":  {map[string]any{"description": "anything"}, true},
	} {
		t.Run(name, func(t *testing.T) {
			schema := map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"note": tc.property},
				"required":             []any{"note"},
				"additionalProperties": false,
			}
			err := jsonschema.Check(schema, map[string]any{})
			if tc.wantErr && err == nil {
				t.Error("an absent property that cannot be null was accepted")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("an absent nullable property was rejected: %v", err)
			}
		})
	}

	// A name in required that no property describes says nothing about null,
	// so it stays required.
	schema := map[string]any{"type": "object", "required": []any{"note"}}
	if err := jsonschema.Check(schema, map[string]any{}); err == nil {
		t.Error("an undescribed required property was accepted as absent")
	}
}

// The drivers hand a schema through JSON before it comes back to be checked,
// which turns every type union into []any of string. Required must survive it.
func TestCheckAfterAJSONRoundTrip(t *testing.T) {
	raw, err := json.Marshal(jsonschema.For[searchArgs]())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if err := jsonschema.Check(schema, decode(t, `{"query":"go","priority":"low"}`)); err != nil {
		t.Errorf("a legitimate call was rejected: %v", err)
	}
}

// A tool that declares no parameters constrains nothing.
func TestCheckWithoutASchemaAcceptsAnything(t *testing.T) {
	for _, input := range []string{`{}`, `{"anything":1}`, `"a string"`, `null`} {
		if err := jsonschema.Check(map[string]any{}, decode(t, input)); err != nil {
			t.Errorf("Check(empty, %s) = %v, want nil", input, err)
		}
	}
}

func decode(t *testing.T, input string) any {
	t.Helper()
	var value any
	if err := json.Unmarshal([]byte(input), &value); err != nil {
		t.Fatalf("the test's own input is not JSON: %v", err)
	}
	return value
}
