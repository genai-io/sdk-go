package jsonschema_test

import (
	"encoding/json"
	"fmt"
	"math/big"
	"net/netip"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

// The types under test are the test data, so they sit next to the cases that
// describe them rather than inside a helper.

type basics struct {
	Name    string  `json:"name"`
	Age     int     `json:"age"`
	Score   float64 `json:"score"`
	Active  bool    `json:"active"`
	Dropped string  `json:"-"`
	hidden  string  //nolint:unused // present to prove encoding/json never writes it
}

// optionals covers every way a field is allowed to be absent, including the
// tag that names nothing.
type optionals struct {
	Limit  int     `json:"limit,omitempty"`
	Cursor *string `json:"cursor"`
	Note   string  `json:",omitempty"`
	Draft  string  `json:"draft,omitzero"`
}

// pointers pins what a pointer derives to. It is nullable once, from
// schemaForType, whether or not the field is also tagged optional.
type pointers struct {
	Plain    *int       `json:"plain"`
	Optional *int       `json:"optional,omitempty"`
	At       *time.Time `json:"at"`
	List     []*string  `json:"list"`
}

type embeddedInner struct {
	Tag string `json:"tag"`
}

type named struct {
	Label string `json:"label"`
}

type embedding struct {
	embeddedInner        // promoted, the way encoding/json promotes it
	Named         named  `json:"named"`
	Own           string `json:"own"`
}

type marshalsAsSomethingElse struct {
	At    time.Time     `json:"at"`
	Every time.Duration `json:"every"`
	Blob  []byte        `json:"blob"`
	Addr  netip.Addr    `json:"addr"`
}

type enums struct {
	Priority string  `json:"priority" enum:"low|medium|high"`
	Limit    int     `json:"limit" enum:"1|2|3"`
	Ratio    float64 `json:"ratio" enum:"0.5|1.5"`
	Verbose  bool    `json:"verbose" enum:"true|false"`
}

type maps struct {
	Labels map[string]string `json:"labels"`
	Counts map[string]int    `json:"counts,omitempty"`
}

func TestForDerives(t *testing.T) {
	object := func(properties map[string]any, required ...any) map[string]any {
		if required == nil {
			required = []any{}
		}
		return map[string]any{
			"type":                 "object",
			"properties":           properties,
			"required":             required,
			"additionalProperties": false,
		}
	}

	for name, tc := range map[string]struct {
		got  map[string]any
		want map[string]any
	}{
		"the four scalar kinds, minus what json drops": {
			jsonschema.For[basics](),
			object(map[string]any{
				"name":   map[string]any{"type": "string"},
				"age":    map[string]any{"type": "integer"},
				"score":  map[string]any{"type": "number"},
				"active": map[string]any{"type": "boolean"},
			}, "name", "age", "score", "active"),
		},
		"an optional field is required and nullable": {
			jsonschema.For[optionals](),
			object(map[string]any{
				"limit":  map[string]any{"type": []any{"integer", "null"}},
				"cursor": map[string]any{"type": []any{"string", "null"}},
				// json:",omitempty" names nothing, so encoding/json uses the
				// Go field name and so does the schema.
				"Note":  map[string]any{"type": []any{"string", "null"}},
				"draft": map[string]any{"type": []any{"string", "null"}},
			}, "limit", "cursor", "Note", "draft"),
		},
		"a pointer admits null exactly once": {
			jsonschema.For[pointers](),
			object(map[string]any{
				"plain":    map[string]any{"type": []any{"integer", "null"}},
				"optional": map[string]any{"type": []any{"integer", "null"}},
				"at":       map[string]any{"type": []any{"string", "null"}, "format": "date-time"},
				"list": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": []any{"string", "null"}},
				},
			}, "plain", "optional", "at", "list"),
		},
		"an empty struct still has a required list": {
			jsonschema.For[struct{}](),
			object(map[string]any{}),
		},
		"an embedded struct lands in the parent": {
			jsonschema.For[embedding](),
			object(map[string]any{
				"tag":   map[string]any{"type": "string"},
				"named": object(map[string]any{"label": map[string]any{"type": "string"}}, "label"),
				"own":   map[string]any{"type": "string"},
			}, "tag", "named", "own"),
		},
		"a type is described by what it marshals to": {
			jsonschema.For[marshalsAsSomethingElse](),
			object(map[string]any{
				"at":    map[string]any{"type": "string", "format": "date-time"},
				"every": map[string]any{"type": "integer"},
				"blob":  map[string]any{"type": "string"},
				// netip.Addr marshals through MarshalText, which json quotes.
				"addr": map[string]any{"type": "string"},
			}, "at", "every", "blob", "addr"),
		},
		"enum members are typed by the field they constrain": {
			jsonschema.For[enums](),
			object(map[string]any{
				"priority": map[string]any{"type": "string", "enum": []any{"low", "medium", "high"}},
				"limit":    map[string]any{"type": "integer", "enum": []any{int64(1), int64(2), int64(3)}},
				"ratio":    map[string]any{"type": "number", "enum": []any{0.5, 1.5}},
				"verbose":  map[string]any{"type": "boolean", "enum": []any{true, false}},
			}, "priority", "limit", "ratio", "verbose"),
		},
		"a map is an open object of one value type": {
			jsonschema.For[maps](),
			object(map[string]any{
				"labels": map[string]any{
					"type":                 "object",
					"additionalProperties": map[string]any{"type": "string"},
				},
				"counts": map[string]any{
					"type":                 []any{"object", "null"},
					"additionalProperties": map[string]any{"type": "integer"},
				},
			}, "labels", "counts"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if !reflect.DeepEqual(tc.got, tc.want) {
				t.Errorf("schema =\n%s\nwant\n%s", pretty(t, tc.got), pretty(t, tc.want))
			}
		})
	}
}

// required is a list on the wire, never null: the OpenAI and Google drivers
// forward the derived map verbatim.
func TestRequiredMarshalsAsAList(t *testing.T) {
	raw, err := json.Marshal(jsonschema.For[struct{}]())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	const want = `{"additionalProperties":false,"properties":{},"required":[],"type":"object"}`
	if string(raw) != want {
		t.Errorf("schema = %s\nwant   %s", raw, want)
	}
}

// A tag mistake is the caller's own source, so it panics at construction
// rather than producing a schema nobody can satisfy.
func TestForPanics(t *testing.T) {
	for name, tc := range map[string]struct {
		derive func()
		want   string
	}{
		"a map the model cannot key": {
			func() { _ = jsonschema.For[struct{ M map[int]string }]() }, "keyed by int",
		},
		"an interface": {
			func() { _ = jsonschema.For[struct{ Anything any }]() }, "describes nothing",
		},
		"a channel": {
			func() { _ = jsonschema.For[struct{ C chan int }]() }, "no JSON representation",
		},
		"a type that writes its own JSON": {
			func() { _ = jsonschema.For[struct{ Raw json.RawMessage }]() }, "MarshalJSON",
		},
		"a big.Int, whose fields are not its JSON": {
			func() { _ = jsonschema.For[struct{ N big.Int }]() }, "MarshalJSON",
		},
		"a pointer to one": {
			func() { _ = jsonschema.For[struct{ N *big.Int }]() }, "MarshalJSON",
		},
		"a recursive type": {
			func() { _ = jsonschema.For[recursive]() }, "refers to itself",
		},
		"an enum a boolean cannot hold": {
			func() { _ = jsonschema.For[unbooleanEnum]() }, "not a boolean",
		},
		"an enum a number cannot hold": {
			func() { _ = jsonschema.For[unnumericEnum]() }, "not an integer",
		},
		"a bound that is not a number": {
			func() { _ = jsonschema.For[unnumericBound]() }, "not a number",
		},
		"an empty tag": {
			func() { _ = jsonschema.For[emptyTag]() }, "empty description tag",
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("this was accepted; a mistake in the caller's source must not be")
				}
				if msg := fmt.Sprint(r); !strings.Contains(msg, tc.want) {
					t.Errorf("panic = %q\nwant it to mention %q", msg, tc.want)
				}
			}()
			tc.derive()
		})
	}
}

type recursive struct {
	Next *recursive `json:"next"`
}

type unbooleanEnum struct {
	Verbose bool `json:"verbose" enum:"yes|no"`
}

type unnumericEnum struct {
	Limit int `json:"limit" enum:"one|two"`
}

type unnumericBound struct {
	Limit int `json:"limit" minimum:"lots"`
}

type emptyTag struct {
	Query string `json:"query" description:""`
}

// order is the golden struct: one field per supported keyword, so an edit to
// the deriver shows up here as a diff rather than as a provider's complaint.
type order struct {
	Item     string   `json:"item" description:"what to order, one line" minLength:"1" maxLength:"40" pattern:"^[a-z ]+$"`
	Priority string   `json:"priority" enum:"low|medium|high"`
	Deliver  string   `json:"deliver" format:"date-time"`
	Quantity int      `json:"quantity" description:"how many" minimum:"3" maximum:"99" multipleOf:"3"`
	Tags     []string `json:"tags,omitempty" minItems:"1" maxItems:"5"`
}

func TestGoldenSchema(t *testing.T) {
	const want = `{
  "additionalProperties": false,
  "properties": {
    "deliver": {
      "format": "date-time",
      "type": "string"
    },
    "item": {
      "description": "what to order, one line",
      "maxLength": 40,
      "minLength": 1,
      "pattern": "^[a-z ]+$",
      "type": "string"
    },
    "priority": {
      "enum": [
        "low",
        "medium",
        "high"
      ],
      "type": "string"
    },
    "quantity": {
      "description": "how many",
      "maximum": 99,
      "minimum": 3,
      "multipleOf": 3,
      "type": "integer"
    },
    "tags": {
      "items": {
        "type": "string"
      },
      "maxItems": 5,
      "minItems": 1,
      "type": [
        "array",
        "null"
      ]
    }
  },
  "required": [
    "item",
    "priority",
    "deliver",
    "quantity",
    "tags"
  ],
  "type": "object"
}`
	if got := pretty(t, jsonschema.For[order]()); got != want {
		t.Errorf("schema =\n%s\nwant\n%s", got, want)
	}
}

// ForType is the same derivation from a type only known at run time.
func TestForTypeMatchesFor(t *testing.T) {
	if got, want := jsonschema.ForType(reflect.TypeFor[order]()), jsonschema.For[order](); !reflect.DeepEqual(got, want) {
		t.Errorf("ForType =\n%s\nwant\n%s", pretty(t, got), pretty(t, want))
	}
}

func pretty(t *testing.T, schema map[string]any) string {
	t.Helper()
	raw, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(raw)
}

func TestEmbeddedFieldSelectionMatchesJSON(t *testing.T) {
	type Embedded struct {
		X int `json:"x"`
	}
	type First struct {
		X string `json:"x"`
		Embedded
	}
	type Last struct {
		Embedded
		X string `json:"x"`
	}
	type Untagged struct{ X int }
	type Tagged struct {
		Value string `json:"X"`
	}
	type TaggedWins struct {
		Untagged
		Tagged
	}
	type Conflict struct {
		Embedded
		Other struct {
			X int `json:"x"`
		} `json:"other"`
	}
	type AlsoEmbedded struct {
		X string `json:"x"`
	}
	// Build the intentional tag conflict at runtime; vet rightly rejects it in source.
	ambiguous := reflect.New(reflect.StructOf([]reflect.StructField{
		{Name: "Embedded", Type: reflect.TypeFor[Embedded](), Anonymous: true},
		{Name: "AlsoEmbedded", Type: reflect.TypeFor[AlsoEmbedded](), Anonymous: true},
	})).Elem().Interface()
	type Common struct{ X int }
	type Left struct{ Common }
	type Right struct{ Common }
	type Diamond struct {
		Left
		Right
	}
	for _, tc := range []struct {
		name     string
		value    any
		property string
	}{
		{"outer first", First{}, "x"},
		{"outer last", Last{}, "x"},
		{"tagged wins at the same depth", TaggedWins{}, "X"},
		{"named fields stay nested", Conflict{}, ""},
		{"ambiguous fields disappear", ambiguous, ""},
		{"diamond embeddings are ambiguous", Diamond{}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			schema := jsonschema.ForType(reflect.TypeOf(tc.value))
			properties := schema["properties"].(map[string]any)
			raw, err := json.Marshal(tc.value)
			if err != nil {
				t.Fatal(err)
			}
			var encoded map[string]any
			if err := json.Unmarshal(raw, &encoded); err != nil {
				t.Fatal(err)
			}
			if len(properties) != len(encoded) {
				t.Fatalf("properties=%v, encoding/json=%s", properties, raw)
			}
			for name := range encoded {
				if _, exists := properties[name]; !exists {
					t.Errorf("schema lost JSON property %q", name)
				}
			}
			required := schema["required"].([]any)
			if len(required) != len(properties) {
				t.Errorf("required=%v, properties=%v", required, properties)
			}
			if tc.property != "" && properties[tc.property].(map[string]any)["type"] != "string" {
				t.Errorf("selected the wrong field for %s: %v", tc.property, properties[tc.property])
			}
		})
	}
}

func TestStringJSONTagsRequireACustomSchema(t *testing.T) {
	defer func() {
		if value := recover(); value == nil || !strings.Contains(fmt.Sprint(value), "custom schema") {
			t.Errorf("panic=%v, want an explicit unsupported string-tag error", value)
		}
	}()
	_ = jsonschema.For[struct {
		ID int64 `json:"id,string"`
	}]()
}

func TestIgnoredStringJSONTagsMatchJSON(t *testing.T) {
	type Nested struct {
		Count int `json:"count"`
	}
	for _, value := range []any{[]string{"a"}, [1]string{"b"}, map[string]string{"key": "value"}, Nested{1}, &Nested{2}} {
		// Construct the ignored tags at runtime; staticcheck rejects them in source.
		typ := reflect.StructOf([]reflect.StructField{{Name: "Value", Type: reflect.TypeOf(value), Tag: `json:"value,string"`}})
		field := reflect.New(typ).Elem()
		field.Field(0).Set(reflect.ValueOf(value))
		raw, err := json.Marshal(field.Interface())
		if err != nil {
			t.Fatal(err)
		}
		var encoded any
		if err := json.Unmarshal(raw, &encoded); err != nil {
			t.Fatal(err)
		}
		if err := jsonschema.Check(jsonschema.ForType(typ), encoded); err != nil {
			t.Errorf("schema rejected encoding/json output %s: %v", raw, err)
		}
	}
}

func TestNullableEnumsIncludeNullAfterFieldTags(t *testing.T) {
	schema := jsonschema.For[struct {
		Mode    string  `json:"mode,omitempty" enum:"on|off"`
		Pointer *string `json:"pointer" enum:"on|off"`
	}]()
	for _, name := range []string{"mode", "pointer"} {
		property := schema["properties"].(map[string]any)[name].(map[string]any)
		if !slices.Contains(property["enum"].([]any), any(nil)) {
			t.Errorf("%s's nullable enum excludes null: %v", name, property)
		}
	}
	if err := jsonschema.Check(schema, map[string]any{"mode": nil, "pointer": nil}); err != nil {
		t.Errorf("nullable enum rejected null: %v", err)
	}
	if err := jsonschema.Check(map[string]any{"type": []any{"string", "null"}, "enum": []any{"on", "off"}}, nil); err == nil {
		t.Error("null bypassed the enum constraint")
	}
}
