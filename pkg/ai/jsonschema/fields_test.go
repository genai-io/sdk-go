package jsonschema_test

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"
)

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
