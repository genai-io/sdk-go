package jsonschema_test

import (
	"encoding/json"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm/jsonschema"
)

type searchArgs struct {
	Query   string   `json:"query" jsonschema:"description=what to look for"`
	Limit   int      `json:"limit,omitempty"`
	Sort    string   `json:"sort,omitempty" jsonschema:"enum=relevance|date"`
	Tags    []string `json:"tags,omitempty"`
	Deep    bool     `json:"deep,omitempty" jsonschema:"required"`
	private string   //nolint:unused // exercises the unexported skip
	Skipped string   `json:"-"`
}

func TestJSONSchemaOfStruct(t *testing.T) {
	schema := jsonschema.Of[searchArgs]()

	if schema["type"] != "object" {
		t.Fatalf("type = %v", schema["type"])
	}
	if schema["additionalProperties"] != false {
		t.Error("unknown properties should be rejected")
	}

	props, _ := schema["properties"].(map[string]any)
	for _, name := range []string{"query", "limit", "sort", "tags", "deep"} {
		if _, ok := props[name]; !ok {
			t.Errorf("property %q missing", name)
		}
	}
	if _, ok := props["private"]; ok {
		t.Error("an unexported field leaked into the schema")
	}
	if _, ok := props["Skipped"]; ok {
		t.Error("a json:\"-\" field leaked into the schema")
	}

	// Required is everything without omitempty, plus anything tagged required.
	required := map[string]bool{}
	for _, r := range schema["required"].([]string) {
		required[r] = true
	}
	if !required["query"] {
		t.Error("query should be required")
	}
	if required["limit"] {
		t.Error("an omitempty field should be optional")
	}
	if !required["deep"] {
		t.Error("jsonschema:\"required\" should override omitempty")
	}

	query, _ := props["query"].(map[string]any)
	if query["type"] != "string" || query["description"] != "what to look for" {
		t.Errorf("query = %v", query)
	}
	sort, _ := props["sort"].(map[string]any)
	if enum, _ := sort["enum"].([]any); len(enum) != 2 || enum[0] != "relevance" {
		t.Errorf("sort enum = %v", sort["enum"])
	}
	tags, _ := props["tags"].(map[string]any)
	items, _ := tags["items"].(map[string]any)
	if tags["type"] != "array" || items["type"] != "string" {
		t.Errorf("tags = %v", tags)
	}

	// It has to survive the trip to a provider.
	if _, err := json.Marshal(schema); err != nil {
		t.Errorf("schema does not marshal: %v", err)
	}
}

func TestJSONSchemaOfNestedAndSpecialTypes(t *testing.T) {
	type inner struct {
		N float64 `json:"n"`
	}
	type outer struct {
		Child    inner          `json:"child"`
		Many     []inner        `json:"many"`
		Lookup   map[string]int `json:"lookup"`
		Blob     []byte         `json:"blob"`
		Pointer  *inner         `json:"pointer"`
		Anything any            `json:"anything"`
	}
	props := jsonschema.Of[outer]()["properties"].(map[string]any)

	child, _ := props["child"].(map[string]any)
	if child["type"] != "object" {
		t.Errorf("child = %v", child)
	}
	many, _ := props["many"].(map[string]any)
	if items, _ := many["items"].(map[string]any); items["type"] != "object" {
		t.Errorf("many = %v", many)
	}
	lookup, _ := props["lookup"].(map[string]any)
	if extra, _ := lookup["additionalProperties"].(map[string]any); extra["type"] != "integer" {
		t.Errorf("lookup = %v", lookup)
	}
	// []byte is base64 text on the wire, not an array of numbers.
	blob, _ := props["blob"].(map[string]any)
	if blob["type"] != "string" {
		t.Errorf("blob = %v", blob)
	}
	pointer, _ := props["pointer"].(map[string]any)
	if pointer["type"] != "object" {
		t.Errorf("pointer = %v", pointer)
	}
	// Nothing truthful can be said about an interface, so nothing is said —
	// claiming a shape it does not have is the worse failure.
	anything, _ := props["anything"].(map[string]any)
	if len(anything) != 0 {
		t.Errorf("anything = %v, want an unconstrained value", anything)
	}
}

// A self-referential type must not recurse forever.
func TestJSONSchemaOfCyclicType(t *testing.T) {
	type node struct {
		Value string  `json:"value"`
		Next  *node   `json:"next,omitempty"`
		Kids  []*node `json:"kids,omitempty"`
	}
	done := make(chan map[string]any, 1)
	go func() { done <- jsonschema.Of[node]() }()
	select {
	case schema := <-done:
		if schema["type"] != "object" {
			t.Errorf("schema = %v", schema)
		}
	default:
		// Give it a moment; a hang would block the test binary.
		schema := <-done
		if schema["type"] != "object" {
			t.Errorf("schema = %v", schema)
		}
	}
}

// Model output is wrong sometimes. Running the tool anyway turns a mistake the
// model could have corrected into whatever the tool does with nonsense.
