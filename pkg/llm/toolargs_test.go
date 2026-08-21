package llm_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
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

// A hand-written map and the struct the arguments decode into are two
// descriptions of one shape, and nothing keeps them in step. Deriving one from
// the other removes the second description.
func TestValidateArgsCatchesWhatModelsGetWrong(t *testing.T) {
	tool := llm.ToolFor[searchArgs]("search", "search the web")

	tests := map[string]struct {
		input string
		want  string
	}{
		"missing a required field": {`{"deep":true}`, "query is required"},
		"wrong type":               {`{"query":"x","deep":true,"limit":"ten"}`, "limit must be an integer"},
		"fractional integer":       {`{"query":"x","deep":true,"limit":1.5}`, "limit must be an integer"},
		"invented property":        {`{"query":"x","deep":true,"colour":"red"}`, "colour is not a known property"},
		"outside its enum":         {`{"query":"x","deep":true,"sort":"price"}`, "sort must be one of"},
		"wrong array element":      {`{"query":"x","deep":true,"tags":[1]}`, "tags[0] must be a string"},
		"not JSON at all":          {`{"query":`, "not valid JSON"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			err := tool.ValidateArgs(tc.input)
			if err == nil {
				t.Fatalf("accepted %s", tc.input)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want it to mention %q", err, tc.want)
			}
			// The message has to name the tool, or a turn with six calls is
			// unreadable.
			if !strings.Contains(err.Error(), "search") {
				t.Errorf("err = %v, want it to name the tool", err)
			}
		})
	}

	if err := tool.ValidateArgs(`{"query":"x","deep":true,"limit":10,"sort":"date","tags":["a"]}`); err != nil {
		t.Errorf("a valid call was rejected: %v", err)
	}
}

func TestValidateArgsWithoutASchema(t *testing.T) {
	// Nothing to check against is not a failure.
	bare := llm.Tool{Name: "anything"}
	if err := bare.ValidateArgs(`{"whatever":true}`); err != nil {
		t.Errorf("a tool with no schema should accept anything: %v", err)
	}
	// An empty argument string is the empty object every protocol sends for a
	// no-argument call.
	type none struct{}
	if err := llm.ToolFor[none]("noop", "").ValidateArgs(""); err != nil {
		t.Errorf("an empty argument string should be accepted: %v", err)
	}
}

func TestUnmarshalArgsRejectsInventedFields(t *testing.T) {
	call := llm.ToolCall{Name: "search", Input: `{"query":"go","limit":5}`}
	got, err := llm.UnmarshalArgs[searchArgs](call)
	if err != nil {
		t.Fatalf("UnmarshalArgs: %v", err)
	}
	if got.Query != "go" || got.Limit != 5 {
		t.Errorf("got %+v", got)
	}

	invented := llm.ToolCall{Name: "search", Input: `{"query":"go","colour":"red"}`}
	if _, err := llm.UnmarshalArgs[searchArgs](invented); err == nil {
		t.Error("an invented argument should be surfaced, not dropped")
	}
}

// A schema decoded from JSON has []any where a Go-built one has []string, and
// both have to validate.
func TestValidateAgainstADecodedSchema(t *testing.T) {
	built := jsonschema.Of[searchArgs]()
	raw, _ := json.Marshal(built)
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	var value any
	_ = json.Unmarshal([]byte(`{"deep":true}`), &value)
	if err := jsonschema.Validate(decoded, value); err == nil {
		t.Error("a round-tripped schema stopped enforcing required fields")
	}
}
