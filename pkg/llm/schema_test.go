package llm_test

import (
	"context"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/llmtest"
)

type person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

var personSchema = &llm.Schema{
	Name:        "person",
	Description: "a person's name and age",
	Definition: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"required":             []string{"name", "age"},
		"additionalProperties": false,
	},
	Strict: true,
}

// A constrained answer is bare JSON, and that is the easy case.
func TestParseNativeAnswer(t *testing.T) {
	drv := llmtest.Text(`{"name":"Ada","age":36}`)
	got, err := llm.Parse[person](llmtest.Client(drv).Complete(
		context.Background(), &llm.Prompt{}, &llm.Options{Schema: personSchema}))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Name != "Ada" || got.Age != 36 {
		t.Errorf("got %+v", got)
	}
}

// An instructed answer is not, and a caller should not have to know which path
// produced the response they are holding.
func TestExtractionSurvivesWhatModelsActuallyReturn(t *testing.T) {
	tests := map[string]string{
		"bare":               `{"name":"Ada","age":36}`,
		"markdown fence":     "```json\n{\"name\":\"Ada\",\"age\":36}\n```",
		"unlabelled fence":   "```\n{\"name\":\"Ada\",\"age\":36}\n```",
		"preamble":           "Sure! Here is the JSON:\n{\"name\":\"Ada\",\"age\":36}",
		"trailing prose":     "{\"name\":\"Ada\",\"age\":36}\n\nLet me know if you need anything else.",
		"both sides":         "Here you go:\n```json\n{\"name\":\"Ada\",\"age\":36}\n```\nHope that helps!",
		"leading whitespace": "\n\n  {\"name\":\"Ada\",\"age\":36}",
	}
	for name, content := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := llm.Parse[person](llmtest.Client(llmtest.Text(content)).Complete(
				context.Background(), &llm.Prompt{}, nil))
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			if got.Name != "Ada" || got.Age != 36 {
				t.Errorf("got %+v", got)
			}
		})
	}
}

// A brace inside a quoted value must not end the scan early.
func TestExtractionRespectsStringLiterals(t *testing.T) {
	content := `Here: {"name":"Ada } Lovelace","age":36} done`
	var got person
	resp, err := llmtest.Client(llmtest.Text(content)).Complete(context.Background(), &llm.Prompt{}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if err := resp.Unmarshal(&got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Name != "Ada } Lovelace" {
		t.Errorf("Name = %q", got.Name)
	}
}

func TestExtractionReportsWhenThereIsNoJSON(t *testing.T) {
	resp, _ := llmtest.Client(llmtest.Text("I would rather not.")).Complete(
		context.Background(), &llm.Prompt{}, nil)
	var got person
	err := resp.Unmarshal(&got)
	if err == nil {
		t.Fatal("expected an error")
	}
	// The message has to show what came back, or the caller cannot tell a
	// refusal from a parse bug.
	if !strings.Contains(err.Error(), "rather not") {
		t.Errorf("err = %v, want it to quote the answer", err)
	}
}

func TestParseThreadsTheCallError(t *testing.T) {
	_, err := llm.Parse[person](llmtest.Client(llmtest.Fail(
		llm.Classify("test", 500, nil, "", "boom", nil))).Complete(
		context.Background(), &llm.Prompt{}, nil))
	if err == nil || !llm.IsRetryable(err) {
		t.Fatalf("err = %v, want the call's error threaded through", err)
	}
}

// Asking a model that cannot constrain output is refused, with the remedy in
// the message — not answered with a weaker guarantee the caller never asked for.
func TestSchemaOnAnIncapableModelIsRefused(t *testing.T) {
	model := llmtest.Model
	model.Unsupported.Schema = true

	drv := llmtest.Text("ok")
	_, err := llm.New(drv, model).Complete(context.Background(), &llm.Prompt{},
		&llm.Options{Schema: personSchema})
	if !llm.IsUnsupported(err) {
		t.Fatalf("err = %v, want it refused", err)
	}
	if !strings.Contains(err.Error(), "SimulateSchema") {
		t.Errorf("err = %v, want it to name the remedy", err)
	}
	if drv.CallCount() != 0 {
		t.Error("a request was sent for a shape the model cannot produce")
	}
}

// Some endpoints can constrain output or offer tools, but not both at once.
func TestSchemaWithToolsIsRefusedWhereUnsupported(t *testing.T) {
	model := llmtest.Model
	model.Unsupported.SchemaWithTools = true
	prompt := &llm.Prompt{Tools: []llm.Tool{{Name: "ls"}}}

	client := llm.New(llmtest.Text("ok"), model)
	if _, err := client.Complete(context.Background(), prompt,
		&llm.Options{Schema: personSchema}); !llm.IsUnsupported(err) {
		t.Fatalf("err = %v, want it refused", err)
	}
	// Either one alone is fine.
	if _, err := client.Complete(context.Background(), prompt, nil); err != nil {
		t.Errorf("tools alone: %v", err)
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{},
		&llm.Options{Schema: personSchema}); err != nil {
		t.Errorf("schema alone: %v", err)
	}
}

// The fallback asks in words. It is opt-in because instructing is not
// constraining — the model can still return prose.
func TestSimulateSchemaAsksInTheSystemPrompt(t *testing.T) {
	model := llmtest.Model
	model.Unsupported.Schema = true

	drv := llmtest.Text(`{"name":"Ada","age":36}`)
	client := llm.New(drv, model, llm.WithMiddleware(llm.SimulateSchema()))

	got, err := llm.Parse[person](client.Complete(context.Background(),
		&llm.Prompt{System: "be brief", Messages: []llm.Message{llm.User("who?")}},
		&llm.Options{Schema: personSchema}))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Name != "Ada" {
		t.Errorf("got %+v", got)
	}

	sent := drv.Last()
	if !strings.Contains(sent.Prompt.System, "be brief") {
		t.Error("the original system prompt was dropped")
	}
	if !strings.Contains(sent.Prompt.System, "JSON Schema") {
		t.Errorf("the schema was not put to the model: %q", sent.Prompt.System)
	}
	if !strings.Contains(sent.Prompt.System, `"age"`) {
		t.Error("the schema definition itself was not included")
	}
	// The driver must be told it is not enforcing, or it would send a
	// constraint the endpoint cannot honour.
	if sent.Options.Schema != nil {
		t.Error("the schema was still passed to the driver as a constraint")
	}
}

func TestSimulateSchemaIsInertWithoutOne(t *testing.T) {
	drv := llmtest.Text("plain answer")
	client := llm.New(drv, llmtest.Model, llm.WithMiddleware(llm.SimulateSchema()))
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{System: "be brief"}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if drv.Last().Prompt.System != "be brief" {
		t.Errorf("System = %q, want it untouched", drv.Last().Prompt.System)
	}
}
