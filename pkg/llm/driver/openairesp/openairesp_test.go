package openairesp_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/openairesp"
)

type recorder struct {
	server *httptest.Server
	body   map[string]any
}

func newRecorder(t *testing.T, events ...string) *recorder {
	t.Helper()
	r := &recorder{}
	r.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		raw, _ := io.ReadAll(req.Body)
		if err := json.Unmarshal(raw, &r.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			fmt.Fprintf(w, "data: %s\n\n", e)
		}
	}))
	t.Cleanup(r.server.Close)
	return r
}

// effortLadder mirrors OpenAI's reasoning.effort values, "none" included:
// the protocol accepts it, so "off" is a real rung here.
var effortLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff, Value: "none"},
	{Effort: llm.EffortLow, Value: "low"},
	{Effort: llm.EffortMedium, Value: "medium", Default: true},
	{Effort: llm.EffortHigh, Value: "high"},
}

func open(t *testing.T, url string, model llm.Model) *llm.Client {
	t.Helper()
	model.API = llm.APIOpenAIResponses
	client, err := llm.Open(llm.Config{Model: model, APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return client
}

const (
	textDelta = `{"type":"response.output_text.delta","delta":"Hi"}`
	completed = `{"type":"response.completed","response":{"id":"r1","model":"gpt-test","status":"completed",` +
		`"output":[],"usage":{"input_tokens":100,"output_tokens":5,"input_tokens_details":{"cached_tokens":25}}}}`
)

func TestStreamSplitsCachedTokens(t *testing.T) {
	rec := newRecorder(t, textDelta, completed)
	client := open(t, rec.server.URL, llm.Model{ID: "gpt-test"})

	resp, err := client.Complete(context.Background(), &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
	}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "Hi" || resp.StopReason != llm.StopEndTurn {
		t.Errorf("resp = %+v", resp)
	}
	want := llm.Usage{Input: 75, Output: 5, CacheRead: 25}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
	// The system prompt is an instructions field here, not a message.
	if rec.body["instructions"] != "be brief" {
		t.Errorf("instructions = %v", rec.body["instructions"])
	}
}

func TestReasoningEffortIsSentAsALevel(t *testing.T) {
	rec := newRecorder(t, completed)
	model := llm.Model{ID: "gpt-test", Reasoning: effortLadder}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortHigh}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	reasoning, _ := rec.body["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" {
		t.Errorf("reasoning = %v", rec.body["reasoning"])
	}
}

func TestStatelessBackendEchoesEncryptedReasoning(t *testing.T) {
	rec := newRecorder(t, completed)
	model := llm.Model{ID: "gpt-test", Compat: llm.OpenAIResponsesCompat{Stateless: true}}
	client := open(t, rec.server.URL, model)

	msgs := []llm.Message{
		llm.User("go"),
		{
			Role:      llm.RoleAssistant,
			Reasoning: []llm.ReasoningItem{{ID: "rs_1", EncryptedContent: "enc"}},
			ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "ls", Input: "{}"}},
		},
		llm.ToolResultsMessage(llm.ToolResult{ToolCallID: "call_1", Content: "ok"}),
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: msgs}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if rec.body["store"] != false {
		t.Errorf("store = %v, want false", rec.body["store"])
	}
	input, _ := rec.body["input"].([]any)
	var kinds []string
	for _, item := range input {
		entry, _ := item.(map[string]any)
		kind, _ := entry["type"].(string)
		kinds = append(kinds, kind)
	}
	// The reasoning item must precede the function call it belongs to, or the
	// stateless backend rejects the request.
	var reasoningAt, callAt = -1, -1
	for i, k := range kinds {
		if k == "reasoning" && reasoningAt < 0 {
			reasoningAt = i
		}
		if k == "function_call" && callAt < 0 {
			callAt = i
		}
	}
	if reasoningAt < 0 || callAt < 0 || reasoningAt > callAt {
		t.Errorf("input item order = %v", kinds)
	}
}

func TestInBandFailureIsClassified(t *testing.T) {
	// A Responses failure arrives inside a 200, so there is no status to
	// classify from — only the error code says whether a retry could work.
	rec := newRecorder(t, `{"type":"response.completed","response":{"id":"r1","model":"gpt-test",`+
		`"status":"failed","output":[],"error":{"code":"server_error","message":"upstream exploded"},`+
		`"usage":{"input_tokens":1,"output_tokens":0}}}`)
	client := open(t, rec.server.URL, llm.Model{ID: "gpt-test"})

	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !llm.IsRetryable(err) {
		t.Errorf("err = %v, want retryable", err)
	}
}

func TestInBandContextOverflowIsNotRetried(t *testing.T) {
	rec := newRecorder(t, `{"type":"error","code":"invalid_request_error",`+
		`"message":"This model's maximum context length is 400000 tokens"}`)
	client := open(t, rec.server.URL, llm.Model{ID: "gpt-test"})

	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if !llm.IsContextExceeded(err) {
		t.Fatalf("err = %v, want a context-exceeded failure", err)
	}
	if llm.IsRetryable(err) {
		t.Error("resending the same oversized prompt cannot help")
	}
}

// The reasoning models accept reasoning.effort "none", so EffortOff is a real
// setting here rather than something the catalog has to hide.
func TestEffortOffSendsNone(t *testing.T) {
	rec := newRecorder(t, completed)
	model := llm.Model{ID: "gpt-test", Reasoning: effortLadder}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortOff}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	reasoning, _ := rec.body["reasoning"].(map[string]any)
	if reasoning["effort"] != "none" {
		t.Errorf("reasoning = %v, want effort=none", rec.body["reasoning"])
	}
}

// An unset effort leaves the parameter off entirely, so the provider's own
// default stands.
func TestUnsetEffortSendsNoReasoningParam(t *testing.T) {
	rec := newRecorder(t, completed)
	client := open(t, rec.server.URL, llm.Model{ID: "gpt-test"})

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if _, ok := rec.body["reasoning"]; ok {
		t.Errorf("reasoning was sent for a model with no ladder: %v", rec.body["reasoning"])
	}
}

func TestSchemaBecomesTextFormat(t *testing.T) {
	rec := newRecorder(t, completed)
	client := open(t, rec.server.URL, llm.Model{ID: "gpt-test"})

	schema := &llm.Schema{Name: "person", Strict: true, Definition: map[string]any{"type": "object"}}
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("who?")}},
		&llm.Options{Schema: schema}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	text, _ := rec.body["text"].(map[string]any)
	format, _ := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "person" || format["strict"] != true {
		t.Errorf("text.format = %v", text["format"])
	}
	if format["schema"] == nil {
		t.Error("the schema itself was not sent")
	}
}
