package google_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/google"
)

type recorder struct {
	server *httptest.Server
	body   map[string]any
}

func newRecorder(t *testing.T, chunks ...string) *recorder {
	t.Helper()
	r := &recorder{}
	r.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		raw, _ := io.ReadAll(req.Body)
		if err := json.Unmarshal(raw, &r.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\r\n\r\n", c)
		}
	}))
	t.Cleanup(r.server.Close)
	return r
}

// levelLadder is Gemini 3's thinkingLevel; budgetLadder is Gemini 2.5's
// thinkingBudget. Same rungs, different field.
var levelLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff, Default: true},
	{Effort: llm.EffortLow, Value: "LOW"},
	{Effort: llm.EffortMedium, Value: "MEDIUM"},
	{Effort: llm.EffortHigh, Value: "HIGH"},
}

var budgetLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff, Default: true},
	{Effort: llm.EffortLow, Budget: 5_000},
	{Effort: llm.EffortMedium, Budget: 32_000},
	{Effort: llm.EffortHigh, Budget: 128_000},
}

func open(t *testing.T, url string, model llm.Model) *llm.Client {
	t.Helper()
	model.API = llm.APIGoogleGenAI
	client, err := llm.Open(llm.Config{Model: model, APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return client
}

const geminiDone = `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]},"finishReason":"STOP"}],` +
	`"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"cachedContentTokenCount":30}}`

func TestStreamSplitsCachedTokens(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test"})

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
	// Gemini reports the cached prefix inside the prompt count, like the
	// OpenAI protocols do.
	want := llm.Usage{Input: 70, Output: 5, CacheRead: 30}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
}

func TestSystemPromptBecomesSystemInstruction(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test"})
	if _, err := client.Complete(context.Background(), &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if rec.body["systemInstruction"] == nil {
		t.Errorf("systemInstruction missing: %v", rec.body)
	}
}

func TestThoughtsAreSeparatedFromTheAnswer(t *testing.T) {
	rec := newRecorder(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"considering","thought":true},{"text":"Answer"}]}}]}`,
		`{"candidates":[{"finishReason":"STOP"}]}`,
	)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test"})

	resp, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Thinking != "considering" || resp.Content != "Answer" {
		t.Errorf("thinking=%q content=%q", resp.Thinking, resp.Content)
	}
}

func TestFunctionCallCarriesThoughtSignature(t *testing.T) {
	rec := newRecorder(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c1","name":"ls","args":{"path":"/tmp"}},`+
			`"thoughtSignature":"c2ln"}]},"finishReason":"STOP"}]}`,
	)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test"})

	resp, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %+v", resp.ToolCalls)
	}
	call := resp.ToolCalls[0]
	if call.Name != "ls" || call.Input != `{"path":"/tmp"}` {
		t.Errorf("call = %+v", call)
	}
	// Gemini rejects the next turn if the signature does not come back with
	// the call.
	if len(call.Signature) == 0 {
		t.Error("thought signature was dropped")
	}
}

func TestNonJSONToolResultIsWrapped(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test"})

	msgs := []llm.Message{
		llm.User("go"),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "ls", Input: "{}"}}},
		llm.ToolResultsMessage(llm.ToolResult{ToolCallID: "c1", ToolName: "ls", Content: "plain text"}),
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: msgs}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	raw, _ := json.Marshal(rec.body)
	var body struct {
		Contents []struct {
			Parts []struct {
				FunctionResponse *struct {
					Response map[string]any `json:"response"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	fr := body.Contents[2].Parts[0].FunctionResponse
	if fr == nil || fr.Response["result"] != "plain text" {
		// The protocol demands an object, so a plain-text result has to be
		// wrapped rather than rejected.
		t.Errorf("functionResponse = %+v", fr)
	}
}

func TestCorruptImageIsReportedBeforeSending(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test", Input: []llm.Modality{llm.ModalityText, llm.ModalityImage}})

	bad := llm.Image{MediaType: "image/png", Data: "not base64!!", FileName: "shot.png"}
	_, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("look", bad)},
	}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !llm.IsKind(err, llm.KindInvalidRequest) {
		t.Errorf("err = %v, want an invalid-request failure naming the file", err)
	}
}

// Gemini 3 replaced the thinking budget with a level; sending a budget to it
// is the wrong control entirely.
func TestGemini3SendsThinkingLevel(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	model := llm.Model{
		ID:        "gemini-test",
		Compat:    llm.GoogleCompat{ThinkingLevel: true},
		Reasoning: levelLadder,
	}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortMedium}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	cfg, _ := rec.body["generationConfig"].(map[string]any)
	thinking, _ := cfg["thinkingConfig"].(map[string]any)
	if thinking["thinkingLevel"] != "MEDIUM" {
		t.Errorf("thinkingConfig = %v, want thinkingLevel=MEDIUM", cfg["thinkingConfig"])
	}
	if _, ok := thinking["thinkingBudget"]; ok {
		t.Errorf("a token budget was sent to a level-based model: %v", thinking)
	}
}

// Gemini 2.5 still takes a budget.
func TestGemini25SendsThinkingBudget(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	model := llm.Model{
		ID:        "gemini-test",
		Compat:    llm.GoogleCompat{},
		Reasoning: budgetLadder,
	}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortMedium}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	cfg, _ := rec.body["generationConfig"].(map[string]any)
	thinking, _ := cfg["thinkingConfig"].(map[string]any)
	if thinking["thinkingBudget"] == nil {
		t.Errorf("thinkingConfig = %v, want a thinkingBudget", cfg["thinkingConfig"])
	}
}

// The JSON mime type alone would only promise valid JSON; the schema is what
// promises the shape, so both have to be sent.
func TestSchemaBecomesResponseSchema(t *testing.T) {
	rec := newRecorder(t, geminiDone)
	client := open(t, rec.server.URL, llm.Model{ID: "gemini-test"})

	schema := &llm.Schema{Name: "person", Definition: map[string]any{"type": "object"}}
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("who?")}},
		&llm.Options{Schema: schema}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	cfg, _ := rec.body["generationConfig"].(map[string]any)
	if cfg["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v", cfg["responseMimeType"])
	}
	if cfg["responseJsonSchema"] == nil {
		t.Error("the schema itself was not sent")
	}
}

// The paths below were not exercised before: the SDK owned them. Rewriting
// them by hand means they need covering by hand.

func TestCountTokensSendsTheWholeRequest(t *testing.T) {
	var path string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		path = req.URL.Path
		raw, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(raw, &body)
		fmt.Fprint(w, `{"totalTokens":1234}`)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "gemini-test"})
	count, err := client.CountTokens(context.Background(), &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
		Tools:    []llm.Tool{{Name: "ls", Parameters: map[string]any{"type": "object"}}},
	}, nil)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if count.Tokens != 1234 || !count.Exact {
		t.Errorf("count = %+v, want an exact 1234", count)
	}
	if !strings.HasSuffix(path, "/v1beta/models/gemini-test:countTokens") {
		t.Errorf("path = %q", path)
	}
	// The system instruction and the tools count against the window too, and
	// the wrapper is the only request form that accepts them.
	inner, ok := body["generateContentRequest"].(map[string]any)
	if !ok {
		t.Fatalf("body = %v, want a generateContentRequest wrapper", body)
	}
	if inner["model"] != "models/gemini-test" {
		t.Errorf("model = %v, want the qualified name the wrapper requires", inner["model"])
	}
	if inner["systemInstruction"] == nil {
		t.Error("the system instruction was left out of the count")
	}
	if inner["tools"] == nil {
		t.Error("the tool declarations were left out of the count")
	}
}

func TestModelsListPaginatesAndFilters(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Query().Get("pageToken") == "" {
			fmt.Fprint(w, `{"models":[
				{"name":"models/gemini-3-pro","displayName":"Gemini 3 Pro","inputTokenLimit":1048576,"outputTokenLimit":65536},
				{"name":"models/gemini-2.5-flash-exp","displayName":"Experimental"},
				{"name":"models/embedding-001","displayName":"Embedding"}
			],"nextPageToken":"next"}`)
			return
		}
		fmt.Fprint(w, `{"models":[
			{"name":"models/gemini-2.5-flash","displayName":"Gemini 2.5 Flash","inputTokenLimit":1048576},
			{"name":"models/gemini-pro-latest","displayName":"Latest alias"}
		]}`)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "gemini-test"})
	models, err := client.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}

	var ids []string
	for _, m := range models {
		ids = append(ids, m.ID)
	}
	// A second page has to be fetched, and the aliases dropped: "-exp" and
	// "-latest" name a model whose meaning changes without notice, and
	// non-Gemini models do not speak this protocol.
	want := []string{"gemini-2.5-flash", "gemini-3-pro"}
	if !slices.Equal(ids, want) {
		t.Errorf("ids = %v, want %v", ids, want)
	}
	if models[1].ContextWindow != 1048576 || models[1].MaxOutput != 65536 {
		t.Errorf("limits = %d/%d", models[1].ContextWindow, models[1].MaxOutput)
	}
	if models[1].Name != "Gemini 3 Pro" {
		t.Errorf("Name = %q", models[1].Name)
	}
}

func TestErrorCarriesStatusAndRetryAfter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota exceeded"}}`)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "gemini-test"})
	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !llm.IsRetryable(err) {
		t.Errorf("err = %v, want a 429 reported as retryable", err)
	}
	if !strings.Contains(err.Error(), "quota exceeded") {
		t.Errorf("err = %v, want the provider's message", err)
	}
	// Keeping the response is what makes this available; the SDK's error type
	// dropped it, so a 429's own backoff was never honoured.
	if d := llm.RetryAfter(err); d != 7*time.Second {
		t.Errorf("RetryAfter = %v, want 7s", d)
	}
}

func TestAPIKeyTravelsInAHeaderNotTheURL(t *testing.T) {
	var header, rawQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		header, rawQuery = req.Header.Get("x-goog-api-key"), req.URL.RawQuery
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\r\n\r\n", geminiDone)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "gemini-test"})
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if header != "k" {
		t.Errorf("x-goog-api-key = %q", header)
	}
	// A URL ends up in proxy logs and error reports; a credential should not
	// go with it.
	if strings.Contains(rawQuery, "k") && strings.Contains(rawQuery, "key=") {
		t.Errorf("query = %q, want no credential in the URL", rawQuery)
	}
	if !strings.Contains(rawQuery, "alt=sse") {
		t.Errorf("query = %q, want alt=sse to request server-sent events", rawQuery)
	}
}
