package openaichat_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/openaichat"
)

// recorder is a stand-in Chat Completions endpoint: it captures the request
// body and replays a scripted SSE stream.
type recorder struct {
	server  *httptest.Server
	body    map[string]any
	path    string
	headers http.Header
}

func newRecorder(t *testing.T, events ...string) *recorder {
	t.Helper()
	r := &recorder{}
	r.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		raw, _ := io.ReadAll(req.Body)
		r.path = req.URL.Path
		r.headers = req.Header.Clone()
		if err := json.Unmarshal(raw, &r.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			fmt.Fprintf(w, "data: %s\n\n", e)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(r.server.Close)
	return r
}

func (r *recorder) extra(key string) any { return r.body[key] }

// The three ladder shapes the Chat Completions dialects take: a level string,
// a boolean switch, and a token budget.
var (
	effortLadder = []llm.ReasoningLevel{
		{Effort: llm.EffortOff},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium"},
		{Effort: llm.EffortHigh, Value: "high", Default: true},
	}
	switchLadder = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortHigh, Value: "enabled"},
	}
	budgetLadder = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Budget: 5_000},
		{Effort: llm.EffortMedium, Budget: 32_000},
		{Effort: llm.EffortHigh, Budget: 128_000},
	}
)

func openChat(t *testing.T, url string, model llm.Model) *llm.Client {
	t.Helper()
	model.API = llm.APIOpenAIChat
	client, err := llm.Open(llm.Config{Model: model, APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return client
}

const (
	textEvent = `{"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"Hi"}}]}`
	doneEvent = `{"id":"1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":40}}}`
)

func TestStreamAggregatesAndSplitsCachedTokens(t *testing.T) {
	rec := newRecorder(t, textEvent, doneEvent)
	client := openChat(t, rec.server.URL, llm.Model{ID: "m"})

	resp, err := client.Complete(context.Background(), &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
	}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "Hi" {
		t.Errorf("Content = %q", resp.Content)
	}
	if resp.StopReason != llm.StopEndTurn {
		t.Errorf("StopReason = %q", resp.StopReason)
	}
	// prompt_tokens is the whole prompt; the cached slice has to come out of
	// Input so a turn's sums do not count the re-read cache twice.
	want := llm.Usage{Input: 60, Output: 5, CacheRead: 40}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
	if resp.Usage.TotalInput() != 100 {
		t.Errorf("TotalInput = %d, want the API's 100", resp.Usage.TotalInput())
	}
}

func TestSystemPromptBecomesFirstMessage(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	client := openChat(t, rec.server.URL, llm.Model{ID: "m"})
	if _, err := client.Complete(context.Background(), &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	msgs, _ := rec.extra("messages").([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v", msgs)
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be brief" {
		t.Errorf("first message = %v", first)
	}
}

func TestToolCallsAssembleAcrossChunks(t *testing.T) {
	rec := newRecorder(t,
		`{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"ls","arguments":"{\"path\""}}]}}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"/tmp\"}"}}]}}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	)
	client := openChat(t, rec.server.URL, llm.Model{ID: "m"})

	resp, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %+v", resp.ToolCalls)
	}
	call := resp.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "ls" || call.Input != `{"path":"/tmp"}` {
		t.Errorf("call = %+v", call)
	}
	if resp.StopReason != llm.StopToolUse {
		t.Errorf("StopReason = %q", resp.StopReason)
	}
}

// Each vendor spells the same idea differently; the dialect is catalog data,
// so this is the table that proves the data reaches the wire.
func TestReasoningDialects(t *testing.T) {
	tests := []struct {
		name    string
		dialect llm.ThinkingFormat
		effort  llm.Effort
		ladder  []llm.ReasoningLevel
		check   func(t *testing.T, rec *recorder)
	}{
		{
			name: "no dialect sends nothing", dialect: llm.ThinkingNone, effort: llm.EffortHigh, ladder: effortLadder,
			check: func(t *testing.T, rec *recorder) {
				if rec.extra("reasoning_effort") != nil || rec.extra("thinking") != nil {
					t.Errorf("unexpected reasoning fields: %v", rec.body)
				}
			},
		},
		{
			name: "effort passes the level through", dialect: llm.ThinkingEffort, effort: llm.EffortHigh, ladder: effortLadder,
			check: func(t *testing.T, rec *recorder) {
				if got := rec.extra("reasoning_effort"); got != "high" {
					t.Errorf("reasoning_effort = %v", got)
				}
			},
		},
		{
			name: "effort omits the field when off", dialect: llm.ThinkingEffort, effort: llm.EffortOff, ladder: effortLadder,
			check: func(t *testing.T, rec *recorder) {
				if got := rec.extra("reasoning_effort"); got != nil {
					t.Errorf("reasoning_effort = %v, want absent", got)
				}
			},
		},
		{
			// DeepSeek reasons by default, so "off" has to be said out loud.
			name: "effort_or_disable switches off explicitly", dialect: llm.ThinkingEffortOrDisable, effort: llm.EffortOff, ladder: effortLadder,
			check: func(t *testing.T, rec *recorder) {
				thinking, _ := rec.extra("thinking").(map[string]any)
				if thinking["type"] != "disabled" {
					t.Errorf("thinking = %v, want type=disabled", rec.extra("thinking"))
				}
			},
		},
		{
			name: "thinking_type enables without a level", dialect: llm.ThinkingType, effort: llm.EffortHigh, ladder: switchLadder,
			check: func(t *testing.T, rec *recorder) {
				thinking, _ := rec.extra("thinking").(map[string]any)
				if thinking["type"] != "enabled" {
					t.Errorf("thinking = %v", rec.extra("thinking"))
				}
				if rec.extra("reasoning_effort") != nil {
					t.Error("thinking_type endpoints take no level")
				}
			},
		},
		{
			// OpenRouter normalizes every upstream onto this shape.
			name: "reasoning_object nests the level", dialect: llm.ThinkingReasoningObject, effort: llm.EffortHigh, ladder: effortLadder,
			check: func(t *testing.T, rec *recorder) {
				reasoning, _ := rec.extra("reasoning").(map[string]any)
				if reasoning["effort"] != "high" {
					t.Errorf("reasoning = %v, want effort=high", rec.extra("reasoning"))
				}
				if rec.extra("reasoning_effort") != nil {
					t.Error("the flat field was sent alongside the nested one")
				}
			},
		},
		{
			name: "enable_thinking carries a budget", dialect: llm.ThinkingEnableFlag, effort: llm.EffortMedium, ladder: budgetLadder,
			check: func(t *testing.T, rec *recorder) {
				if rec.extra("enable_thinking") != true {
					t.Errorf("enable_thinking = %v", rec.extra("enable_thinking"))
				}
				if got := rec.extra("thinking_budget"); got != float64(32_000) {
					t.Errorf("thinking_budget = %v", got)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := newRecorder(t, doneEvent)
			model := llm.Model{
				ID:        "m",
				Compat:    llm.OpenAIChatCompat{Thinking: tc.dialect},
				Reasoning: tc.ladder,
			}
			client := openChat(t, rec.server.URL, model)

			if _, err := client.Complete(context.Background(), &llm.Prompt{
				Messages: []llm.Message{llm.User("hi")},
			}, &llm.Options{Effort: tc.effort}); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			tc.check(t, rec)
		})
	}
}

func TestReasoningContentAlwaysPresent(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	model := llm.Model{ID: "m", Compat: llm.OpenAIChatCompat{ReasoningContent: true}}
	client := openChat(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{
		llm.User("hi"),
		{Role: llm.RoleAssistant, Content: llm.Text("prior"), Thinking: "because"},
		llm.User("go on"),
	}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	msgs, _ := rec.body["messages"].([]any)
	var assistant map[string]any
	for _, m := range msgs {
		if entry, _ := m.(map[string]any); entry["role"] == "assistant" {
			assistant = entry
		}
	}
	if assistant == nil {
		t.Fatalf("no assistant message: %v", msgs)
	}
	if got, ok := assistant["reasoning_content"]; !ok || got != "because" {
		t.Errorf("reasoning_content = %v (present=%v)", got, ok)
	}
}

// Sending an image to a text-only endpoint used to drop it silently, so the
// model answered about a picture it had never seen. It is now refused before
// the request is made.
func TestTextOnlyModelRefusesImages(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	client := openChat(t, rec.server.URL, llm.Model{ID: "m"})

	img := llm.Image{MediaType: "image/png", Data: "AAAA"}
	_, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("look", img)},
	}, nil)
	if err == nil {
		t.Fatal("expected the request to be refused")
	}
	if !llm.IsUnsupported(err) {
		t.Errorf("err = %v, want an unsupported-capability failure", err)
	}
	if !strings.Contains(err.Error(), "image") {
		t.Errorf("err = %v, want it to name the problem", err)
	}
	// Nothing was sent, so nothing was spent.
	if rec.body != nil {
		t.Error("a request reached the endpoint")
	}
}

func TestVisionModelSendsInterleavedParts(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	client := openChat(t, rec.server.URL, llm.Model{ID: "m", Input: []llm.Modality{llm.ModalityText, llm.ModalityImage}})

	img := llm.Image{MediaType: "image/png", Data: "AAAA"}
	content := llm.Content{llm.TextPart("before"), llm.ImagePart(img), llm.TextPart("after")}
	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.UserContent(content)},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	msgs, _ := rec.body["messages"].([]any)
	user, _ := msgs[0].(map[string]any)
	parts, ok := user["content"].([]any)
	if !ok || len(parts) != 3 {
		t.Fatalf("content = %v", user["content"])
	}
	kinds := make([]string, 3)
	for i, p := range parts {
		entry, _ := p.(map[string]any)
		kinds[i], _ = entry["type"].(string)
	}
	// Order is the whole point of Content being a sequence.
	if kinds[0] != "text" || kinds[1] != "image_url" || kinds[2] != "text" {
		t.Errorf("part order = %v", kinds)
	}
}

func TestAuthFailureIsClassified(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"Incorrect API key provided","code":"invalid_api_key"}}`)
	}))
	defer server.Close()

	client := openChat(t, server.URL, llm.Model{ID: "m"})
	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !llm.IsAuth(err) {
		t.Errorf("err = %v, want an auth failure", err)
	}
	if llm.IsRetryable(err) {
		t.Error("a bad key is not worth retrying")
	}
}

func TestOverloadIsRetryable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"error":{"message":"overloaded"}}`)
	}))
	defer server.Close()

	client := openChat(t, server.URL, llm.Model{ID: "m"})
	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if !llm.IsRetryable(err) {
		t.Errorf("err = %v, want retryable", err)
	}
}

func TestToolChoice(t *testing.T) {
	tests := map[llm.ToolChoice]any{
		llm.ToolChoiceAuto:     nil,
		llm.ToolChoiceNone:     "none",
		llm.ToolChoiceRequired: "required",
	}
	for choice, want := range tests {
		rec := newRecorder(t, doneEvent)
		client := openChat(t, rec.server.URL, llm.Model{ID: "m"})

		prompt := &llm.Prompt{
			Messages: []llm.Message{llm.User("hi")},
			Tools:    []llm.Tool{{Name: "ls", Parameters: map[string]any{"type": "object"}}},
		}
		if _, err := client.Complete(context.Background(), prompt, &llm.Options{ToolChoice: choice}); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if got := rec.extra("tool_choice"); got != want {
			t.Errorf("ToolChoice %q sent tool_choice = %v, want %v", choice, got, want)
		}
	}
}

// A server that predates max_completion_tokens ignores the newer name, which
// silently uncaps the response rather than erroring.
func TestMaxTokensFieldCompat(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	model := llm.Model{ID: "m", Compat: llm.OpenAIChatCompat{MaxTokensField: "max_tokens"}}
	client := openChat(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}},
		&llm.Options{MaxTokens: 256}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := rec.extra("max_tokens"); got != float64(256) {
		t.Errorf("max_tokens = %v", got)
	}
	if rec.extra("max_completion_tokens") != nil {
		t.Error("both output-cap fields were sent")
	}
}

// Sampling parameters reach the wire verbatim, which is what lets a caller
// drive a llama.cpp or vLLM server through parameters this SDK does not model.
func TestSamplingParamsReachTheWire(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	model := llm.Model{ID: "m", SamplingParams: map[string]any{"top_k": 40, "min_p": 0.05}}
	client := openChat(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}},
		&llm.Options{SamplingParams: map[string]any{"min_p": 0.1}}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := rec.extra("top_k"); got != float64(40) {
		t.Errorf("top_k = %v, want the model's 40", got)
	}
	if got := rec.extra("min_p"); got != 0.1 {
		t.Errorf("min_p = %v, want the caller's 0.1", got)
	}
}

func TestNoUsageInStreamCompat(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	model := llm.Model{ID: "m", Compat: llm.OpenAIChatCompat{NoUsageInStream: true}}
	client := openChat(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if rec.extra("stream_options") != nil {
		t.Errorf("stream_options = %v, want it omitted", rec.extra("stream_options"))
	}
}

// Per-model headers are sent, and a Config header of the same name wins.
func TestModelHeaders(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	model := llm.Model{ID: "m", API: llm.APIOpenAIChat, Headers: map[string]string{
		"X-Tenant": "acme",
		"X-Both":   "from-model",
	}}
	client, err := llm.Open(llm.Config{
		Model: model, APIKey: "k", BaseURL: rec.server.URL,
		Headers: map[string]string{"X-Both": "from-config"},
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := rec.headers.Get("X-Tenant"); got != "acme" {
		t.Errorf("X-Tenant = %q", got)
	}
	if got := rec.headers.Get("X-Both"); got != "from-config" {
		t.Errorf("X-Both = %q, want the Config value to win", got)
	}
}

func TestSchemaBecomesResponseFormat(t *testing.T) {
	rec := newRecorder(t, doneEvent)
	client := openChat(t, rec.server.URL, llm.Model{ID: "m"})

	schema := &llm.Schema{
		Name: "person", Description: "a person", Strict: true,
		Definition: map[string]any{"type": "object"},
	}
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("who?")}},
		&llm.Options{Schema: schema}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	format, _ := rec.body["response_format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Fatalf("response_format = %v", rec.body["response_format"])
	}
	js, _ := format["json_schema"].(map[string]any)
	if js["name"] != "person" || js["description"] != "a person" || js["strict"] != true {
		t.Errorf("json_schema = %v", js)
	}
	if js["schema"] == nil {
		t.Error("the schema itself was not sent")
	}
}
