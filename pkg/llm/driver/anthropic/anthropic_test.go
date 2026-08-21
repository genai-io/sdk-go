package anthropic_test

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
	"github.com/genai-io/sdk-go/pkg/llm/driver/anthropic"
)

type sseEvent struct{ name, data string }

// recorder is a stand-in Messages endpoint.
type recorder struct {
	server  *httptest.Server
	body    map[string]any
	headers http.Header
}

func newRecorder(t *testing.T, events ...sseEvent) *recorder {
	t.Helper()
	r := &recorder{}
	r.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		raw, _ := io.ReadAll(req.Body)
		r.headers = req.Header.Clone()
		if err := json.Unmarshal(raw, &r.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.name, e.data)
		}
	}))
	t.Cleanup(r.server.Close)
	return r
}

var (
	startEvent = sseEvent{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test",` +
		`"usage":{"input_tokens":20,"cache_creation_input_tokens":7,"cache_read_input_tokens":13}}}`}
	textEvent = sseEvent{"content_block_delta",
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`}
	endEvent = sseEvent{"message_delta",
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`}
	stopEvent = sseEvent{"message_stop", `{"type":"message_stop"}`}
)

// budgetLadder is the pre-4.6 shape: a token budget per rung.
var budgetLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff, Default: true},
	{Effort: llm.EffortLow, Budget: 5_000},
	{Effort: llm.EffortMedium, Budget: 32_000},
	{Effort: llm.EffortHigh, Budget: 128_000},
}

// adaptiveLadder is the 4.6-and-later shape: an output_config.effort literal.
var adaptiveLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff},
	{Effort: llm.EffortLow, Value: "low"},
	{Effort: llm.EffortMedium, Value: "medium"},
	{Effort: llm.EffortHigh, Value: "high", Default: true},
}

func open(t *testing.T, url string, model llm.Model) *llm.Client {
	t.Helper()
	model.API = llm.APIAnthropicMessages
	client, err := llm.Open(llm.Config{Model: model, APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return client
}

func TestStreamAggregatesUsage(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

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
	// Input arrives at message_start and output at message_delta; a
	// whole-struct replace would lose the half that came first.
	want := llm.Usage{Input: 20, Output: 4, CacheWrite: 7, CacheRead: 13}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
	if resp.Model != "claude-test" {
		t.Errorf("Model = %q", resp.Model)
	}
}

func TestSystemPromptIsCacheable(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})
	if _, err := client.Complete(context.Background(), &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	system, _ := rec.body["system"].([]any)
	if len(system) != 1 {
		t.Fatalf("system = %v", rec.body["system"])
	}
	block, _ := system[0].(map[string]any)
	cache, _ := block["cache_control"].(map[string]any)
	if cache["type"] != "ephemeral" {
		t.Errorf("cache_control = %v; the breakpoint is what makes the cache counts mean tools+system", block)
	}
}

func TestToolResultsCollapseIntoOneUserTurn(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

	msgs := []llm.Message{
		llm.User("do two things"),
		{Role: llm.RoleAssistant, Content: llm.Text("on it"), ToolCalls: []llm.ToolCall{
			{ID: "toolu_1", Name: "a", Input: `{}`},
			{ID: "toolu_2", Name: "b", Input: `{}`},
		}},
		llm.ToolResultsMessage(
			llm.ToolResult{ToolCallID: "toolu_1", Content: "one"},
			llm.ToolResult{ToolCallID: "toolu_2", Content: "two"},
		),
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: msgs}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	wire, _ := rec.body["messages"].([]any)
	if len(wire) != 3 {
		t.Fatalf("messages = %d, want 3: %v", len(wire), wire)
	}
	last, _ := wire[2].(map[string]any)
	blocks, _ := last["content"].([]any)
	// The API requires every tool_result answering one assistant turn to
	// arrive in a single user message.
	if last["role"] != "user" || len(blocks) != 2 {
		t.Errorf("final turn = %v", last)
	}
}

func TestForeignToolIDsAreRewritten(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

	// An ID another provider produced, carrying characters Claude rejects.
	const foreign = "call:with/bad chars"
	msgs := []llm.Message{
		llm.User("hi"),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: foreign, Name: "a", Input: "{}"}}},
		llm.ToolResultsMessage(llm.ToolResult{ToolCallID: foreign, Content: "ok"}),
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: msgs}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	raw, _ := json.Marshal(rec.body)
	if strings.Contains(string(raw), foreign) {
		t.Errorf("the invalid ID reached the wire: %s", raw)
	}
	// The rewrite has to be stable, or the call and its result stop matching.
	var body struct {
		Messages []struct {
			Content []struct {
				ID        string `json:"id"`
				ToolUseID string `json:"tool_use_id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	useID := body.Messages[1].Content[0].ID
	resultID := body.Messages[2].Content[0].ToolUseID
	if useID == "" || useID != resultID {
		t.Errorf("tool_use %q and tool_result %q no longer match", useID, resultID)
	}
}

func TestThinkingBudgetRaisesMaxTokens(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{
		ID:        "claude-test",
		MaxOutput: 1024,
		Compat:    llm.AnthropicCompat{},
		Reasoning: budgetLadder,
	}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortHigh}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	thinking, _ := rec.body["thinking"].(map[string]any)
	budget, _ := thinking["budget_tokens"].(float64)
	maxTokens, _ := rec.body["max_tokens"].(float64)
	if thinking["type"] != "enabled" || budget == 0 {
		t.Fatalf("thinking = %v", rec.body["thinking"])
	}
	// budget_tokens must leave room for an answer, so the model's own 1024
	// output cap cannot stand.
	if maxTokens <= budget {
		t.Errorf("max_tokens = %v, must exceed budget %v", maxTokens, budget)
	}
}

func TestExtendedContextSuffixBecomesBetaHeader(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test[1m]", MaxOutput: 1024})

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	// The suffix is ours, not Anthropic's: it must not reach the model field.
	if rec.body["model"] != "claude-test" {
		t.Errorf("model = %v, want the suffix stripped", rec.body["model"])
	}
	if got := rec.headers.Get("Anthropic-Beta"); got == "" {
		t.Error("the 1M-context beta header was not sent")
	}
}

func TestBearerAuthStyle(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{ID: "claude-test", MaxOutput: 1024,
		Compat: llm.AnthropicCompat{BearerAuth: true}}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := rec.headers.Get("Authorization"); got != "Bearer k" {
		t.Errorf("Authorization = %q", got)
	}
	if rec.headers.Get("X-Api-Key") != "" {
		t.Error("x-api-key should not be sent in bearer mode")
	}
}

func TestOverloadedIsRetryable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(529) // Anthropic's "overloaded"
		fmt.Fprint(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})
	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if !llm.IsRetryable(err) {
		t.Errorf("err = %v, want retryable", err)
	}
}

func TestPromptTooLongIsContextExceeded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000"}}`)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})
	_, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if !llm.IsContextExceeded(err) {
		t.Errorf("err = %v, want a context-exceeded failure so the caller compacts", err)
	}
}

// Claude 4.6 and later take adaptive thinking with the level in
// output_config.effort. On Opus 5, Opus 4.7/4.8, Sonnet 5 and Fable 5 a
// budget_tokens request is rejected outright, so sending one is not merely
// dated — it is a 400.
func TestAdaptiveThinkingSendsEffortNotBudget(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{
		ID: "claude-test", MaxOutput: 1024,
		Compat:    llm.AnthropicCompat{ForceAdaptiveThinking: true},
		Reasoning: adaptiveLadder,
	}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortHigh}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	thinking, _ := rec.body["thinking"].(map[string]any)
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking = %v, want type=adaptive", rec.body["thinking"])
	}
	if _, ok := thinking["budget_tokens"]; ok {
		t.Errorf("budget_tokens was sent on an adaptive model: %v", thinking)
	}
	outputConfig, _ := rec.body["output_config"].(map[string]any)
	if outputConfig["effort"] != "high" {
		t.Errorf("output_config = %v, want effort=high", rec.body["output_config"])
	}
	// Nothing raised max_tokens: there is no budget to leave room for.
	if got, _ := rec.body["max_tokens"].(float64); got != 1024 {
		t.Errorf("max_tokens = %v, want the model's 1024", got)
	}
}

// These models think by default, so switching it off has to be explicit.
func TestAdaptiveThinkingOffIsExplicit(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{
		ID: "claude-test", MaxOutput: 1024,
		Compat:    llm.AnthropicCompat{ForceAdaptiveThinking: true},
		Reasoning: adaptiveLadder,
	}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
	}, &llm.Options{Effort: llm.EffortOff}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	thinking, _ := rec.body["thinking"].(map[string]any)
	if thinking["type"] != "disabled" {
		t.Errorf("thinking = %v, want type=disabled", rec.body["thinking"])
	}
	if _, ok := rec.body["output_config"]; ok {
		// An effort must not be paired with disabled thinking — xhigh/max
		// pairings are a 400, and there is no level to ask for anyway.
		t.Errorf("output_config was sent alongside disabled thinking: %v", rec.body["output_config"])
	}
}

// A thinking block is replayed only into a request that is itself thinking.
// The adaptive path has no budget to gate that on, so it must gate on whether
// reasoning is on at all.
func TestThinkingIsReplayedUnderAdaptive(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{
		ID: "claude-test", MaxOutput: 1024,
		Compat:    llm.AnthropicCompat{ForceAdaptiveThinking: true},
		Reasoning: adaptiveLadder,
	}
	client := open(t, rec.server.URL, model)

	msgs := []llm.Message{
		llm.User("hi"),
		{Role: llm.RoleAssistant, Content: llm.Text("prior"), Thinking: "because", ThinkingSignature: "sig"},
		llm.User("go on"),
	}
	if _, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: msgs,
	}, &llm.Options{Effort: llm.EffortHigh}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	raw, _ := json.Marshal(rec.body)
	if !strings.Contains(string(raw), `"thinking"`) || !strings.Contains(string(raw), "sig") {
		t.Errorf("the signed thinking block was not replayed: %s", raw)
	}
}

// Claude Opus 4.7 and later reject a non-default temperature outright, so the
// field has to be droppable per model rather than always sent.
func TestNoTemperatureCompat(t *testing.T) {
	for _, tc := range []struct {
		name   string
		compat llm.AnthropicCompat
		want   bool
	}{
		{"default sends it", llm.AnthropicCompat{}, true},
		{"NoTemperature drops it", llm.AnthropicCompat{NoTemperature: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
			client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024, Compat: tc.compat})

			if _, err := client.Complete(context.Background(),
				&llm.Prompt{Messages: []llm.Message{llm.User("hi")}},
				&llm.Options{Temperature: 0.7}); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			_, sent := rec.body["temperature"]
			if sent != tc.want {
				t.Errorf("temperature sent = %v, want %v", sent, tc.want)
			}
		})
	}
}

func TestToolChoice(t *testing.T) {
	tests := map[llm.ToolChoice]string{
		llm.ToolChoiceAuto:     "",
		llm.ToolChoiceNone:     "none",
		llm.ToolChoiceRequired: "any",
	}
	for choice, want := range tests {
		rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
		client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

		prompt := &llm.Prompt{
			Messages: []llm.Message{llm.User("hi")},
			Tools:    []llm.Tool{{Name: "ls", Parameters: map[string]any{"type": "object"}}},
		}
		if _, err := client.Complete(context.Background(), prompt, &llm.Options{ToolChoice: choice}); err != nil {
			t.Fatalf("Complete: %v", err)
		}

		sent, _ := rec.body["tool_choice"].(map[string]any)
		got, _ := sent["type"].(string)
		if got != want {
			t.Errorf("ToolChoice %q sent tool_choice type %q, want %q", choice, got, want)
		}
	}
}

// "Call this specific tool" is a constraint ToolChoice cannot express, and
// every supported protocol has it.
func TestForceTool(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

	prompt := &llm.Prompt{
		Messages: []llm.Message{llm.User("hi")},
		Tools:    []llm.Tool{{Name: "ls", Parameters: map[string]any{"type": "object"}}},
	}
	if _, err := client.Complete(context.Background(), prompt, &llm.Options{ForceTool: "ls"}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	choice, _ := rec.body["tool_choice"].(map[string]any)
	if choice["type"] != "tool" || choice["name"] != "ls" {
		t.Errorf("tool_choice = %v, want the named tool", rec.body["tool_choice"])
	}
}

// The native escape hatch reaches settings the normalized options deliberately
// do not model, so needing one of them does not mean writing a driver.
func TestNativeOptions(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{
		ID: "claude-test", MaxOutput: 1024,
		Compat:    llm.AnthropicCompat{ForceAdaptiveThinking: true},
		Reasoning: adaptiveLadder,
	}
	client := open(t, rec.server.URL, model)

	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("hi")}},
		&llm.Options{
			Effort: llm.EffortHigh,
			Native: anthropic.Native{ThinkingDisplay: "omitted", Betas: []string{"my-beta-2026-01-01"}},
		}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	thinking, _ := rec.body["thinking"].(map[string]any)
	if thinking["display"] != "omitted" {
		t.Errorf("thinking = %v, want display=omitted", rec.body["thinking"])
	}
	if got := rec.headers.Get("Anthropic-Beta"); !strings.Contains(got, "my-beta-2026-01-01") {
		t.Errorf("Anthropic-Beta = %q, want the caller's beta", got)
	}
}

// A Native meant for another protocol is ignored rather than failing the
// request, so the same Options stay usable when the model is swapped.
func TestNativeForAnotherProtocolIsIgnored(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("hi")}},
		&llm.Options{Native: struct{ Unrelated string }{"x"}}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if rec.headers.Get("Anthropic-Beta") != "" {
		t.Error("an unrelated Native leaked into the request")
	}
}

// A 1-hour cache write costs twice the input rate against 1.25x for five
// minutes, so it is asked for deliberately rather than applied by default.
func TestCacheRetention(t *testing.T) {
	tests := []struct {
		name      string
		retention llm.CacheRetention
		compat    llm.AnthropicCompat
		wantCache bool
		wantTTL   any
	}{
		{"default caches short", llm.CacheDefault, llm.AnthropicCompat{}, true, nil},
		{"short caches short", llm.CacheShort, llm.AnthropicCompat{}, true, nil},
		{"long sets the 1h ttl", llm.CacheLong, llm.AnthropicCompat{}, true, "1h"},
		{"none skips the breakpoint", llm.CacheNone, llm.AnthropicCompat{}, false, nil},
		{
			"long falls back where unsupported", llm.CacheLong,
			llm.AnthropicCompat{NoLongCacheRetention: true}, true, nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
			client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024, Compat: tc.compat})

			if _, err := client.Complete(context.Background(),
				&llm.Prompt{System: "be brief", Messages: []llm.Message{llm.User("hi")}},
				&llm.Options{CacheRetention: tc.retention}); err != nil {
				t.Fatalf("Complete: %v", err)
			}

			system, _ := rec.body["system"].([]any)
			block, _ := system[0].(map[string]any)
			cache, cached := block["cache_control"].(map[string]any)
			if cached != tc.wantCache {
				t.Fatalf("cache_control present = %v, want %v", cached, tc.wantCache)
			}
			if tc.wantCache && cache["ttl"] != tc.wantTTL {
				t.Errorf("ttl = %v, want %v", cache["ttl"], tc.wantTTL)
			}
		})
	}
}

// The 1-hour slice is billed differently from the rest of the cache write, so
// it has to travel separately rather than folded into the total.
func TestLongCacheWriteIsReportedSeparately(t *testing.T) {
	start := sseEvent{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test",` +
		`"usage":{"input_tokens":10,"cache_creation_input_tokens":1000,` +
		`"cache_creation":{"ephemeral_1h_input_tokens":600,"ephemeral_5m_input_tokens":400},` +
		`"cache_read_input_tokens":0}}}`}
	rec := newRecorder(t, start, textEvent, endEvent, stopEvent)
	client := open(t, rec.server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})

	resp, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Usage.CacheWrite != 1000 {
		t.Errorf("CacheWrite = %d", resp.Usage.CacheWrite)
	}
	if resp.Usage.CacheWrite1h != 600 {
		t.Errorf("CacheWrite1h = %d, want the 1h slice", resp.Usage.CacheWrite1h)
	}
	if resp.ID != "m1" {
		t.Errorf("ID = %q, want the provider's message id", resp.ID)
	}
}

// Anthropic publishes a counting endpoint, so a caller never has to estimate —
// and never has to spend a generation request to find out the prompt was too
// large.
func TestNativeTokenCounting(t *testing.T) {
	var gotPath string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"input_tokens":4321}`)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024, ContextWindow: 200_000})
	prompt := &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("hello")},
		Tools:    []llm.Tool{{Name: "ls", Parameters: map[string]any{"type": "object"}}},
	}

	count, err := client.CountTokens(context.Background(), prompt, nil)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if !count.Exact {
		t.Error("a provider count should be reported as exact")
	}
	if count.Tokens != 4321 {
		t.Errorf("Tokens = %d", count.Tokens)
	}
	if !strings.HasSuffix(gotPath, "/count_tokens") {
		t.Errorf("path = %q", gotPath)
	}
	// The system prompt and the tools count against the window too, so both
	// have to reach the counting endpoint.
	if body["system"] == nil {
		t.Error("the system prompt was not counted")
	}
	if body["tools"] == nil {
		t.Error("the tool definitions were not counted")
	}

	left, _, err := client.Headroom(context.Background(), prompt, nil)
	if err != nil {
		t.Fatalf("Headroom: %v", err)
	}
	if left != 200_000-4321 {
		t.Errorf("headroom = %d", left)
	}
}

// A counting endpoint that is down should not stop a caller from sizing the
// prompt: the estimate is worse than an exact count, not useless.
func TestCountingFallsBackToTheEstimate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	client := open(t, server.URL, llm.Model{ID: "claude-test", MaxOutput: 1024})
	count, err := client.CountTokens(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User(strings.Repeat("word ", 100))}}, nil)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if count.Exact {
		t.Error("a failed count must not be reported as exact")
	}
	if count.Tokens == 0 {
		t.Error("no estimate was produced")
	}
}

// Structured output shares output_config with the effort level, so setting one
// must not wipe the other.
func TestSchemaAndEffortShareOutputConfig(t *testing.T) {
	rec := newRecorder(t, startEvent, textEvent, endEvent, stopEvent)
	model := llm.Model{
		ID: "claude-test", MaxOutput: 1024,
		Compat:    llm.AnthropicCompat{ForceAdaptiveThinking: true},
		Reasoning: adaptiveLadder,
	}
	client := open(t, rec.server.URL, model)

	schema := &llm.Schema{Name: "person", Definition: map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
	}}
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("who?")}},
		&llm.Options{Effort: llm.EffortHigh, Schema: schema}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	cfg, _ := rec.body["output_config"].(map[string]any)
	if cfg["effort"] != "high" {
		t.Errorf("effort = %v, want it kept alongside the format", cfg["effort"])
	}
	format, _ := cfg["format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Errorf("format = %v", cfg["format"])
	}
	if format["schema"] == nil {
		t.Error("the schema itself was not sent")
	}
}
