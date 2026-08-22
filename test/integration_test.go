// Package integration exercises the path a caller actually walks: resolve a
// model, open a client for it, send a prompt, read the stream back.
//
// It is one suite rather than a unit test per file, and it is deliberately
// black-box — it imports the SDK the way an application does and asserts on
// two things only: the bytes that reached the endpoint, and the value that
// came back. Everything it covers is therefore something a user could break.
//
// Every endpoint here is a stub HTTP server. Nothing in this suite reaches the
// network or needs a credential.
package integration_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

// stub is a fake server that records the request body it was given and
// replays a scripted response.
type stub struct {
	server *httptest.Server
	body   map[string]any
	path   string
}

// sse replays server-sent events. Anthropic names its events; the others do
// not, which is the only difference the framing has.
func sse(t *testing.T, named bool, events ...[2]string) *stub {
	t.Helper()
	e := &stub{}
	e.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		e.path = r.URL.Path
		_ = json.Unmarshal(raw, &e.body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			if named {
				fmt.Fprintf(w, "event: %s\n", ev[0])
			}
			fmt.Fprintf(w, "data: %s\n\n", ev[1])
		}
	}))
	t.Cleanup(e.server.Close)
	return e
}

// json replies once with a JSON body, for the non-streaming calls.
func jsonEndpoint(t *testing.T, status int, body string) *stub {
	t.Helper()
	e := &stub{}
	e.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		e.path = r.URL.Path
		_ = json.Unmarshal(raw, &e.body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(e.server.Close)
	return e
}

func open(t *testing.T, url string, m ai.Model) *ai.Client {
	t.Helper()
	c, err := ai.NewClient(ai.Config{Model: m, APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return c
}

func ask(t *testing.T, c *ai.Client) *ai.Response {
	t.Helper()
	resp, err := c.Complete(context.Background(),
		[]ai.Message{ai.UserMessage("hello")}, ai.WithSystem("be brief"))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	return resp
}

// One prompt, one answer, over every protocol this SDK speaks. The point is
// that the caller's code is identical across all four: only the model changes.
func TestEveryProtocolCompletesAPrompt(t *testing.T) {
	tests := map[string]struct {
		model  ai.Model
		serve  func(*testing.T) *stub
		system func(body map[string]any) bool
	}{
		"anthropic": {
			model: ai.Model{ID: "claude-test", API: ai.APIAnthropicMessages, MaxOutput: 1024},
			serve: func(t *testing.T) *stub {
				return sse(t, true,
					[2]string{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test",` +
						`"usage":{"input_tokens":20,"cache_read_input_tokens":13}}}`},
					[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,` +
						`"delta":{"type":"text_delta","text":"Hi"}}`},
					[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},` +
						`"usage":{"output_tokens":4}}`},
				)
			},
			// Anthropic keeps the system prompt out of the turns.
			system: func(b map[string]any) bool { return b["system"] != nil },
		},
		"openai chat completions": {
			model: ai.Model{ID: "gpt-test", API: ai.APIOpenAIChat},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"id":"1","model":"gpt-test","choices":[{"index":0,"delta":{"content":"Hi"}}]}`},
					[2]string{"", `{"id":"1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
						`"usage":{"prompt_tokens":100,"completion_tokens":5}}`},
					[2]string{"", "[DONE]"},
				)
			},
			// Chat Completions carries it as the first message instead.
			system: func(b map[string]any) bool {
				msgs, _ := b["messages"].([]any)
				return len(msgs) > 0 && strings.Contains(fmt.Sprint(msgs[0]), "be brief")
			},
		},
		"openai responses": {
			model: ai.Model{ID: "gpt-test", API: ai.APIOpenAIResponses},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"type":"response.output_text.delta","delta":"Hi"}`},
					[2]string{"", `{"type":"response.completed","response":{"id":"r1","model":"gpt-test",` +
						`"status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":5}}}`},
				)
			},
			system: func(b map[string]any) bool { return b["instructions"] != nil },
		},
		"google gemini": {
			model: ai.Model{ID: "gemini-test", API: ai.APIGoogleGenAI},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]},` +
						`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5}}`},
				)
			},
			system: func(b map[string]any) bool { return b["systemInstruction"] != nil },
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			e := tc.serve(t)
			resp := ask(t, open(t, e.server.URL, tc.model))

			if resp.Text() != "Hi" {
				t.Errorf("Text = %q, want the streamed answer", resp.Text())
			}
			if resp.StopReason != ai.StopEndTurn {
				t.Errorf("StopReason = %q", resp.StopReason)
			}
			if resp.Usage.Output == 0 || resp.Usage.TotalInput() == 0 {
				t.Errorf("Usage = %+v, want the reported counts", resp.Usage)
			}
			if !tc.system(e.body) {
				t.Errorf("the system prompt did not reach the wire: %+v", e.body)
			}
		})
	}
}

// A tool offered, called, answered, and replayed — the shape of every turn a
// tool-using application sends after its first.
func TestToolsRoundTrip(t *testing.T) {
	type SearchArgs struct {
		Query string `json:"query" jsonschema:"what to look for"`
	}
	tool := ai.ToolFor[SearchArgs]("search", "search the web")

	e := sse(t, true,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test","usage":{"input_tokens":10}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,` +
			`"content_block":{"type":"tool_use","id":"call_1","name":"search","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,` +
			`"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"go\"}"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`},
	)
	client := open(t, e.server.URL, ai.Model{ID: "claude-test", API: ai.APIAnthropicMessages, MaxOutput: 1024})

	resp, err := client.Complete(context.Background(),
		[]ai.Message{ai.UserMessage("find go")}, ai.WithTools(tool))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != ai.StopToolUse {
		t.Fatalf("StopReason = %q, want the turn to be waiting on a result", resp.StopReason)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "search" {
		t.Fatalf("ToolCalls = %+v", calls)
	}

	// The schema the model was shown is derived from the Go type, and the
	// arguments it produced decode back into that same type.
	wire, _ := json.Marshal(e.body["tools"])
	if !strings.Contains(string(wire), "what to look for") {
		t.Errorf("tool schema did not reach the wire: %s", wire)
	}
	if err := tool.ValidateArgs(calls[0].Input); err != nil {
		t.Fatalf("the model's own arguments failed their schema: %v", err)
	}
	args, err := ai.UnmarshalArgs[SearchArgs](calls[0])
	if err != nil || args.Query != "go" {
		t.Errorf("UnmarshalArgs = %+v, %v", args, err)
	}

	// Replaying the call with its result is what the next turn sends.
	history := []ai.Message{
		ai.UserMessage("find go"),
		resp.Message(),
		ai.ToolResultsMessage(ai.ToolResult{ToolCallID: calls[0].ID, ToolName: "search", Content: "found"}),
	}
	if _, err := client.Complete(context.Background(), history, ai.WithTools(tool)); err != nil {
		t.Fatalf("replaying the call: %v", err)
	}
	replayed, _ := json.Marshal(e.body["messages"])
	for _, want := range []string{"call_1", "found"} {
		if !strings.Contains(string(replayed), want) {
			t.Errorf("history = %s\nwant it to contain %q", replayed, want)
		}
	}
}

// A failure has to arrive classified, because the answer to "what now" differs
// completely: fix the key, compact the prompt, wait and retry, or give up.
func TestFailuresArriveClassified(t *testing.T) {
	tests := map[string]struct {
		status int
		body   string
		check  func(error) bool
		what   string
	}{
		"bad key": {
			http.StatusUnauthorized, `{"error":{"message":"invalid api key","code":"invalid_api_key"}}`,
			ai.IsAuth, "an auth failure",
		},
		"endpoint down": {
			http.StatusServiceUnavailable, `{"error":{"message":"overloaded"}}`,
			ai.IsRetryable, "worth retrying",
		},
		"prompt too long": {
			http.StatusBadRequest, `{"error":{"message":"prompt is too long: 300000 tokens > 200000"}}`,
			ai.IsContextExceeded, "a context overflow",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			e := jsonEndpoint(t, tc.status, tc.body)
			client := open(t, e.server.URL, ai.Model{ID: "gpt-test", API: ai.APIOpenAIChat})
			_, err := client.Complete(context.Background(), []ai.Message{ai.UserMessage("hi")})
			if err == nil {
				t.Fatal("expected a failure")
			}
			if !tc.check(err) {
				t.Errorf("err = %v, want %s", err, tc.what)
			}
		})
	}
}

// A request the model cannot serve is refused before the network, naming the
// model — nothing is spent finding out.
func TestUnsupportedRequestsFailBeforeTheNetwork(t *testing.T) {
	reached := false
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { reached = true }))
	defer server.Close()

	textOnly := ai.Model{ID: "text-only", API: ai.APIOpenAIChat, Input: []ai.Modality{ai.ModalityText}}
	client := open(t, server.URL, textOnly)

	_, err := client.Complete(context.Background(),
		[]ai.Message{ai.UserMessage("what is this?", ai.Image{MediaType: "image/png", Data: "AAAA"})})
	if !ai.IsUnsupported(err) {
		t.Fatalf("err = %v, want an unsupported-request failure", err)
	}
	if !strings.Contains(err.Error(), "text-only") {
		t.Errorf("err = %v, want it to name the model", err)
	}
	if reached {
		t.Error("the request reached the endpoint despite being impossible")
	}
}

// The catalog turns a reference a person types into a model a driver can open.
func TestCatalogResolvesAReference(t *testing.T) {
	m, err := catalog.Model("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.API != ai.APIOpenAIChat {
		t.Errorf("API = %q; DeepSeek speaks Chat Completions, which is why it needs no driver of its own", m.API)
	}
	if m.Vendor != "deepseek" || m.BaseURL == "" || m.ContextWindow == 0 {
		t.Errorf("model = %+v, want the vendor's endpoint and limits filled in", m)
	}

	if _, err := catalog.Model("no-such-model"); err == nil {
		t.Error("an unknown bare reference should not resolve")
	}

	// Every vendor in the table must name a protocol some driver implements,
	// or its models are unreachable and nothing says so until a call fails.
	registered := map[ai.API]bool{}
	for _, a := range ai.RegisteredAPIs() {
		registered[a] = true
	}
	for _, v := range catalog.All() {
		if !registered[v.API] {
			t.Errorf("vendor %s speaks %q, which no linked driver implements", v.ID, v.API)
		}
	}
}

// Sizing a prompt before sending it is what makes it possible to compact
// first, and the answer says whether it can be trusted.
func TestPromptsCanBeSizedBeforeSending(t *testing.T) {
	e := sse(t, false, [2]string{"", `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]},` +
		`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}`})
	client := open(t, e.server.URL, ai.Model{ID: "m", API: ai.APIOpenAIChat, ContextWindow: 1000})

	messages := []ai.Message{ai.UserMessage(strings.Repeat("word ", 200))}
	count, err := client.CountTokens(context.Background(), messages)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if count.Tokens == 0 {
		t.Error("a prompt with a thousand characters cannot be zero tokens")
	}
	if count.Exact {
		t.Error("Chat Completions publishes no counting endpoint; the answer must say it is an estimate")
	}

	left, _, err := client.Headroom(context.Background(), messages)
	if err != nil {
		t.Fatalf("Headroom: %v", err)
	}
	if left <= 0 || left >= 1000 {
		t.Errorf("headroom = %d, want what is left of a 1000-token window", left)
	}
}

// A turn that failed partway still produced text and still cost tokens, and a
// caller needs both.
func TestAFailedTurnHandsBackWhatItProduced(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: "+`{"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"I was part way thr"}}],`+
			`"usage":{"prompt_tokens":3000,"completion_tokens":40}}`+"\n\n")
		w.(http.Flusher).Flush()
		server := w.(http.Hijacker)
		conn, _, _ := server.Hijack()
		_ = conn.Close() // die mid-stream
	}))
	defer server.Close()

	client := open(t, server.URL, ai.Model{ID: "m", API: ai.APIOpenAIChat})
	resp, err := client.Complete(context.Background(), []ai.Message{ai.UserMessage("hi")})
	if err == nil {
		t.Fatal("expected the stream to fail")
	}
	if resp == nil {
		t.Fatal("a failed turn must still hand back what it produced")
	}
	if !strings.HasPrefix(resp.Text(), "I was part way") {
		t.Errorf("Text = %q, want the partial answer", resp.Text())
	}
	if resp.Usage.Output == 0 {
		t.Errorf("Usage = %+v, want the tokens already billed", resp.Usage)
	}
	if !resp.Failed() {
		t.Error("the response should report that the turn failed")
	}
}

// The SDK repairs a conversation before sending it — an interrupted session
// leaves tool calls with no results, and every protocol rejects those. The
// repair must not reach back into the history the caller still holds: their
// conversation is theirs, and silently editing it is how a session loses a
// turn nobody asked it to lose.
func TestRepairDoesNotEditTheCallersHistory(t *testing.T) {
	e := sse(t, false, [2]string{"", `{"id":"1","model":"m","choices":[{"index":0,` +
		`"delta":{"content":"ok"},"finish_reason":"stop"}]}`})

	// An assistant turn whose tool call was never answered — what a mid-stream
	// cancel leaves behind.
	history := []ai.Message{
		ai.UserMessage("go"),
		{Role: ai.RoleAssistant, Content: ai.Content{
			ai.TextBlock("working"),
			ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "t", Input: "{}"}),
		}},
	}

	client := open(t, e.server.URL, ai.Model{ID: "m", API: ai.APIOpenAIChat})
	if _, err := client.Complete(context.Background(), history); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// The unpaired call was stripped from what went on the wire...
	sent, _ := json.Marshal(e.body["messages"])
	if strings.Contains(string(sent), "tool_calls") {
		t.Errorf("the unpaired call reached the endpoint: %s", sent)
	}
	// ...and left alone in the caller's own history.
	if got := len(history[1].ToolCalls()); got != 1 {
		t.Errorf("the caller's history lost a tool call: %d remain, want 1", got)
	}
	if len(history) != 2 {
		t.Errorf("the caller's history has %d messages, want the 2 it started with", len(history))
	}
}

// The other half of the same repair: invalid UTF-8 is replaced on the way out,
// and the caller's own strings are not rewritten under them. A lone UTF-16
// surrogate is what a conversation that passed through a JavaScript runtime
// carries; Go holds it in a string without complaint and providers do not.
func TestRepairReplacesBrokenTextWithoutRewritingTheCallers(t *testing.T) {
	const broken = "hi \xed\xa0\x80 there" // half a surrogate pair

	history := []ai.Message{
		ai.UserMessage(broken),
		{Role: ai.RoleAssistant, Content: ai.Content{
			ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "t", Input: `{"q":"` + broken + `"}`}),
		}},
		ai.ToolResultsMessage(ai.ToolResult{ToolCallID: "1", ToolName: "t", Content: broken}),
	}

	repaired := ai.RepairHistory(history)
	if len(repaired) != 3 {
		t.Fatalf("repaired history has %d messages, want 3; the checks below would pass vacuously", len(repaired))
	}

	for i, m := range repaired {
		for j, block := range m.Content {
			for what, got := range map[string]string{
				"text":        block.Text,
				"tool input":  toolCallInput(block),
				"tool result": toolResultContent(block),
			} {
				if strings.Contains(got, broken) {
					t.Errorf("message %d block %d: %s still carries invalid UTF-8", i, j, what)
				}
			}
		}
	}

	// The caller's own history is untouched, down through every pointer it
	// holds: cloneBlock copies each payload before anything is rewritten.
	if history[0].Content[0].Text != broken {
		t.Error("the caller's message text was rewritten in place")
	}
	if got := history[1].Content[0].ToolCall.Input; !strings.Contains(got, broken) {
		t.Error("the caller's tool-call arguments were rewritten in place")
	}
	if got := history[2].Content[0].ToolResult.Content; got != broken {
		t.Error("the caller's tool result was rewritten in place")
	}
}

func toolCallInput(b ai.Block) string {
	if b.ToolCall == nil {
		return ""
	}
	return b.ToolCall.Input
}

func toolResultContent(b ai.Block) string {
	if b.ToolResult == nil {
		return ""
	}
	return b.ToolResult.Content
}
