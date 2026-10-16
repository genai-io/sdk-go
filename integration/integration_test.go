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
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

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
	header http.Header
}

// sse replays server-sent events. Anthropic names its events; the others do
// not, which is the only difference the framing has.
func sse(t *testing.T, named bool, events ...[2]string) *stub {
	t.Helper()
	e := &stub{}
	e.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		e.path = r.URL.Path
		e.header = r.Header.Clone()
		_ = json.Unmarshal(raw, &e.body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			if named {
				_, _ = fmt.Fprintf(w, "event: %s\n", ev[0])
			}
			_, _ = fmt.Fprintf(w, "data: %s\n\n", ev[1])
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
		e.header = r.Header.Clone()
		_ = json.Unmarshal(raw, &e.body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(e.server.Close)
	return e
}

func open(t *testing.T, url string, m ai.Model) *ai.Client {
	t.Helper()
	c, err := ai.New(ai.Config{Model: m, APIKey: "k", BaseURL: url})
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

// A header the Config names reaches the endpoint on every protocol.
func TestEveryProtocolCarriesTheConfigHeaders(t *testing.T) {
	tests := map[string]struct {
		model ai.Model
		serve func(*testing.T) *stub
	}{
		"anthropic": {
			model: ai.Model{ID: "claude-test", API: ai.APIAnthropicMessages, MaxOutput: 1024},
			serve: func(t *testing.T) *stub {
				return sse(t, true,
					[2]string{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test","usage":{}}}`},
					[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}`},
				)
			},
		},
		"openai chat completions": {
			model: ai.Model{ID: "gpt-test", API: ai.APIOpenAIChat},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"id":"1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`},
					[2]string{"", "[DONE]"},
				)
			},
		},
		"openai responses": {
			model: ai.Model{ID: "gpt-test", API: ai.APIOpenAIResponses},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"type":"response.completed","response":{"id":"r1","model":"gpt-test",` +
						`"status":"completed","output":[]}}`},
				)
			},
		},
		"google gemini": {
			model: ai.Model{ID: "gemini-test", API: ai.APIGoogleGenAI},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}`},
				)
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			e := tc.serve(t)
			client, err := ai.New(ai.Config{
				Model: tc.model, APIKey: "k", BaseURL: e.server.URL,
				Headers: map[string]string{"X-Tenant": "endpoint"},
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			if _, err = client.Complete(context.Background(), []ai.Message{ai.UserMessage("hello")}); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if got := e.header.Get("X-Tenant"); got != "endpoint" {
				t.Errorf("X-Tenant = %q, want %q", got, "endpoint")
			}
		})
	}
}

// A tool offered, called, answered, and replayed — the shape of every turn a
// tool-using application sends after its first.
func TestToolsRoundTrip(t *testing.T) {
	tool := searchTool()

	e := sse(t, true,
		[2]string{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test","usage":{"input_tokens":10}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,` +
			`"content_block":{"type":"tool_use","id":"call_1","name":"search","input":{}}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,` +
			`"delta":{"type":"input_json_delta","partial_json":` +
			`"{\"query\":\"go\",\"priority\":\"low\",\"limit\":5}"}}`},
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
	if err := tool.Validate(calls[0].Input); err != nil {
		t.Fatalf("the model's own arguments failed their schema: %v", err)
	}
	var args SearchArgs
	if err := calls[0].UnmarshalArgs(&args); err != nil || args.Query != "go" {
		t.Errorf("UnmarshalArgs = %+v, %v", args, err)
	}

	// Replaying the call with its result is what the next turn sends.
	history := []ai.Message{
		ai.UserMessage("find go"),
		resp.Message(),
		ai.ToolResultsMessage(ai.ToolResult{ToolCallID: calls[0].ID, ToolName: "search", Content: ai.TextContent("found")}),
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
	if !ai.IsKind(err, ai.KindUnsupported) {
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
	if m.Vendor != "deepseek" || m.BaseURL == "" {
		t.Errorf("model = %+v, want the vendor's endpoint filled in", m)
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

// A turn that failed partway still produced text and still cost tokens, and a
// caller needs both.
func TestAFailedTurnHandsBackWhatItProduced(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: "+`{"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"I was part way thr"}}],`+
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
		ai.ToolResultsMessage(ai.ToolResult{ToolCallID: "1", ToolName: "t", Content: ai.TextContent(broken)}),
	}

	repaired := ai.Repair(history)
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
	if got := history[2].Content[0].ToolResult.Text(); got != broken {
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
	return b.ToolResult.Text()
}

// Go ignores an unrecognised struct tag without a word. For this package's
// keys that means a field silently loses the one thing it was annotated with —
// the description the model was supposed to read, or the enum that was
// supposed to constrain it. Every way of getting the key wrong is refused by
// name instead.
func TestAMistypedSchemaTagIsNotSilent(t *testing.T) {
	type OtherLibrary struct {
		Q string `json:"q" jsonschema:"description=what to look for"`
	}
	type Pluralised struct {
		Q string `json:"q" enums:"low|high"`
	}
	type Misspelled struct {
		Q string `json:"q" descrption:"what to look for"`
	}

	for name, tc := range map[string]struct {
		derive func()
		want   string
	}{
		"another library's key": {func() { _ = ai.ToolSchema[OtherLibrary]("t", "") }, "jsonschema"},
		"a pluralised keyword":  {func() { _ = ai.ToolSchema[Pluralised]("t", "") }, "did you mean enum"},
		"a misspelled keyword":  {func() { _ = ai.ToolSchema[Misspelled]("t", "") }, "did you mean description"},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("the tag must not be silently dropped")
				}
				if msg := fmt.Sprint(r); !strings.Contains(msg, tc.want) {
					t.Errorf("panic = %q\nwant it to mention %q", msg, tc.want)
				}
			}()
			tc.derive()
		})
	}

	// An unrelated tool's tag is left alone: it is not one edit from anything
	// here, so it is somebody else's and none of our business.
	type Unrelated struct {
		Q string `json:"q" db:"query" validate:"required" description:"what to look for"`
	}
	got := ai.ToolSchema[Unrelated]("t", "").DefinitionMap()
	props := got["properties"].(map[string]any)["q"].(map[string]any)
	if props["description"] != "what to look for" {
		t.Errorf("q = %v, want the description alongside the unrelated tags", props)
	}
}

// The schema has to be one the providers accept, not merely one that is valid.
// These are the rules a general-purpose generator gets wrong, and each is a
// documented requirement of OpenAI's strict structured output.
func TestDerivedSchemasAreWhatProvidersAccept(t *testing.T) {
	type Deep struct {
		N int `json:"n"`
	}
	type inner struct {
		Tag string `json:"tag"`
	}
	type Case struct {
		inner              // promoted, the way encoding/json promotes it
		At       time.Time `json:"at"`
		List     []Deep    `json:"list"`
		Maybe    *int      `json:"maybe,omitempty"`
		Priority string    `json:"priority" enum:"low|medium|high"`
		Count    int       `json:"count" description:"how many" minimum:"1" maximum:"99"`
		Skip     string    `json:"-"`
	}

	schema := ai.ToolSchema[Case]("t", "").DefinitionMap()
	properties, _ := schema["properties"].(map[string]any)

	names := map[string]bool{}
	for name := range properties {
		names[name] = true
	}
	want := []string{"tag", "at", "list", "maybe", "priority", "count"}
	for _, name := range want {
		if !names[name] {
			t.Errorf("property %q is missing; got %v", name, sortedKeys(properties))
		}
	}
	if names["Skip"] || names["-"] {
		t.Error(`a json:"-" field reached the schema`)
	}

	// Every field is required, the optional one included: strict mode demands
	// it, and optionality is the null in the type union instead.
	required := map[string]bool{}
	for _, name := range schema["required"].([]any) {
		required[name.(string)] = true
	}
	for _, name := range want {
		if !required[name] {
			t.Errorf("%q is not in required; strict structured output requires every field", name)
		}
	}
	maybe := properties["maybe"].(map[string]any)
	if fmt.Sprint(maybe["type"]) != "[integer null]" {
		t.Errorf(`maybe.type = %v, want ["integer","null"] — that is how strict mode says optional`, maybe["type"])
	}

	// No boolean schemas, anywhere: strict mode rejects them outright.
	var boolSchemas func(any, string)
	boolSchemas = func(node any, path string) {
		switch n := node.(type) {
		case bool:
			t.Errorf("%s is a boolean schema, which strict structured output rejects", path)
		case map[string]any:
			for k, v := range n {
				if k == "additionalProperties" {
					if _, isBool := v.(bool); isBool {
						continue // false here is required, not a schema
					}
				}
				boolSchemas(v, path+"."+k)
			}
		case []any:
			for i, v := range n {
				boolSchemas(v, fmt.Sprintf("%s[%d]", path, i))
			}
		}
	}
	boolSchemas(schema, "root")

	// A type that marshals to something other than its fields is described by
	// what it marshals to, or the schema rejects its own Go type's JSON.
	at := properties["at"].(map[string]any)
	if at["type"] != "string" || at["format"] != "date-time" {
		t.Errorf("at = %v, want a date-time string", at)
	}

	// Constraints from the tag.
	priority := properties["priority"].(map[string]any)
	if fmt.Sprint(priority["enum"]) != "[low medium high]" {
		t.Errorf("priority.enum = %v", priority["enum"])
	}
	count := properties["count"].(map[string]any)
	if count["minimum"] != 1.0 || count["maximum"] != 99.0 || count["description"] != "how many" {
		t.Errorf("count = %v, want the tag's description and bounds", count)
	}

	// Objects are closed, at every level.
	if schema["additionalProperties"] != false {
		t.Error("the root object is not closed; strict structured output requires it")
	}
	items := properties["list"].(map[string]any)["items"].(map[string]any)
	if items["additionalProperties"] != false {
		t.Error("a nested object is not closed")
	}
}

func sortedKeys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// The checker exists for one job: turn a model's mistake into a sentence the
// model can act on. These are the mistakes models actually make.
func TestArgumentCheckingSaysWhatToFix(t *testing.T) {
	tool := searchTool()

	for name, tc := range map[string]struct{ input, want string }{
		"a missing field": {
			`{"priority":"low","limit":3}`, "missing required property: query",
		},
		"a value outside the enum": {
			`{"query":"go","priority":"urgent","limit":3}`, "priority must be one of low, medium or high",
		},
		"a string where a number belongs": {
			`{"query":"go","priority":"low","limit":"three"}`, "limit must be integer, not string",
		},
		"a number out of range": {
			`{"query":"go","priority":"low","limit":900}`, "limit must be at most 50",
		},
		"an invented property": {
			`{"query":"go","priority":"low","limit":3,"sort":"asc"}`, "unknown property: sort",
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := tool.Validate(tc.input)
			if err == nil {
				t.Fatalf("%s was accepted", tc.input)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %q\nwant it to contain %q", err, tc.want)
			}
		})
	}

	if err := tool.Validate(`{"query":"go","priority":"low","limit":3}`); err != nil {
		t.Errorf("a correct call was rejected: %v", err)
	}
}

// The argument types for these tests. A tool's struct holds exactly what the
// model may send it, and every tag on it is prompt text.
type (
	SearchArgs struct {
		Query    string `json:"query" description:"what to look for"`
		Priority string `json:"priority" enum:"low|medium|high"`
		Limit    int    `json:"limit" description:"how many results" minimum:"1" maximum:"50"`
	}
)

func searchTool() ai.Schema {
	return ai.ToolSchema[SearchArgs]("search", "search the knowledge base")
}

// Tool choice is one field on the wire with four states, and it is one value
// here for the same reason. A mode beside a name could say "no tools" and
// "this tool" at once, and then every driver would have to agree on which
// wins — four independent copies of one rule, with nothing checking they stay
// the same. This pins what each protocol actually sends for each state.
func TestEveryProtocolSaysTheSameFourThingsAboutToolChoice(t *testing.T) {
	tool := searchTool()

	// Where each protocol puts the constraint, and what it puts there.
	protocols := []struct {
		name  string
		model ai.Model
		at    func(body map[string]any) any // nil when the field is absent
		want  map[string]string             // state name -> the JSON it sends
	}{
		{
			name:  "anthropic-messages",
			model: ai.Model{ID: "m", API: ai.APIAnthropicMessages, MaxOutput: 64},
			at:    func(b map[string]any) any { return b["tool_choice"] },
			want: map[string]string{
				"auto":     "absent",
				"none":     `{"type":"none"}`,
				"required": `{"type":"any"}`,
				"named":    `{"name":"search","type":"tool"}`,
			},
		},
		{
			name:  "openai-chat-completions",
			model: ai.Model{ID: "m", API: ai.APIOpenAIChat},
			at:    func(b map[string]any) any { return b["tool_choice"] },
			want: map[string]string{
				"auto":     "absent",
				"none":     `"none"`,
				"required": `"required"`,
				"named":    `{"function":{"name":"search"},"type":"function"}`,
			},
		},
		{
			name:  "openai-responses",
			model: ai.Model{ID: "m", API: ai.APIOpenAIResponses},
			at:    func(b map[string]any) any { return b["tool_choice"] },
			want: map[string]string{
				"auto":     "absent",
				"none":     `"none"`,
				"required": `"required"`,
				"named":    `{"name":"search","type":"function"}`,
			},
		},
		{
			name:  "google-genai",
			model: ai.Model{ID: "m", API: ai.APIGoogleGenAI},
			at:    func(b map[string]any) any { return b["toolConfig"] },
			want: map[string]string{
				"auto":     "absent",
				"none":     `{"functionCallingConfig":{"mode":"NONE"}}`,
				"required": `{"functionCallingConfig":{"mode":"ANY"}}`,
				"named":    `{"functionCallingConfig":{"allowedFunctionNames":["search"],"mode":"ANY"}}`,
			},
		},
	}

	states := map[string]ai.ToolChoice{
		"auto":     ai.ToolChoiceAuto,
		"none":     ai.ToolChoiceNone,
		"required": ai.ToolChoiceRequired,
		"named":    ai.ToolChoiceNamed("search"),
	}

	for _, p := range protocols {
		for state, choice := range states {
			t.Run(p.name+"/"+state, func(t *testing.T) {
				e := sse(t, p.model.API == ai.APIAnthropicMessages)
				client := open(t, e.server.URL, p.model)
				_, _ = client.Complete(context.Background(),
					[]ai.Message{ai.UserMessage("hello")},
					ai.WithTools(tool), ai.WithToolChoice(choice))

				got := "absent"
				if field := p.at(e.body); field != nil {
					raw, err := json.Marshal(field)
					if err != nil {
						t.Fatalf("marshal: %v", err)
					}
					got = string(raw)
				}
				if got != p.want[state] {
					t.Errorf("%s sent %s\nwant %s", state, got, p.want[state])
				}
			})
		}
	}
}

// The one thing that has to hold: what the model is told it may send, and
// what the arguments decode into, are the same declaration. Naming T once is
// what makes them so — a schema built from one type and a decode into another
// compiles perfectly well and fails only once a model is on the other end.
func TestASchemaAndItsArgumentsAreOneDeclaration(t *testing.T) {
	tool := searchTool()

	schema := tool.DefinitionMap()
	properties := schema["properties"].(map[string]any)
	fields := reflect.VisibleFields(reflect.TypeOf(SearchArgs{}))
	if len(properties) != len(fields) {
		t.Errorf("properties = %v, want one per field of SearchArgs", properties)
	}
	for _, field := range fields {
		name := strings.Split(field.Tag.Get("json"), ",")[0]
		if _, described := properties[name]; !described {
			t.Errorf("%s is a field the model is never told about", name)
		}
	}

	// So a call the schema accepts is one the arguments can hold, and one it
	// refuses never reaches the function.
	if err := tool.Validate(`{"query":"go","priority":"low","limit":3}`); err != nil {
		t.Errorf("a correct call was rejected: %v", err)
	}
}

// A tool with no name is one the model cannot call, and arguments that are not
// a struct are not arguments. Both are mistakes in a declaration rather than
// conditions to handle, so they are refused where the tool is built.
func TestAToolIsRefusedBeforeItCanBeOffered(t *testing.T) {
	for name, tc := range map[string]struct {
		build func()
		want  string
	}{
		"no name": {
			func() { ai.ToolSchema[SearchArgs]("", "d") },
			"has no name",
		},
		"arguments that are not a struct": {
			func() { ai.ToolSchema[string]("s", "d") },
			"must be a struct",
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("the tool was built anyway")
				}
				if msg := fmt.Sprint(r); !strings.Contains(msg, tc.want) {
					t.Errorf("panic = %q\nwant it to mention %q", msg, tc.want)
				}
			}()
			tc.build()
		})
	}
}

// A tool that looked at something answers with a picture. Three protocols carry
// one; Gemini is refused before the network, because a model asked about an
// image it was never shown answers anyway.
func TestAToolResultCarriesAnImageWhereTheProtocolTakesOne(t *testing.T) {
	history := []ai.Message{
		ai.UserMessage("what does it look like?"),
		{Role: ai.RoleAssistant, Content: ai.Content{
			ai.ToolCallBlock(ai.ToolCall{ID: "call_1", Name: "screenshot", Input: "{}"}),
		}},
		ai.ToolResultsMessage(ai.ToolResult{
			ToolCallID: "call_1", ToolName: "screenshot",
			Content: ai.Content{
				ai.TextBlock("the page, rendered"),
				ai.ImageBlock(ai.Image{MediaType: "image/png", Data: "AAAA"}),
			},
		}),
	}

	// The models can see, so what is tested is the protocol, not the capability check.
	seeing := []ai.Modality{ai.ModalityText, ai.ModalityImage}

	carried := map[string]struct {
		model ai.Model
		serve func(*testing.T) *stub
	}{
		"anthropic": {
			model: ai.Model{ID: "claude-test", API: ai.APIAnthropicMessages, MaxOutput: 1024, Input: seeing},
			serve: func(t *testing.T) *stub {
				return sse(t, true,
					[2]string{"message_start", `{"type":"message_start","message":{"id":"m1","model":"claude-test","usage":{}}}`},
					[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}`},
				)
			},
		},
		"openai chat completions": {
			model: ai.Model{ID: "gpt-test", API: ai.APIOpenAIChat, Input: seeing},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"id":"1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`},
				)
			},
		},
		"openai responses": {
			model: ai.Model{ID: "gpt-test", API: ai.APIOpenAIResponses, Input: seeing},
			serve: func(t *testing.T) *stub {
				return sse(t, false,
					[2]string{"", `{"type":"response.completed","response":{"id":"r1","model":"gpt-test",` +
						`"status":"completed","output":[]}}`},
				)
			},
		},
	}
	for name, tc := range carried {
		t.Run(name, func(t *testing.T) {
			e := tc.serve(t)
			if _, err := open(t, e.server.URL, tc.model).Complete(context.Background(), history); err != nil {
				t.Fatalf("Complete: %v", err)
			}
			sent, _ := json.Marshal(e.body)
			if !strings.Contains(string(sent), "AAAA") {
				t.Errorf("the image did not reach the wire: %s", sent)
			}
			if !strings.Contains(string(sent), "the page, rendered") {
				t.Errorf("the text beside it did not reach the wire: %s", sent)
			}
		})
	}

	refused := map[string]ai.Model{
		"google gemini": {ID: "gemini-test", API: ai.APIGoogleGenAI},
	}
	for name, model := range refused {
		t.Run(name, func(t *testing.T) {
			reached := false
			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { reached = true }))
			defer server.Close()

			_, err := open(t, server.URL, model).Complete(context.Background(), history)
			if err == nil {
				t.Fatal("a protocol that cannot carry the image accepted it")
			}
			if !strings.Contains(err.Error(), "tool result") {
				t.Errorf("err = %v, want it to name what cannot be sent", err)
			}
			if reached {
				t.Error("the request went out with the image dropped from it")
			}
		})
	}
}
