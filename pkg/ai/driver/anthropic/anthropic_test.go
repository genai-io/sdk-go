package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// stub is a Messages endpoint that replays a scripted stream and remembers what
// reached it. Nothing here touches the network or a credential.
type stub struct {
	*httptest.Server

	mu     sync.Mutex
	header http.Header
	body   string
	path   string
}

// event is one named server-sent event, which this protocol — unlike the
// others — requires: the SDK dispatches on the event name, not the payload.
type event struct{ name, data string }

func sse(t *testing.T, events ...event) *stub { return streamOf(t, false, events...) }
func cut(t *testing.T, events ...event) *stub { return streamOf(t, true, events...) }

func streamOf(t *testing.T, abort bool, events ...event) *stub {
	return serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.name, e.data)
		}
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		if abort {
			panic(http.ErrAbortHandler)
		}
	})
}

func replies(t *testing.T, status int, header map[string]string, body string) *stub {
	return serve(t, func(w http.ResponseWriter, r *http.Request) {
		for k, v := range header {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprint(w, body)
	})
}

func serve(t *testing.T, handle func(w http.ResponseWriter, r *http.Request)) *stub {
	t.Helper()
	s := &stub{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.header = r.Header.Clone()
		s.body = string(raw)
		s.path = r.URL.Path
		s.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *stub) seen() (http.Header, string, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.header, s.body, s.path
}

func driverFor(t *testing.T, cfg ai.Config) ai.Driver {
	t.Helper()
	if cfg.Model.ID == "" {
		cfg.Model.ID = "claude-test"
	}
	cfg.Model.API = ai.APIAnthropicMessages
	d, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return d
}

func collect(t *testing.T, ctx context.Context, d ai.Driver, req *ai.Request) ([]ai.Delta, error) {
	t.Helper()
	if len(req.Messages) == 0 {
		req.Messages = []ai.Message{ai.UserMessage("hi")}
	}
	var out []ai.Delta
	for delta, err := range d.Stream(ctx, req) {
		if err != nil {
			return out, err
		}
		out = append(out, delta)
	}
	return out, nil
}

// done is the shortest stream an endpoint can send: enough to make the request
// and read a reply.
var done = []event{
	{"message_start", `{"type":"message_start","message":{"id":"m","model":"claude-test","usage":{"input_tokens":1}}}`},
	{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
}

// The four prompt figures are billed at four different rates, so they have to
// arrive apart rather than as one total.
func TestCacheTokensArriveSplitByWhatTheyCost(t *testing.T) {
	s := sse(t, event{"message_start", `{"type":"message_start","message":{"id":"m","model":"claude-test",` +
		`"usage":{"input_tokens":12,"cache_creation_input_tokens":900,` +
		`"cache_creation":{"ephemeral_1h_input_tokens":600,"ephemeral_5m_input_tokens":300},` +
		`"cache_read_input_tokens":4000}}}`})

	deltas, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}), &ai.Request{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(deltas) == 0 || deltas[0].Usage == nil {
		t.Fatal("message_start carried no usage")
	}
	want := ai.Usage{Input: 12, CacheWrite: 900, CacheWrite1h: 600, CacheRead: 4000}
	if got := *deltas[0].Usage; got != want {
		t.Errorf("usage = %+v, want %+v", got, want)
	}
}

// The error type is the machine-readable half and the message is the readable
// one. The SDK's own Error method renders neither: it is the request line, the
// status, the request ID and the raw body all at once.
func TestAFailureKeepsItsCodeAndItsMessage(t *testing.T) {
	for _, tc := range []struct {
		name    string
		status  int
		header  map[string]string
		body    string
		code    string
		kind    ai.ErrorKind
		message string
		after   time.Duration
	}{
		{
			name:    "an oversized prompt is not an ordinary bad request",
			status:  http.StatusBadRequest,
			body:    `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 300000 tokens > 200000 maximum"}}`,
			code:    "invalid_request_error",
			kind:    ai.KindContextExceeded,
			message: "prompt is too long: 300000 tokens > 200000 maximum",
		},
		{
			name:    "a rate limit carries the provider's own hint",
			status:  http.StatusTooManyRequests,
			header:  map[string]string{"Retry-After": "18"},
			body:    `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`,
			code:    "rate_limit_error",
			kind:    ai.KindRateLimit,
			message: "slow down",
			after:   18 * time.Second,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := replies(t, tc.status, tc.header, tc.body)
			_, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}), &ai.Request{})
			var e *ai.Error
			if !errors.As(err, &e) {
				t.Fatalf("error is %T (%v), want *ai.Error", err, err)
			}
			if e.Code != tc.code {
				t.Errorf("code = %q, want %q", e.Code, tc.code)
			}
			if e.Kind != tc.kind {
				t.Errorf("kind = %q, want %q", e.Kind, tc.kind)
			}
			if e.Message != tc.message {
				t.Errorf("message = %q, want just the provider's own", e.Message)
			}
			if e.RetryAfter != tc.after {
				t.Errorf("retry after = %v, want %v", e.RetryAfter, tc.after)
			}
		})
	}
}

// The zero value of the option changes nothing, so the field is sent only when
// it is on: an Anthropic-compatible host may reject a property it has never
// heard of.
func TestParallelToolUseIsOnlyDisabledWhenAsked(t *testing.T) {
	tool := ai.Tool{Schema: ai.Schema{Name: "weather", Definition: map[string]any{"type": "object"}}}

	for _, tc := range []struct {
		name  string
		off   bool
		wants bool
	}{
		{name: "left alone", off: false, wants: false},
		{name: "asked for", off: true, wants: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := sse(t, done...)
			req := &ai.Request{
				Messages:        []ai.Message{ai.UserMessage("hi")},
				Tools:           []ai.Tool{tool},
				ToolChoice:      ai.ToolChoiceRequired,
				ProtocolOptions: Options{DisableParallelToolUse: tc.off},
			}
			if _, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}), req); err != nil {
				t.Fatalf("stream: %v", err)
			}
			_, body, _ := s.seen()
			if got := strings.Contains(body, "disable_parallel_tool_use"); got != tc.wants {
				t.Errorf("disable_parallel_tool_use present = %v, want %v; body was %s", got, tc.wants, body)
			}
		})
	}
}

// Redacted thinking is unreadable but not disposable: the API rejects a
// tool-use continuation whose history dropped it.
func TestRedactedThinkingComesBackAndGoesBackOut(t *testing.T) {
	s := sse(t,
		event{"message_start", `{"type":"message_start","message":{"id":"m","model":"claude-test","usage":{"input_tokens":1}}}`},
		event{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"EroBCkYIA..."}}`},
		event{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		event{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
	)
	thinker := ai.Model{ID: "claude-test", Reasoning: []ai.ReasoningLevel{
		{Effort: ai.EffortOff}, {Effort: ai.EffortHigh, Budget: 8000},
	}}

	deltas, err := collect(t, context.Background(),
		driverFor(t, ai.Config{BaseURL: s.URL, Model: thinker}),
		&ai.Request{Effort: ai.EffortHigh})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var item ai.ReasoningItem
	for _, d := range deltas {
		if d.Block.Type == ai.BlockReasoning && d.Block.Reasoning != nil {
			item = *d.Block.Reasoning
		}
	}
	if item.EncryptedContent != "EroBCkYIA..." {
		t.Fatalf("reasoning = %+v, want the redacted block's data", item)
	}

	// Now send it back, the way a tool-use continuation has to.
	back := sse(t, done...)
	req := &ai.Request{
		Effort: ai.EffortHigh,
		Messages: []ai.Message{
			ai.UserMessage("hi"),
			{Role: ai.RoleAssistant, Content: ai.Content{ai.ReasoningBlock(item)}},
			ai.UserMessage("go on"),
		},
	}
	if _, err := collect(t, context.Background(),
		driverFor(t, ai.Config{BaseURL: back.URL, Model: thinker}), req); err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, body, _ := back.seen()
	if !strings.Contains(body, `"type":"redacted_thinking"`) || !strings.Contains(body, "EroBCkYIA...") {
		t.Errorf("request body dropped the redacted thinking: %s", body)
	}
}

// Two parallel calls that arrived with no IDs of their own — Gemini leaves them
// empty — must not become the same Anthropic ID, or the results cannot be
// paired with the calls.
func TestUnnamedToolCallsGetDistinctIDsThatTheirResultsStillMatch(t *testing.T) {
	s := sse(t, done...)
	req := &ai.Request{
		Messages: []ai.Message{
			ai.UserMessage("hi"),
			{Role: ai.RoleAssistant, Content: ai.Content{
				ai.ToolCallBlock(ai.ToolCall{Name: "a", Input: "{}"}),
				ai.ToolCallBlock(ai.ToolCall{Name: "b", Input: "{}"}),
			}},
			ai.ToolResultsMessage(
				ai.ToolResult{ToolName: "a", Content: ai.TextContent("one")},
				ai.ToolResult{ToolName: "b", Content: ai.TextContent("two")},
			),
		},
	}
	if _, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}), req); err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, body, _ := s.seen()
	if strings.Count(body, `"toolu_compat_1"`) != 2 || strings.Count(body, `"toolu_compat_2"`) != 2 {
		t.Errorf("each call and its result should share one ID, and the two calls differ; body was %s", body)
	}
}

// The whole truth about a request is its Config. A credential or host sitting
// in the process environment must not reach the wire.
func TestAnAmbientCredentialIsNeverUsed(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
	t.Setenv("ANTHROPIC_BASE_URL", "https://ambient.invalid")

	s := sse(t, done...)
	if _, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}), &ai.Request{}); err != nil {
		t.Fatalf("stream: %v", err)
	}
	header, _, path := s.seen()
	if path == "" {
		t.Fatal("the request never reached the Config's endpoint; ANTHROPIC_BASE_URL won")
	}
	for _, name := range []string{"X-Api-Key", "Authorization"} {
		if got := header.Get(name); got != "" {
			t.Errorf("%s = %q, want nothing: the Config carried no credential", name, got)
		}
	}
}

// A bearer endpoint and a key endpoint want the credential in different
// headers, and the Config's key is what goes there.
func TestTheConfigsCredentialTravelsInTheHeaderTheEndpointWants(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")

	for _, tc := range []struct {
		name   string
		compat ai.AnthropicCompat
		header string
	}{
		{name: "Anthropic itself", header: "X-Api-Key"},
		{name: "a re-host taking a bearer token", compat: ai.AnthropicCompat{BearerAuth: true}, header: "Authorization"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := sse(t, done...)
			cfg := ai.Config{BaseURL: s.URL, APIKey: "sk-this-tenant", Model: ai.Model{ID: "claude-test", Compat: tc.compat}}
			if _, err := collect(t, context.Background(), driverFor(t, cfg), &ai.Request{}); err != nil {
				t.Fatalf("stream: %v", err)
			}
			header, _, _ := s.seen()
			if !strings.Contains(header.Get(tc.header), "sk-this-tenant") {
				t.Errorf("%s = %q, want the Config's key", tc.header, header.Get(tc.header))
			}
		})
	}
}

// A stream that stops mid-answer is worth another attempt; a cancelled one is
// not.
func TestACutStreamIsRetryableAndACancelledOneIsNot(t *testing.T) {
	t.Run("cut", func(t *testing.T) {
		s := cut(t, done[0])
		_, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}), &ai.Request{})
		if !ai.IsKind(err, ai.KindNetwork) {
			t.Fatalf("error = %v, want %q", err, ai.KindNetwork)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		s := sse(t, done...)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := collect(t, ctx, driverFor(t, ai.Config{BaseURL: s.URL}), &ai.Request{})
		if !ai.IsKind(err, ai.KindCanceled) {
			t.Fatalf("error = %v, want %q", err, ai.KindCanceled)
		}
	})
}

// roundTripFunc lets a test see the URL a request was about to go to without
// letting it leave the process.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// A Config with no endpoint means the vendor's own host, not whatever the
// environment happens to hold.
func TestTheDefaultHostIsTheVendorsOwn(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "https://ambient.invalid")

	var asked string
	cfg := ai.Config{HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		asked = r.URL.String()
		return nil, errors.New("no further")
	})}}
	if _, err := collect(t, context.Background(), driverFor(t, cfg), &ai.Request{}); err == nil {
		t.Fatal("the stubbed transport let a request through")
	}
	if want := "https://api.anthropic.com/v1/messages"; asked != want {
		t.Errorf("request went to %q, want %q", asked, want)
	}
}
