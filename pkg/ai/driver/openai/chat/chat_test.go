package chat

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

// stub is a Chat Completions endpoint that replays a scripted stream and
// remembers what reached it. Nothing here touches the network or a credential.
type stub struct {
	*httptest.Server

	mu     sync.Mutex
	header http.Header
	body   string
	path   string
}

// sse replays events and ends the stream cleanly. cut ends it mid-flight, the
// way a dropped connection does.
func sse(t *testing.T, events ...string) *stub { return stream(t, false, events...) }
func cut(t *testing.T, events ...string) *stub { return stream(t, true, events...) }

func stream(t *testing.T, abort bool, events ...string) *stub {
	t.Helper()
	s := &stub{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.header = r.Header.Clone()
		s.body = string(raw)
		s.path = r.URL.Path
		s.mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
		}
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		if abort {
			// Closes the connection without a reply the client can finish
			// reading, which is what a cut stream looks like from here.
			panic(http.ErrAbortHandler)
		}
	}))
	t.Cleanup(s.Close)
	return s
}

func replies(t *testing.T, status int, header map[string]string, body string) *stub {
	t.Helper()
	s := &stub{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.path = r.URL.Path
		s.mu.Unlock()
		for k, v := range header {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprint(w, body)
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
		cfg.Model.ID = "some-model"
	}
	cfg.Model.API = ai.APIOpenAIChat
	d, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return d
}

func collect(t *testing.T, ctx context.Context, d ai.Driver) ([]ai.Delta, error) {
	t.Helper()
	var out []ai.Delta
	for delta, err := range d.Stream(ctx, &ai.Request{Messages: []ai.Message{ai.UserMessage("hi")}}) {
		if err != nil {
			return out, err
		}
		out = append(out, delta)
	}
	return out, nil
}

func toolCalls(deltas []ai.Delta) []ai.ToolCall {
	var out []ai.ToolCall
	for _, d := range deltas {
		if d.Block.Type == ai.BlockToolCall && d.Block.ToolCall != nil {
			out = append(out, *d.Block.ToolCall)
		}
	}
	return out
}

func thinking(deltas []ai.Delta) string {
	var sb strings.Builder
	for _, d := range deltas {
		if d.Block.Type == ai.BlockThinking {
			sb.WriteString(d.Block.Text)
		}
	}
	return sb.String()
}

// One call's arguments arrive as fragments across chunks, keyed by index, and
// the name or ID may turn up later than the first fragment for that index.
func TestToolCallFragmentsAccumulateByIndex(t *testing.T) {
	s := sse(t,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[`+
			`{"index":0,"function":{"arguments":"{\"city\""}},`+
			`{"index":1,"function":{"arguments":"{\"n\""}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[`+
			`{"index":0,"id":"call_a","function":{"name":"weather","arguments":":\"Oslo\"}"}},`+
			`{"index":1,"id":"call_b","function":{"name":"count","arguments":":2}"}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		"[DONE]",
	)
	deltas, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	got := toolCalls(deltas)
	want := []ai.ToolCall{
		{ID: "call_a", Name: "weather", Input: `{"city":"Oslo"}`},
		{ID: "call_b", Name: "count", Input: `{"n":2}`},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d calls, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i].ID || got[i].Name != want[i].Name || got[i].Input != want[i].Input {
			t.Errorf("call %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// The Delta contract says everything produced stays on the Response. A stream
// that dies after the arguments have arrived must still hand them over.
func TestToolCallsSurviveACutStream(t *testing.T) {
	s := cut(t,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[`+
			`{"index":0,"id":"call_a","function":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}}]}}]}`,
	)
	deltas, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
	if err == nil {
		t.Fatal("a cut stream ended without an error")
	}
	if got := toolCalls(deltas); len(got) != 1 || got[0].Name != "weather" {
		t.Errorf("tool calls = %+v, want the one collected before the cut", got)
	}
	if !ai.IsKind(err, ai.KindNetwork) {
		t.Errorf("kind = %v, want %q — a cut stream is worth retrying", err, ai.KindNetwork)
	}
}

// Two spellings carry the same thing, and neither is in the standard schema.
func TestReasoningIsReadUnderEitherKey(t *testing.T) {
	for _, tc := range []struct {
		name  string
		delta string
	}{
		{"reasoning_content", `{"reasoning_content":"weighing it up"}`},
		{"reasoning", `{"reasoning":"weighing it up"}`},
		{"reasoning object is not text", `{"reasoning":{"effort":"high"},"reasoning_content":"weighing it up"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := sse(t, `{"id":"c1","model":"m","choices":[{"index":0,"delta":`+tc.delta+`}]}`, "[DONE]")
			deltas, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			if got := thinking(deltas); got != "weighing it up" {
				t.Errorf("thinking = %q, want the reasoning the endpoint streamed", got)
			}
		})
	}
}

// A model with no reasoning ladder can still be served by an endpoint that
// reasons anyway. Gating on the rung threw that away.
func TestReasoningIsKeptFromAModelThatDeclaresNoLadder(t *testing.T) {
	s := sse(t, `{"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning":"thinking out loud"}}]}`, "[DONE]")
	d := driverFor(t, ai.Config{BaseURL: s.URL, Model: ai.Model{ID: "local-model"}})
	deltas, err := collect(t, context.Background(), d)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := thinking(deltas); got != "thinking out loud" {
		t.Errorf("thinking = %q, want it kept: the endpoint sent it", got)
	}
}

// A 429 carries the provider's own hint about when to come back.
func TestARateLimitCarriesItsRetryAfter(t *testing.T) {
	s := replies(t, http.StatusTooManyRequests,
		map[string]string{"Retry-After": "30"},
		`{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"}}`)

	_, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
	if !ai.IsKind(err, ai.KindRateLimit) {
		t.Fatalf("error = %v, want a rate limit", err)
	}
	if got := ai.RetryAfter(err); got != 30*time.Second {
		t.Errorf("retry after = %v, want 30s", got)
	}
	var e *ai.Error
	if errors.As(err, &e) && e.Driver != Name {
		t.Errorf("driver = %q, want %q", e.Driver, Name)
	}
}

// A caller who cancels gets told so, and told that retrying is not the answer.
func TestACanceledCallIsNotAFailure(t *testing.T) {
	s := sse(t, `{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := collect(t, ctx, driverFor(t, ai.Config{BaseURL: s.URL}))
	if !ai.IsKind(err, ai.KindCanceled) {
		t.Fatalf("error = %v, want %q", err, ai.KindCanceled)
	}
	if ai.IsRetryable(err) {
		t.Error("a cancelled call is reported as retryable")
	}
}

// The whole truth about a request is its Config. A credential or host sitting
// in the process environment must not reach the wire.
func TestAnAmbientCredentialIsNeverUsed(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-ambient-tenant-key")
	t.Setenv("OPENAI_BASE_URL", "https://ambient.invalid")
	t.Setenv("OPENAI_ORG_ID", "org-ambient")
	t.Setenv("OPENAI_PROJECT_ID", "proj-ambient")

	s := sse(t, `{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`, "[DONE]")
	if _, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL})); err != nil {
		t.Fatalf("stream: %v", err)
	}
	header, _, path := s.seen()
	if path == "" {
		t.Fatal("the request never reached the Config's endpoint; OPENAI_BASE_URL won")
	}
	for _, name := range []string{"Authorization", "OpenAI-Organization", "OpenAI-Project"} {
		if got := header.Get(name); got != "" {
			t.Errorf("%s = %q, want nothing: the Config carried no credential", name, got)
		}
	}
}
