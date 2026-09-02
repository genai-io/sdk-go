package google

import (
	"context"
	"encoding/json"
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

// stub is a Gemini endpoint that replays a scripted reply and remembers what
// reached it. Nothing here touches the network or a credential.
type stub struct {
	*httptest.Server

	mu     sync.Mutex
	header http.Header
	body   string
	path   string
	query  string
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
		s.query = r.URL.RawQuery
		s.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

// sse replays events; cut ends the stream mid-flight the way a dropped
// connection does.
func sse(t *testing.T, events ...string) *stub { return streamOf(t, false, events...) }
func cut(t *testing.T, events ...string) *stub { return streamOf(t, true, events...) }

func streamOf(t *testing.T, abort bool, events ...string) *stub {
	return serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
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

func (s *stub) seen() (http.Header, string, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.header, s.path, s.query
}

func driverFor(t *testing.T, cfg ai.Config) *Driver {
	t.Helper()
	if cfg.Model.ID == "" {
		cfg.Model.ID = "gemini-2.5-flash"
	}
	cfg.Model.API = ai.APIGoogleGenAI
	d, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return d.(*Driver)
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

func usage(deltas []ai.Delta) ai.Usage {
	var total ai.Usage
	for _, d := range deltas {
		if d.Usage != nil {
			total = *d.Usage
		}
	}
	return total
}

// Gemini counts thinking outside candidatesTokenCount, unlike every other
// protocol here. Reading only candidates under-reported both the turn and its
// price.
func TestThinkingTokensAreCountedAsOutput(t *testing.T) {
	s := sse(t, `{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}],`+
		`"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":30,`+
		`"candidatesTokenCount":50,"thoughtsTokenCount":400}}`)

	deltas, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	want := ai.Usage{Input: 70, Output: 450, Reasoning: 400, CacheRead: 30}
	if got := usage(deltas); got != want {
		t.Errorf("usage = %+v, want %+v", got, want)
	}
}

// The rung decides which field carries the reasoning switch, and an explicit
// "off" has to be said out loud on a model that reasons by default.
func TestTheReasoningRungReachesTheFieldItsModelWants(t *testing.T) {
	budgets := []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Default: true},
		{Effort: ai.EffortHigh, Budget: 128_000},
	}
	levels := []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Default: true},
		{Effort: ai.EffortHigh, Value: "HIGH"},
	}

	for _, tc := range []struct {
		name   string
		model  ai.Model
		effort ai.Effort
		want   string
	}{
		{
			name:   "a budget model asked for high sends the budget",
			model:  ai.Model{ID: "gemini-2.5-flash", Reasoning: budgets},
			effort: ai.EffortHigh,
			want:   `{"includeThoughts":true,"thinkingBudget":128000}`,
		},
		{
			name:   "a budget model asked for off sends a zero budget",
			model:  ai.Model{ID: "gemini-2.5-flash", Reasoning: budgets},
			effort: ai.EffortOff,
			want:   `{"thinkingBudget":0}`,
		},
		{
			name:   "a level model asked for high sends the level",
			model:  ai.Model{ID: "gemini-3-flash", Reasoning: levels, Compat: ai.GoogleCompat{ThinkingLevel: true}},
			effort: ai.EffortHigh,
			want:   `{"includeThoughts":true,"thinkingLevel":"HIGH"}`,
		},
		{
			name:   "a level model asked for off sends no config, having no level for it",
			model:  ai.Model{ID: "gemini-3-flash", Reasoning: levels, Compat: ai.GoogleCompat{ThinkingLevel: true}},
			effort: ai.EffortOff,
			want:   "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := driverFor(t, ai.Config{Model: tc.model})
			cfg := d.generationConfig(&ai.Request{Effort: tc.effort})
			var got string
			if cfg.ThinkingConfig != nil {
				raw, err := json.Marshal(cfg.ThinkingConfig)
				if err != nil {
					t.Fatalf("marshal: %v", err)
				}
				got = string(raw)
			}
			if got != tc.want {
				t.Errorf("thinkingConfig = %s, want %s", got, tc.want)
			}
		})
	}
}

// The listing carries embedding, image, speech and live models beside the ones
// that answer a prompt, and says which is which.
func TestModelsKeepsOnlyWhatCanAnswerAPrompt(t *testing.T) {
	s := replies(t, http.StatusOK, nil, `{"models":[
		{"name":"models/gemini-2.5-flash","displayName":"Gemini 2.5 Flash","inputTokenLimit":1048576,
		 "supportedGenerationMethods":["generateContent","countTokens"]},
		{"name":"models/text-embedding-004","displayName":"Embedding",
		 "supportedGenerationMethods":["embedContent"]},
		{"name":"models/gemini-2.5-flash-tts","displayName":"TTS",
		 "supportedGenerationMethods":["countTokens"]},
		{"name":"models/gemini-live-2.5","displayName":"Live",
		 "supportedGenerationMethods":["bidiGenerateContent"]},
		{"name":"models/gemini-2.5-pro-exp","displayName":"Experimental",
		 "supportedGenerationMethods":["generateContent"]},
		{"name":"models/imagen-4.0","displayName":"Imagen",
		 "supportedGenerationMethods":["predict"]}]}`)

	models, err := driverFor(t, ai.Config{BaseURL: s.URL}).Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gemini-2.5-flash" {
		t.Fatalf("models = %+v, want only gemini-2.5-flash", models)
	}
	if models[0].ContextWindow != 1048576 {
		t.Errorf("context window = %d, want the listed limit", models[0].ContextWindow)
	}
}

// The endpoint's own error envelope, and the hint it sends with a 429.
func TestAFailedCallKeepsWhatTheEndpointSaid(t *testing.T) {
	s := replies(t, http.StatusTooManyRequests,
		map[string]string{"Retry-After": "12"},
		`{"error":{"code":429,"message":"quota for this model is used up","status":"RESOURCE_EXHAUSTED"}}`)

	_, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
	if !ai.IsKind(err, ai.KindRateLimit) {
		t.Fatalf("error = %v, want a rate limit", err)
	}
	if got := ai.RetryAfter(err); got != 12*time.Second {
		t.Errorf("retry after = %v, want 12s", got)
	}
	var e *ai.Error
	if !errors.As(err, &e) {
		t.Fatalf("error is %T, want *ai.Error", err)
	}
	if e.Code != "RESOURCE_EXHAUSTED" {
		t.Errorf("code = %q, want the status the envelope carried", e.Code)
	}
	if e.Driver != Name {
		t.Errorf("driver = %q, want %q", e.Driver, Name)
	}
}

// A stream that stops mid-answer is worth another attempt; a cancelled one is
// not.
func TestACutStreamIsRetryableAndACancelledOneIsNot(t *testing.T) {
	t.Run("cut", func(t *testing.T) {
		s := cut(t, `{"candidates":[{"content":{"parts":[{"text":"half"}]}}]}`)
		deltas, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL}))
		if !ai.IsKind(err, ai.KindNetwork) {
			t.Fatalf("error = %v, want %q", err, ai.KindNetwork)
		}
		if len(deltas) == 0 {
			t.Error("the text that did arrive was thrown away")
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		s := sse(t, `{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := collect(t, ctx, driverFor(t, ai.Config{BaseURL: s.URL})); !ai.IsKind(err, ai.KindCanceled) {
			t.Fatalf("error = %v, want %q", err, ai.KindCanceled)
		}
	})
}

// The credential travels in a header rather than the query string, where it
// would end up in proxy logs.
func TestTheKeyTravelsInAHeader(t *testing.T) {
	s := sse(t, `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`)
	if _, err := collect(t, context.Background(), driverFor(t, ai.Config{BaseURL: s.URL, APIKey: "k"})); err != nil {
		t.Fatalf("stream: %v", err)
	}
	header, path, query := s.seen()
	if header.Get("x-goog-api-key") != "k" {
		t.Errorf("x-goog-api-key = %q, want the Config's key", header.Get("x-goog-api-key"))
	}
	if strings.Contains(query, "k") && strings.Contains(query, "key=") {
		t.Errorf("query = %q, want no credential in it", query)
	}
	if want := "/v1beta/models/gemini-2.5-flash:streamGenerateContent"; path != want {
		t.Errorf("path = %q, want %q", path, want)
	}
}

// The event reader: framing this protocol actually produces, and one event too
// large for a scanner's default buffer.
func TestTheEventReaderHandlesTheFramingThatArrives(t *testing.T) {
	huge := strings.Repeat("x", 200*1024)
	raw := "data: {\"a\":1}\r\n\r\n" + // CRLF line endings
		": a comment\n\n" + // ignored
		"event: message\ndata: {\"b\":2}\n\n" + // a named event
		"data: [DONE]\n\n" + // the terminator, not a payload
		"data: " + huge + "\n\n" // larger than the default 64KiB buffer

	var got []string
	for payload, err := range sseEvents(strings.NewReader(raw)) {
		if err != nil {
			t.Fatalf("sseEvents: %v", err)
		}
		got = append(got, string(payload))
	}
	want := []string{`{"a":1}`, `{"b":2}`, huge}
	if len(got) != len(want) {
		t.Fatalf("got %d events, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event %d = %.40q…, want %.40q…", i, got[i], want[i])
		}
	}
}
