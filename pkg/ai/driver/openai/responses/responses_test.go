package responses

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

	"github.com/genai-io/sdk-go/pkg/ai"
)

// stub is a Responses endpoint that replays a scripted event stream and
// remembers what reached it. Nothing here touches the network or a credential.
type stub struct {
	*httptest.Server

	mu     sync.Mutex
	header http.Header
	body   string
	path   string
}

func sse(t *testing.T, events ...string) *stub {
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
		cfg.Model = ai.Model{ID: "gpt-5", API: ai.APIOpenAIResponses}
	}
	d, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return d
}

// collect drains a stream into the deltas it produced and the error it ended
// on.
func collect(t *testing.T, d ai.Driver) ([]ai.Delta, error) {
	t.Helper()
	var out []ai.Delta
	for delta, err := range d.Stream(context.Background(), &ai.Request{
		Messages: []ai.Message{ai.UserMessage("hello")},
	}) {
		if err != nil {
			return out, err
		}
		out = append(out, delta)
	}
	return out, nil
}

// usage returns the merged token accounting the deltas carried.
func usage(deltas []ai.Delta) ai.Usage {
	var total ai.Usage
	for _, d := range deltas {
		if d.Usage != nil {
			total = *d.Usage
		}
	}
	return total
}

func stopReason(deltas []ai.Delta) ai.StopReason {
	var stop ai.StopReason
	for _, d := range deltas {
		if d.StopReason != "" {
			stop = d.StopReason
		}
	}
	return stop
}

func text(deltas []ai.Delta) string {
	var sb strings.Builder
	for _, d := range deltas {
		if d.Block.Type == ai.BlockText {
			sb.WriteString(d.Block.Text)
		}
	}
	return sb.String()
}

// A truncated answer arrives as its own event type. Read only inside
// response.completed it looked like an empty success.
func TestATruncatedAnswerSaysSoAndCostsWhatItCost(t *testing.T) {
	s := sse(t,
		`{"type":"response.output_text.delta","delta":"half an ans"}`,
		`{"type":"response.incomplete","response":{"id":"resp_1","model":"gpt-5","status":"incomplete",`+
			`"incomplete_details":{"reason":"max_output_tokens"},`+
			`"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":40},`+
			`"output_tokens":64,"output_tokens_details":{"reasoning_tokens":16},"total_tokens":164}}}`,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{BaseURL: s.URL}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := text(deltas); got != "half an ans" {
		t.Errorf("text = %q, want the fragment that did arrive", got)
	}
	if got := stopReason(deltas); got != ai.StopMaxTokens {
		t.Errorf("stop reason = %q, want %q", got, ai.StopMaxTokens)
	}
	want := ai.Usage{Input: 60, Output: 64, Reasoning: 16, CacheRead: 40}
	if got := usage(deltas); got != want {
		t.Errorf("usage = %+v, want %+v", got, want)
	}
}

// A server-side failure is an error, not a silent empty turn.
func TestAFailedResponseIsAnError(t *testing.T) {
	s := sse(t,
		`{"type":"response.failed","response":{"id":"resp_2","model":"gpt-5","status":"failed",`+
			`"error":{"code":"server_error","message":"the model is having a bad day"}}}`,
	)
	_, err := collect(t, driverFor(t, ai.Config{BaseURL: s.URL}))
	if err == nil {
		t.Fatal("a failed response ended the stream without an error")
	}
	var e *ai.Error
	if !errors.As(err, &e) {
		t.Fatalf("error is %T, want *ai.Error", err)
	}
	if e.Code != "server_error" {
		t.Errorf("code = %q, want the one the endpoint sent", e.Code)
	}
	if e.Kind != ai.KindOverloaded {
		t.Errorf("kind = %q, want %q — a server error is worth retrying", e.Kind, ai.KindOverloaded)
	}
	if !strings.Contains(e.Message, "bad day") {
		t.Errorf("message = %q, want the endpoint's own", e.Message)
	}
}

// A refusal is an answer: the words are the model's, and the stop reason says
// it declined rather than finished.
func TestARefusalIsAnAnswerAndSaysSo(t *testing.T) {
	s := sse(t,
		`{"type":"response.refusal.delta","delta":"I can't help with that."}`,
		`{"type":"response.completed","response":{"id":"resp_3","model":"gpt-5","status":"completed",`+
			`"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":0},`+
			`"output_tokens":8,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18}}}`,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{BaseURL: s.URL}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := text(deltas); got != "I can't help with that." {
		t.Errorf("text = %q, want the refusal the model wrote", got)
	}
	if got := stopReason(deltas); got != ai.StopRefusal {
		t.Errorf("stop reason = %q, want %q — response.completed must not overwrite it", got, ai.StopRefusal)
	}
}

// Reasoning tokens are inside output_tokens; they travel separately so a caller
// can see what thinking cost.
func TestReasoningTokensAreReportedAndStillCountedAsOutput(t *testing.T) {
	s := sse(t,
		`{"type":"response.completed","response":{"id":"resp_4","model":"gpt-5","status":"completed",`+
			`"usage":{"input_tokens":7,"input_tokens_details":{"cached_tokens":0},`+
			`"output_tokens":900,"output_tokens_details":{"reasoning_tokens":850},"total_tokens":907}}}`,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{BaseURL: s.URL}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	got := usage(deltas)
	if got.Reasoning != 850 {
		t.Errorf("reasoning = %d, want 850", got.Reasoning)
	}
	if got.Output != 900 {
		t.Errorf("output = %d, want 900 — reasoning is already inside it", got.Output)
	}
	if got := stopReason(deltas); got != ai.StopEndTurn {
		t.Errorf("stop reason = %q, want %q", got, ai.StopEndTurn)
	}
}

// The whole truth about a request is its Config. A credential or host sitting
// in the process environment must not reach the wire.
func TestAnAmbientCredentialIsNeverUsed(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-ambient-tenant-key")
	t.Setenv("OPENAI_BASE_URL", "https://ambient.invalid")
	t.Setenv("OPENAI_ORG_ID", "org-ambient")
	t.Setenv("OPENAI_PROJECT_ID", "proj-ambient")

	s := sse(t, `{"type":"response.completed","response":{"id":"r","model":"gpt-5","status":"completed",`+
		`"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},`+
		`"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`)

	if _, err := collect(t, driverFor(t, ai.Config{BaseURL: s.URL})); err != nil {
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

// The credential the Config does carry is the one that is sent.
func TestTheConfigsCredentialIsWhatTravels(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-ambient-tenant-key")

	s := sse(t, `{"type":"response.completed","response":{"id":"r","model":"gpt-5","status":"completed",`+
		`"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},`+
		`"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`)

	cfg := ai.Config{BaseURL: s.URL, APIKey: "sk-this-tenant"}
	if _, err := collect(t, driverFor(t, cfg)); err != nil {
		t.Fatalf("stream: %v", err)
	}
	header, _, _ := s.seen()
	if got := header.Get("Authorization"); got != "Bearer sk-this-tenant" {
		t.Errorf("Authorization = %q, want the Config's key", got)
	}
}

// roundTripFunc lets a test see the URL a request was about to go to without
// letting it leave the process.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// A Config with no endpoint means the vendor's own host, not whatever the
// environment happens to hold.
func TestTheDefaultHostIsTheVendorsOwn(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "https://ambient.invalid")

	var asked string
	cfg := ai.Config{HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		asked = r.URL.String()
		return nil, errors.New("no further")
	})}}
	if _, err := collect(t, driverFor(t, cfg)); err == nil {
		t.Fatal("the stubbed transport let a request through")
	}
	if want := "https://api.openai.com/v1/responses"; asked != want {
		t.Errorf("request went to %q, want %q", asked, want)
	}
}
