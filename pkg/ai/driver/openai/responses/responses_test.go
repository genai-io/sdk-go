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
		cfg.Model.ID, cfg.Model.API = "gpt-5", ai.APIOpenAIResponses
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
	return collectReq(t, d, &ai.Request{Messages: []ai.Message{ai.UserMessage("hello")}})
}

func collectReq(t *testing.T, d ai.Driver, req *ai.Request) ([]ai.Delta, error) {
	t.Helper()
	var out []ai.Delta
	for delta, err := range d.Stream(context.Background(), req) {
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
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
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
	_, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
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

func TestFailedTerminalEventsPreserveUsageThroughThePublicClient(t *testing.T) {
	for _, event := range []string{"response.failed", "response.completed"} {
		t.Run(event, func(t *testing.T) {
			s := sse(t,
				`{"type":"response.output_text.delta","delta":"half"}`,
				`{"type":"`+event+`","response":{"id":"failed-id","model":"actual-model","status":"failed","error":{"code":"server_error","message":"failed"},"usage":{"input_tokens":20,"output_tokens":10}}}`,
			)
			client := ai.NewClientWithDriver(driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}), ai.Model{ID: "test", API: ai.APIOpenAIResponses})
			response, err := client.Complete(t.Context(), []ai.Message{ai.UserMessage("hi")})
			if !ai.IsKind(err, ai.KindOverloaded) {
				t.Fatalf("error = %v, want overloaded", err)
			}
			if response.Usage.Input != 20 || response.Usage.Output != 10 || response.ID != "failed-id" || response.Model != "actual-model" || response.Text() != "half" {
				t.Errorf("failed response lost its accounting or metadata: %+v", response)
			}
		})
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
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
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
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
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

	if _, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}})); err != nil {
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

	cfg := ai.Config{Model: ai.Model{BaseURL: s.URL}, APIKey: "sk-this-tenant"}
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

func toolCalls(deltas []ai.Delta) []ai.ToolCall {
	var calls []ai.ToolCall
	for _, d := range deltas {
		if d.Block.Type == ai.BlockToolCall {
			calls = append(calls, *d.Block.ToolCall)
		}
	}
	return calls
}

const completed = `{"type":"response.completed","response":{"id":"resp_1","model":"gpt-5","status":"completed"}}`

// A call's arguments stream against its output-item ID and arrive whole.
func TestAToolCallIsAssembledFromItsArgumentStream(t *testing.T) {
	s := sse(t,
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"Read","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"file_path\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"\"a.py\"}"}`,
		`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"Read","arguments":"{\"file_path\":\"a.py\"}"}}`,
		completed,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	calls := toolCalls(deltas)
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Input != `{"file_path":"a.py"}` {
		t.Fatalf("calls = %+v, want one Read of a.py", calls)
	}
}

// An endpoint may name the same output item differently on each event (a
// proxy that re-encodes item IDs). The call is still one call: keyed by
// call_id, with the finished item's arguments — not a second, empty copy of it
// that runs too.
func TestACallWhoseItemIDChangesBetweenEventsIsStillOneCall(t *testing.T) {
	s := sse(t,
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_added","call_id":"call_1","name":"Read","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_delta","delta":"{\"file_path\":\"a.py\"}"}`,
		`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_done","call_id":"call_1","name":"Read","arguments":"{\"file_path\":\"a.py\"}"}}`,
		completed,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	calls := toolCalls(deltas)
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Input != `{"file_path":"a.py"}` {
		t.Fatalf("calls = %+v, want one Read of a.py", calls)
	}
}

func TestUnfinishedToolCallsSurviveStreamFailures(t *testing.T) {
	for _, tc := range []struct {
		name     string
		terminal string
		wantErr  bool
	}{
		{"EOF", "", true},
		{"error event", `{"type":"error","code":"server_error","message":"failed"}`, true},
		{"failed event", `{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"failed"}}}`, true},
		{"completed with failure", `{"type":"response.completed","response":{"status":"failed","error":{"code":"server_error","message":"failed"}}}`, true},
		{"completed", completed, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events := []string{
				`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"Read"}}`,
				`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"Read","arguments":"{}"}}`,
				`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"Read"}}`,
				`{"type":"response.function_call_arguments.delta","item_id":"fc_2","delta":"{\"file_path\":\"a.py\"}"}`,
			}
			if tc.terminal != "" {
				events = append(events, tc.terminal)
			}
			s := sse(t, events...)
			client := ai.NewClientWithDriver(driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}), ai.Model{ID: "gpt-5", API: ai.APIOpenAIResponses})
			resp, err := client.Complete(t.Context(), []ai.Message{ai.UserMessage("go")})
			if (err != nil) != tc.wantErr || tc.terminal == "" && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("err=%v, wantErr=%v", err, tc.wantErr)
			}
			calls := resp.ToolCalls()
			if len(calls) != 2 || calls[0].ID != "call_1" || calls[1].ID != "call_2" || calls[1].Input != `{"file_path":"a.py"}` {
				t.Errorf("calls=%+v, want both calls and the received arguments without duplicates", calls)
			}
		})
	}
}

func TestToolCallsSurviveATransportReadError(t *testing.T) {
	const payload = "data: " + `{"type":"response.output_item.added","item":{"type":"function_call","id":"fc","call_id":"call","name":"Read"}}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"{}"}` + "\n\n"
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Length", fmt.Sprint(len(payload)+1))
		_, _ = io.WriteString(w, payload)
	}))
	t.Cleanup(s.Close)
	client := ai.NewClientWithDriver(driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}), ai.Model{ID: "gpt-5", API: ai.APIOpenAIResponses})
	resp, err := client.Complete(t.Context(), []ai.Message{ai.UserMessage("go")})
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err=%v, want a transport read error", err)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].ID != "call" || calls[0].Input != `{}` {
		t.Errorf("calls=%+v, want the received tool call and arguments", calls)
	}
}

// reasoner is a model with a reasoning ladder, which is what earns a request
// the encrypted reasoning back.
var reasoner = ai.Model{ID: "gpt-5", API: ai.APIOpenAIResponses, Reasoning: []ai.ReasoningLevel{{Effort: ai.EffortHigh}}}

// The whole conversation is replayed every turn, so nothing is left on the
// server and a reasoning model is asked for the reasoning it will need back.
func TestEveryRequestIsStatelessAndAReasonerAsksForItsReasoningBack(t *testing.T) {
	s := sse(t, completed)
	if _, err := collect(t, driverFor(t, ai.Config{Model: at(reasoner, s.URL)})); err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, body, _ := s.seen()
	if !strings.Contains(body, `"store":false`) {
		t.Errorf("body = %s, want store=false", body)
	}
	if !strings.Contains(body, `"include":["reasoning.encrypted_content"]`) {
		t.Errorf("body = %s, want the encrypted reasoning included", body)
	}
}

// A model that cannot reason is not asked for reasoning it will never produce.
func TestAModelThatCannotReasonIsStatelessWithoutAskingForReasoning(t *testing.T) {
	s := sse(t, completed)
	if _, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}})); err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, body, _ := s.seen()
	if !strings.Contains(body, `"store":false`) {
		t.Errorf("body = %s, want store=false", body)
	}
	if strings.Contains(body, `"include"`) {
		t.Errorf("body = %s, want no include", body)
	}
}

// Reasoning items come back on the regular API too, so the next turn can
// replay them.
func TestReasoningItemsAreKeptForReplayOnTheRegularAPI(t *testing.T) {
	s := sse(t,
		`{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"thought"}]}}`,
		completed,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{Model: at(reasoner, s.URL)}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var got []ai.ReasoningItem
	for _, d := range deltas {
		if d.Block.Type == ai.BlockReasoning {
			got = append(got, *d.Block.Reasoning)
		}
	}
	want := ai.ReasoningItem{API: ai.APIOpenAIResponses, ID: "rs_1", EncryptedContent: "opaque", Summary: "thought"}
	if len(got) != 1 || got[0] != want {
		t.Errorf("reasoning = %+v, want [%+v]", got, want)
	}
}

// The output cap is sent unless the backend is marked as rejecting it.
func TestNoMaxOutputTokensLeavesTheCapOff(t *testing.T) {
	for _, tc := range []struct {
		compat ai.OpenAIResponsesCompat
		want   bool
	}{
		{ai.OpenAIResponsesCompat{}, true},
		{ai.OpenAIResponsesCompat{NoMaxOutputTokens: true}, false},
	} {
		s := sse(t, completed)
		model := ai.Model{ID: "gpt-5", API: ai.APIOpenAIResponses, Compat: tc.compat}
		req := &ai.Request{Messages: []ai.Message{ai.UserMessage("hello")}, MaxTokens: 100}
		if _, err := collectReq(t, driverFor(t, ai.Config{Model: at(model, s.URL)}), req); err != nil {
			t.Fatalf("stream: %v", err)
		}
		_, body, _ := s.seen()
		if got := strings.Contains(body, `"max_output_tokens":100`); got != tc.want {
			t.Errorf("compat %+v: max_output_tokens sent = %v, want %v", tc.compat, got, tc.want)
		}
	}
}

// An answer cut short by the content filter was declined, not truncated.
func TestAContentFilteredAnswerIsARefusal(t *testing.T) {
	s := sse(t,
		`{"type":"response.output_text.delta","delta":"Sure, h"}`,
		`{"type":"response.incomplete","response":{"id":"resp_5","model":"gpt-5","status":"incomplete",`+
			`"incomplete_details":{"reason":"content_filter"}}}`,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := stopReason(deltas); got != ai.StopRefusal {
		t.Errorf("stop reason = %q, want %q", got, ai.StopRefusal)
	}
}

// Cache writes are part of input_tokens; they are reported apart so they can
// be priced as writes.
func TestCacheWritesAreReportedApartFromFreshInput(t *testing.T) {
	s := sse(t,
		`{"type":"response.completed","response":{"id":"resp_6","model":"gpt-5","status":"completed",`+
			`"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":40,"cache_write_tokens":30},`+
			`"output_tokens":5,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":105}}}`,
	)
	deltas, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	want := ai.Usage{Input: 30, Output: 5, CacheRead: 40, CacheWrite: 30}
	if got := usage(deltas); got != want {
		t.Errorf("usage = %+v, want %+v", got, want)
	}
}

// TestACodexBackendIsListedFromItsOwnCatalog pins the subscription listing:
// the Codex CLI's query, slugs as IDs, a hidden model left out, and a refusal
// classified rather than swallowed.
func TestACodexBackendIsListedFromItsOwnCatalog(t *testing.T) {
	status := http.StatusOK
	var query string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query = r.URL.Path + "?" + r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, `{"models":[
			{"slug":"gpt-5.5-codex","display_name":"GPT-5.5 Codex","context_window":272000,
			 "default_reasoning_level":"medium",
			 "supported_reasoning_levels":[{"effort":"low"},{"effort":"medium"}]},
			{"slug":"hidden","show_in_picker":false}]}`)
	}))
	t.Cleanup(srv.Close)

	lister := driverFor(t, ai.Config{Model: ai.Model{BaseURL: srv.URL + "/backend-api/codex",
		ID: "-", API: ai.APIOpenAIResponses, Compat: ai.OpenAIResponsesCompat{CodexModelListing: true},
	}}).(ai.ModelLister)

	models, err := lister.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if query != "/backend-api/codex/models?client_version="+codexClientVersion {
		t.Errorf("asked %s, want the Codex catalog", query)
	}
	if len(models) != 1 || models[0].ID != "gpt-5.5-codex" || models[0].ContextWindow != 272000 {
		t.Fatalf("models = %+v, want the one shown model", models)
	}
	if r := models[0].Reasoning; len(r) != 2 || !r[1].Default || r[0].Default {
		t.Errorf("reasoning = %+v, want low and a default medium", r)
	}

	status = http.StatusUnauthorized
	_, err = lister.Models(context.Background())
	var aiErr *ai.Error
	if !errors.As(err, &aiErr) || aiErr.Kind != ai.KindAuth {
		t.Errorf("err = %v, want KindAuth", err)
	}
}

// at is m served from url.
func at(m ai.Model, url string) ai.Model {
	m.BaseURL = url
	return m
}
