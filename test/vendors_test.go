package integration_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"github.com/genai-io/sdk-go/pkg/ai/provider"
)

// The path an application actually walks to reach a vendor: an environment
// variable, a "vendor/model" reference, a request on the wire.
//
// Each case below is a real catalog entry, and the assertion is that the SDK
// reached the endpoint the vendor documents, presented the credential the way
// that vendor wants it, and spoke the protocol the catalog says it speaks. A
// vendor is a row in a table, so this is what "adding a vendor is data, not
// code" has to mean in practice.
func TestReachingEachVendor(t *testing.T) {
	tests := map[string]struct {
		ref      string // what a person types
		envVar   string // where its credential lives
		protocol ai.API // what the catalog says it speaks
		host     string // the endpoint it is documented to use
		// endpointEnv and endpointURL are for a vendor that has no default
		// host, whose endpoint names a tenant's own resource or region.
		endpointEnv string
		endpointURL string
		// credential is how this vendor expects the key presented.
		credential func(h http.Header) string
	}{
		"anthropic": {
			ref:        "anthropic/claude-opus-5",
			envVar:     "ANTHROPIC_API_KEY",
			protocol:   ai.APIAnthropicMessages,
			host:       "", // the protocol's own host
			credential: func(h http.Header) string { return h.Get("X-Api-Key") },
		},
		"openai": {
			ref:        "openai/gpt-4.1",
			envVar:     "OPENAI_API_KEY",
			protocol:   ai.APIOpenAIResponses,
			host:       "",
			credential: func(h http.Header) string { return strings.TrimPrefix(h.Get("Authorization"), "Bearer ") },
		},
		"azure openai": {
			// Azure serves Responses itself, so this is the same protocol as
			// the first-party entry reached at a tenant's own resource.
			ref:         "azure-openai/gpt-5.5",
			envVar:      "AZURE_OPENAI_API_KEY",
			protocol:    ai.APIOpenAIResponses,
			host:        "", // no default: the endpoint variable supplies it
			endpointEnv: "AZURE_OPENAI_ENDPOINT",
			endpointURL: "https://my-resource.openai.azure.com",
			credential:  func(h http.Header) string { return strings.TrimPrefix(h.Get("Authorization"), "Bearer ") },
		},
		"amazon bedrock": {
			// Bedrock fronts the open-weight models with Chat Completions
			// only, so the catalog says so rather than pretending otherwise.
			ref:         "bedrock-openai/openai.gpt-oss-120b-1:0",
			envVar:      "AWS_BEARER_TOKEN_BEDROCK",
			protocol:    ai.APIOpenAIChat,
			host:        "",
			endpointEnv: "AWS_BEDROCK_BASE_URL",
			endpointURL: "https://bedrock-runtime.us-west-2.amazonaws.com",
			credential:  func(h http.Header) string { return strings.TrimPrefix(h.Get("Authorization"), "Bearer ") },
		},
		"alibaba (Qwen)": {
			ref:        "alibaba/qwen3.7-plus",
			envVar:     "DASHSCOPE_API_KEY",
			protocol:   ai.APIOpenAIChat,
			host:       "https://dashscope.aliyuncs.com/compatible-mode/v1",
			credential: func(h http.Header) string { return strings.TrimPrefix(h.Get("Authorization"), "Bearer ") },
		},
		"deepseek": {
			ref:        "deepseek/deepseek-v4-pro",
			envVar:     "DEEPSEEK_API_KEY",
			protocol:   ai.APIOpenAIChat,
			host:       "https://api.deepseek.com",
			credential: func(h http.Header) string { return strings.TrimPrefix(h.Get("Authorization"), "Bearer ") },
		},
		"moonshot (Kimi)": {
			ref:        "moonshot/kimi-k3",
			envVar:     "MOONSHOT_API_KEY",
			protocol:   ai.APIOpenAIChat,
			host:       "https://api.moonshot.cn/v1",
			credential: func(h http.Header) string { return strings.TrimPrefix(h.Get("Authorization"), "Bearer ") },
		},
		"google": {
			ref:      "google/gemini-3-pro",
			envVar:   "GEMINI_API_KEY",
			protocol: ai.APIGoogleGenAI,
			host:     "",
			// Gemini takes its key in a header, never in the URL, where it
			// would end up in proxy and server logs.
			credential: func(h http.Header) string { return h.Get("X-Goog-Api-Key") },
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			// The catalog is what turns the reference into something openable.
			model, err := catalog.Model(tc.ref)
			if err != nil {
				t.Fatalf("catalog.Model(%q): %v", tc.ref, err)
			}
			if model.API != tc.protocol {
				t.Errorf("protocol = %q, want %q", model.API, tc.protocol)
			}
			if model.BaseURL != tc.host {
				t.Errorf("endpoint = %q, want %q", model.BaseURL, tc.host)
			}
			// A window of zero means "unknown", which is honest only when the
			// entry says why — the vendor publishes nothing. Silently zero is
			// a caller who cannot size a prompt and is not told so.
			if model.ContextWindow == 0 {
				v, _ := catalog.Find(model.Vendor)
				if v.Note == "" {
					t.Errorf("%s states no context window and no reason; a caller cannot size a prompt "+
						"against it and nothing says that is deliberate", tc.ref)
				}
			}

			// Credential resolution reads exactly the variable the vendor
			// documents, and nothing else.
			t.Setenv(tc.envVar, "the-key")
			if tc.endpointEnv != "" {
				t.Setenv(tc.endpointEnv, tc.endpointURL)
			}
			cfg, err := auth.Config(tc.ref)
			if err != nil {
				t.Fatalf("auth.Config(%q): %v", tc.ref, err)
			}
			if cfg.APIKey != "the-key" {
				t.Errorf("APIKey = %q, want the value of %s", cfg.APIKey, tc.envVar)
			}

			// And the request that leaves carries it the way this vendor wants.
			var seen http.Header
			var path string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seen, path = r.Header.Clone(), r.URL.Path
				w.Header().Set("Content-Type", "text/event-stream")
			}))
			defer server.Close()

			cfg.BaseURL = server.URL
			client, err := ai.Open(cfg)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			_, _ = client.Complete(context.Background(), []ai.Message{ai.UserMessage("hi")})

			if got := tc.credential(seen); got != "the-key" {
				t.Errorf("the endpoint received %q as the credential; headers = %v", got, seen)
			}
			if path == "" {
				t.Error("no request reached the endpoint")
			}
		})
	}
}

// A missing credential is reported before anything is sent, naming the
// variable to set — not as a 401 from the provider after a round trip.
func TestAMissingCredentialSaysWhichVariable(t *testing.T) {
	for _, v := range []string{"DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"} {
		if err := os.Unsetenv(v); err != nil {
			t.Fatal(err)
		}
	}
	_, err := auth.Config("deepseek/deepseek-v4-pro")
	if err == nil {
		t.Fatal("expected a failure with no credential set")
	}
	if !strings.Contains(err.Error(), "DEEPSEEK_API_KEY") {
		t.Errorf("err = %v, want it to name the variable to set", err)
	}
}

// A vendor whose host names a tenant's own resource has no default to fall
// back to. Falling back to the protocol owner's host would send an Azure or
// Bedrock credential to api.openai.com and report the mistake as a 401 from
// the wrong company, so the endpoint variable is checked before anything is
// sent — and the suffix each endpoint needs is added to the bare URL people
// copy out of a console.
func TestAVendorWithNoDefaultHostRequiresItsEndpoint(t *testing.T) {
	tests := map[string]struct {
		ref         string
		keyEnv      string
		endpointEnv string
		given       string // what a person pastes
		want        string // where the request actually goes
	}{
		"azure": {
			ref:         "azure-openai/gpt-5.5",
			keyEnv:      "AZURE_OPENAI_API_KEY",
			endpointEnv: "AZURE_OPENAI_ENDPOINT",
			given:       "https://my-resource.openai.azure.com",
			want:        "https://my-resource.openai.azure.com/openai/v1",
		},
		"bedrock": {
			ref:         "bedrock-openai/openai.gpt-oss-20b-1:0",
			keyEnv:      "AWS_BEARER_TOKEN_BEDROCK",
			endpointEnv: "AWS_BEDROCK_BASE_URL",
			given:       "https://bedrock-runtime.us-west-2.amazonaws.com",
			want:        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Setenv(tc.keyEnv, "the-key")
			t.Setenv(tc.endpointEnv, "")

			_, err := auth.Config(tc.ref)
			if err == nil {
				t.Fatalf("%s resolved with no endpoint set; the credential would have gone to the "+
					"protocol owner's host", tc.ref)
			}
			if !strings.Contains(err.Error(), tc.endpointEnv) {
				t.Errorf("err = %v, want it to name the variable to set", err)
			}

			// And the suffix is added to what a person actually pastes.
			t.Setenv(tc.endpointEnv, tc.given)
			cfg, err := auth.Config(tc.ref)
			if err != nil {
				t.Fatalf("auth.Config(%q): %v", tc.ref, err)
			}
			if cfg.BaseURL != tc.want {
				t.Errorf("endpoint = %q, want %q", cfg.BaseURL, tc.want)
			}
		})
	}
}

// The same prompt, the same code, three different wire protocols. This is the
// claim the SDK makes, so it is worth asserting rather than describing: what
// reaches each endpoint is shaped by that endpoint's protocol, and the caller
// wrote none of it.
func TestOnePromptThreeWireShapes(t *testing.T) {

	tests := map[string]struct {
		model ai.Model
		want  func(body map[string]any) error
	}{
		"anthropic keeps the system prompt out of the turns": {
			model: ai.Model{ID: "claude-x", API: ai.APIAnthropicMessages, MaxOutput: 1024},
			want: func(b map[string]any) error {
				if b["system"] == nil {
					return fmt.Errorf(`no top-level "system" field`)
				}
				msgs, _ := b["messages"].([]any)
				if len(msgs) != 1 {
					return fmt.Errorf("messages = %d, want only the user turn", len(msgs))
				}
				return nil
			},
		},
		"chat completions makes it the first message": {
			model: ai.Model{ID: "qwen-x", API: ai.APIOpenAIChat},
			want: func(b map[string]any) error {
				msgs, _ := b["messages"].([]any)
				if len(msgs) != 2 {
					return fmt.Errorf("messages = %d, want the system turn and the user turn", len(msgs))
				}
				if !strings.Contains(fmt.Sprint(msgs[0]), "be brief") {
					return fmt.Errorf("first message = %v, want the system prompt", msgs[0])
				}
				return nil
			},
		},
		"gemini calls it systemInstruction": {
			model: ai.Model{ID: "gemini-x", API: ai.APIGoogleGenAI},
			want: func(b map[string]any) error {
				if b["systemInstruction"] == nil {
					return fmt.Errorf(`no "systemInstruction" field`)
				}
				return nil
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw := make([]byte, r.ContentLength)
				_, _ = r.Body.Read(raw)
				_ = json.Unmarshal(raw, &body)
				w.Header().Set("Content-Type", "text/event-stream")
			}))
			defer server.Close()

			client, err := ai.Open(ai.Config{Model: tc.model, APIKey: "k", BaseURL: server.URL})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			_, _ = client.Complete(context.Background(),
				[]ai.Message{ai.UserMessage("hello")}, ai.WithSystem("be brief"))

			if err := tc.want(body); err != nil {
				t.Errorf("%v\n  body = %+v", err, body)
			}
		})
	}
}

// A normalized rung is only worth having if it reaches each endpoint in the
// shape that endpoint wants. DeepSeek is the sharpest case: it reasons unless
// told not to, and "not to" is a different wire field from "yes, this much" —
// so a caller who assumed "unset means off" is billed for reasoning they never
// asked for.
func TestOneRungReachesEachEndpointItsOwnWay(t *testing.T) {
	tests := map[string]struct {
		ref    string
		effort ai.Effort
		want   map[string]string // field the endpoint must receive
	}{
		"deepseek off is an explicit disable": {
			ref: "deepseek/deepseek-v4-pro", effort: ai.EffortOff,
			want: map[string]string{"thinking": `{"type":"disabled"}`},
		},
		"deepseek high is an effort string": {
			ref: "deepseek/deepseek-v4-pro", effort: ai.EffortHigh,
			want: map[string]string{"reasoning_effort": `"high"`},
		},
		"qwen wants a flag and a budget": {
			ref: "alibaba/qwen3.7-plus", effort: ai.EffortMedium,
			want: map[string]string{"enable_thinking": "true"},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			model, err := catalog.Model(tc.ref)
			if err != nil {
				t.Fatalf("catalog.Model: %v", err)
			}
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw := make([]byte, r.ContentLength)
				_, _ = r.Body.Read(raw)
				_ = json.Unmarshal(raw, &body)
				w.Header().Set("Content-Type", "text/event-stream")
			}))
			defer server.Close()

			client, err := ai.Open(ai.Config{Model: model, APIKey: "k", BaseURL: server.URL})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			_, _ = client.Complete(context.Background(),
				[]ai.Message{ai.UserMessage("hi")}, ai.WithEffort(tc.effort))

			for field, want := range tc.want {
				got, present := body[field]
				if !present {
					t.Fatalf("no %q on the wire; body = %+v", field, body)
				}
				raw, _ := json.Marshal(got)
				if string(raw) != want {
					t.Errorf("%s = %s, want %s", field, raw, want)
				}
			}
		})
	}
}

// And the trap itself: on DeepSeek, saying nothing is not the same as saying
// off. A caller who leaves Effort unset is reasoning, and paying for it.
func TestUnsetEffortIsNotOffOnAModelThatReasonsByDefault(t *testing.T) {
	model, err := catalog.Model("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("catalog.Model: %v", err)
	}
	def, ok := model.DefaultLevel()
	if !ok {
		t.Fatal("DeepSeek states a default rung; without one this trap is invisible")
	}
	if def.Effort == ai.EffortOff {
		t.Skip("DeepSeek now defaults to off — update the example that warns about this")
	}
	if !model.Reasons() {
		t.Error("a model with a default rung must report that it reasons")
	}
}

// Pricing.Cost is an estimate from a published card, and some cards are
// conditional in ways the card cannot express — DeepSeek bills half price for
// seventeen hours of every day. A caller who shows that figure as authoritative
// is wrong most of the time, so the entry has to say so.
func TestAConditionalRateCardSaysSo(t *testing.T) {
	model, err := catalog.Model("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("catalog.Model: %v", err)
	}
	if !model.Pricing.Known() {
		t.Fatal("DeepSeek publishes a card; without one this test guards nothing")
	}
	v, ok := catalog.Find(model.Vendor)
	if !ok {
		t.Fatalf("no catalog vendor %q", model.Vendor)
	}
	if !strings.Contains(strings.ToLower(v.Note), "off-peak") {
		t.Errorf("DeepSeek's card is half price off-peak and the entry does not say so:\n  %q", v.Note)
	}

	// And the currency is carried, so a figure is never shown bare. Several
	// vendors publish in CNY, and summing those with USD produces a number
	// that looks authoritative and means nothing.
	if model.Pricing.Currency == "" {
		t.Error("a known rate card with no currency cannot be displayed or summed safely")
	}
}

// Every vendor that states a rate states its currency with it.
func TestEveryRateCardCarriesItsCurrency(t *testing.T) {
	for _, v := range catalog.All() {
		for _, m := range v.ModelList() {
			if m.Pricing.Known() && m.Pricing.Currency == "" {
				t.Errorf("%s states rates with no currency", m)
			}
		}
	}
}

// A cache-write rate can only ever bill if the protocol behind it reports
// cache-write tokens, and only the Anthropic Messages driver does. Anywhere
// else the rate sits in the table unable to apply, which is a figure the SDK
// would quietly under-report — so the entry has to say so out loud.
func TestACacheWriteRateCanBeBilledOrSaysWhyNot(t *testing.T) {
	for _, v := range catalog.All() {
		for _, m := range v.ModelList() {
			if m.Pricing.CacheWrite == 0 || m.API == ai.APIAnthropicMessages {
				continue
			}
			if !strings.Contains(strings.ToLower(v.Note), "cache write") {
				t.Errorf("%s prices cache writes at %g but %s cannot report them, "+
					"and the vendor note does not say so:\n  %q",
					m, m.Pricing.CacheWrite, m.API, v.Note)
			}
		}
	}
}

// A model may name a rung this package has never heard of. The portable
// constants are a vocabulary, not a closed set — a live listing or a
// hand-built Model can declare its own, and asking for it by name sends it.
// What must still fail is a name that is neither, because that is a typo and
// silently falling back to the default is how a request quietly stops meaning
// what it said.
func TestAModelMayNameItsOwnReasoningRung(t *testing.T) {
	const ultra ai.Effort = "ultra" // not one of ai.Efforts

	model := ai.Model{
		ID: "custom", API: ai.APIOpenAIChat,
		Compat:    ai.OpenAIChatCompat{Thinking: ai.ThinkingEffort},
		Reasoning: []ai.ReasoningLevel{{Effort: ai.EffortLow, Value: "low"}, {Effort: ultra, Value: "ultra"}},
	}

	if err := model.Validate(nil, ai.WithEffort(ultra)); err != nil {
		t.Errorf("a rung the model declares must be sendable: %v", err)
	}
	if !model.Offers(ultra) {
		t.Error("Offers must report a rung the ladder declares")
	}

	// ...and it reaches the wire as itself, not as something nearby.
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","choices":[{"index":0,"message":{"role":"assistant",` +
			`"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(server.Close)

	model.BaseURL = server.URL
	client, err := ai.Open(ai.Config{Model: model, APIKey: "k"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := client.Complete(context.Background(),
		[]ai.Message{ai.UserMessage("hi")}, ai.WithEffort(ultra)); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := body["reasoning_effort"]; got != "ultra" {
		t.Errorf("reasoning_effort = %v, want the model's own rung %q; body = %+v", got, ultra, body)
	}

	// A name that is neither portable nor declared is still a typo.
	err = model.Validate(nil, ai.WithEffort("ulta"))
	if err == nil {
		t.Fatal("a rung that is neither portable nor declared must be rejected")
	}
	if !strings.Contains(err.Error(), "ulta") || !strings.Contains(err.Error(), "ultra") {
		t.Errorf("the error should name both the typo and what the model offers: %v", err)
	}
}

// Models and the catalog hand out snapshots. The clone that used to sit on the
// way out of Endpoint.Models was removed as redundant — every entry is built
// fresh — so this is what proves it stayed redundant. A caller that appends to
// a returned reasoning ladder must not be editing what the next caller reads.
func TestReturnedModelsDoNotAliasWhatTheyCameFrom(t *testing.T) {
	ep := provider.New(provider.Config{
		ID:  "acme",
		API: ai.APIOpenAIChat,
		Models: []ai.Model{{
			ID:        "acme-pro",
			Reasoning: []ai.ReasoningLevel{{Effort: ai.EffortLow, Value: "low"}},
			Headers:   map[string]string{"X-Tier": "pro"},
		}},
	})

	first := ep.Models()
	if len(first) != 1 {
		t.Fatalf("Models() = %d entries, want 1", len(first))
	}
	first[0].Reasoning = append(first[0].Reasoning, ai.ReasoningLevel{Effort: ai.EffortMax})
	first[0].Headers["X-Tier"] = "tampered"
	first[0].ID = "renamed"

	second := ep.Models()
	if got := len(second[0].Reasoning); got != 1 {
		t.Errorf("the ladder grew to %d rungs; a caller edited the endpoint's own model", got)
	}
	if got := second[0].Headers["X-Tier"]; got != "pro" {
		t.Errorf("headers = %q, want the endpoint's own value", got)
	}
	if second[0].ID != "acme-pro" {
		t.Errorf("ID = %q, want acme-pro", second[0].ID)
	}

	// The same guarantee from the catalog, whose tables are package-level.
	m, err := catalog.Model("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("catalog.Model: %v", err)
	}
	rungs := len(m.Reasoning)
	m.Reasoning = append(m.Reasoning, ai.ReasoningLevel{Effort: ai.EffortMax, Value: "tampered"})
	again, err := catalog.Model("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("catalog.Model: %v", err)
	}
	if len(again.Reasoning) != rungs {
		t.Errorf("the catalog ladder grew to %d rungs; a caller edited the shared table", len(again.Reasoning))
	}
}
