package provider

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// TestMergeListing pins the rule the whole package exists for: the host is
// authoritative about what exists and about any figure it reported, and the
// catalog fills the rest.
func TestMergeListing(t *testing.T) {
	base := ai.Model{
		ID: "acme-pro", Name: "Acme Pro",
		ContextWindow: 100, MaxOutput: 10,
		Input:     []ai.Modality{ai.ModalityText, ai.ModalityImage},
		Reasoning: []ai.ReasoningLevel{{Effort: ai.EffortHigh, Value: "high"}},
		Pricing:   ai.Pricing{Currency: ai.USD, Input: 1, Output: 2},
		Compat:    ai.OpenAIChatCompat{Thinking: ai.ThinkingEffort},
	}

	tests := map[string]struct {
		live  ai.Model
		check func(t *testing.T, got ai.Model)
	}{
		"a figure the host reported wins": {
			live: ai.Model{ID: "acme-pro", ContextWindow: 999},
			check: func(t *testing.T, got ai.Model) {
				if got.ContextWindow != 999 {
					t.Errorf("ContextWindow = %d, want the host's 999", got.ContextWindow)
				}
			},
		},
		"a figure the host omitted keeps the catalog's": {
			live: ai.Model{ID: "acme-pro"},
			check: func(t *testing.T, got ai.Model) {
				if got.ContextWindow != 100 || got.MaxOutput != 10 {
					t.Errorf("limits = %d/%d, want the baseline's 100/10", got.ContextWindow, got.MaxOutput)
				}
				if len(got.Reasoning) != 1 {
					t.Errorf("Reasoning = %v, want the baseline's ladder", got.Reasoning)
				}
				if !got.Pricing.Known() {
					t.Error("the baseline's rate card was dropped")
				}
			},
		},
		"a name that is only the ID again is not a name": {
			live: ai.Model{ID: "acme-pro", Name: "acme-pro"},
			check: func(t *testing.T, got ai.Model) {
				if got.Name != "Acme Pro" {
					t.Errorf("Name = %q, want the catalog's; a listing that echoes the ID has said nothing", got.Name)
				}
			},
		},
		"a protocol quirk is never in a listing, so it survives": {
			live: ai.Model{ID: "acme-pro", ContextWindow: 5},
			check: func(t *testing.T, got ai.Model) {
				if ai.CompatOf[ai.OpenAIChatCompat](got).Thinking != ai.ThinkingEffort {
					t.Error("Compat was lost; a model stripped of its quirks stops working")
				}
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			tc.check(t, MergeListing(base, tc.live))
		})
	}
}

func TestMergeListingDoesNotAliasEitherSide(t *testing.T) {
	base := ai.Model{ID: "acme", Input: []ai.Modality{ai.ModalityText}}
	live := ai.Model{ID: "acme", Reasoning: []ai.ReasoningLevel{{Effort: ai.EffortHigh}}}

	got := MergeListing(base, live)
	got.Input[0] = "tampered"
	got.Reasoning[0].Effort = "tampered"

	if base.Input[0] == "tampered" {
		t.Error("editing the merged model changed the baseline it came from")
	}
	if live.Reasoning[0].Effort == "tampered" {
		t.Error("editing the merged model changed the listing it came from")
	}
}

// TestModelResolvesAnUnlistedID pins that an ID the endpoint serves but the
// baseline does not list still comes back resolved, not bare.
func TestModelResolvesAnUnlistedID(t *testing.T) {
	p := New(Config{
		ID:  "acme",
		API: ai.APIOpenAIChat,
		// Stands in for a catalog vendor: it sizes a model from its ID and
		// leaves anything already stated alone.
		Resolve: func(m ai.Model) ai.Model {
			if strings.HasPrefix(m.ID, "acme-") && m.ContextWindow == 0 {
				m.ContextWindow = 128_000
				m.Reasoning = []ai.ReasoningLevel{{Effort: ai.EffortHigh, Value: "high"}}
			}
			return m
		},
	})

	got, listed := p.Model("acme-brand-new")
	if listed {
		t.Fatal("the model was reported as listed")
	}
	if got.ContextWindow != 128_000 {
		t.Errorf("ContextWindow = %d, want the resolver's 128000", got.ContextWindow)
	}
	if len(got.Reasoning) != 1 {
		t.Errorf("Reasoning = %v, want the resolver's ladder", got.Reasoning)
	}
	if got.Vendor != "acme" || got.API != ai.APIOpenAIChat {
		t.Errorf("model = %s/%s, want the provider's identity and protocol", got.Vendor, got.API)
	}
}

// TestAListedFigureBeatsTheResolver keeps the resolver in its place: it fills
// what is unknown, and what the host actually reported is not unknown.
func TestAListedFigureBeatsTheResolver(t *testing.T) {
	fill := func(m ai.Model) ai.Model {
		if m.ContextWindow == 0 {
			m.ContextWindow = 1
		}
		if m.MaxOutput == 0 {
			m.MaxOutput = 7
		}
		return m
	}
	p := New(Config{
		ID:      "acme",
		API:     ai.APIOpenAIChat,
		Resolve: fill,
		Fetch: func(context.Context, *Provider) ([]ai.Model, error) {
			return []ai.Model{{ID: "acme-pro", ContextWindow: 900}}, nil
		},
	})
	if err := p.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	got, _ := p.Model("acme-pro")
	if got.ContextWindow != 900 {
		t.Errorf("ContextWindow = %d, want the host's 900", got.ContextWindow)
	}
	if got.MaxOutput != 7 {
		t.Errorf("MaxOutput = %d, want the resolver's, since the host stated none", got.MaxOutput)
	}
}

// TestRefreshFailureLeavesThePreviousListing is why Models never fails: a
// picker has to keep rendering when the endpoint goes down.
func TestRefreshFailureLeavesThePreviousListing(t *testing.T) {
	fail := false
	boom := errors.New("endpoint is down")
	p := New(Config{
		ID:  "acme",
		API: ai.APIOpenAIChat,
		Fetch: func(context.Context, *Provider) ([]ai.Model, error) {
			if fail {
				return nil, boom
			}
			return []ai.Model{{ID: "acme-pro", ContextWindow: 900}}, nil
		},
	})

	if err := p.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	fail = true
	if err := p.Refresh(t.Context()); !errors.Is(err, boom) {
		t.Fatalf("Refresh err = %v, want the fetch failure", err)
	}

	models := p.Models()
	if len(models) != 1 || models[0].ContextWindow != 900 {
		t.Errorf("Models = %v, want what the last successful refresh knew", models)
	}
}

// TestRefreshWithNothingToAsk covers the provider that has neither a fetcher
// nor a protocol: there is nothing to ask, which is not a failure.
func TestRefreshWithNothingToAsk(t *testing.T) {
	if err := New(Config{ID: "acme"}).Refresh(t.Context()); err != nil {
		t.Errorf("Refresh = %v, want nil for a provider with no protocol to ask", err)
	}
}

func TestModelsMergesTheListingOverTheBaseline(t *testing.T) {
	p := New(Config{
		ID: "acme",
		Models: []ai.Model{
			{ID: "acme-pro", Name: "Acme Pro", API: ai.APIOpenAIChat, ContextWindow: 100, MaxOutput: 10},
		},
		Fetch: func(context.Context, *Provider) ([]ai.Model, error) {
			return []ai.Model{
				{ID: "ACME-PRO", ContextWindow: 900},
				{ID: "acme-new"},
			}, nil
		},
	})
	if err := p.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}

	models := p.Models()
	if len(models) != 2 {
		t.Fatalf("Models = %d entries, want the baseline plus the one model only the host knows", len(models))
	}
	// Matched case-insensitively: a host that shouts its IDs must not produce
	// a second row for a model already in the table.
	if models[0].ContextWindow != 900 || models[0].MaxOutput != 10 {
		t.Errorf("merged = %d/%d, want the host's window over the baseline's cap",
			models[0].ContextWindow, models[0].MaxOutput)
	}
	if models[1].ID != "acme-new" || models[1].API != ai.APIOpenAIChat {
		t.Errorf("the host-only model came back as %s/%s, want it carrying the provider's protocol",
			models[1].ID, models[1].API)
	}
}

func TestModelsDoNotAliasTheConfig(t *testing.T) {
	baseline := []ai.Model{{ID: "acme-pro", API: ai.APIOpenAIChat, Input: []ai.Modality{ai.ModalityText}}}
	p := New(Config{ID: "acme", Models: baseline})

	// The caller keeps its own builder: editing it after New must not reach
	// inside the provider.
	baseline[0].ID = "tampered"
	if got := p.Models(); got[0].ID != "acme-pro" {
		t.Errorf("ID = %q, want the provider's own snapshot", got[0].ID)
	}
	got := p.Models()
	got[0].Input[0] = "tampered"
	if again := p.Models(); again[0].Input[0] == "tampered" {
		t.Error("editing a returned model changed what the next caller reads")
	}
}

func TestClientNeedsAProtocol(t *testing.T) {
	_, err := New(Config{ID: "acme"}).Client("acme-pro")
	if err == nil {
		t.Fatal("expected a failure for a provider that states no protocol")
	}
	var aiErr *ai.Error
	if !errors.As(err, &aiErr) || aiErr.Kind != ai.KindInvalidRequest {
		t.Errorf("err = %v, want an invalid-request ai.Error", err)
	}
}

func TestConfigForCarriesTheProviderSettings(t *testing.T) {
	p := New(Config{
		ID: "acme", APIKey: "the-key", BaseURL: "https://acme.test",
		Headers: map[string]string{"X-Acme": "1"},
	})
	cfg := p.ConfigFor(ai.Model{ID: "acme-pro"})
	if cfg.APIKey != "the-key" || cfg.BaseURL != "https://acme.test" || cfg.Headers["X-Acme"] != "1" {
		t.Errorf("ConfigFor = %+v, want the provider's credential, endpoint and headers", cfg)
	}
	// The headers are a copy: a driver that edits them must not rewrite what
	// every other model on this provider sends.
	cfg.Headers["X-Acme"] = "tampered"
	if again := p.ConfigFor(ai.Model{ID: "acme-pro"}); again.Headers["X-Acme"] != "1" {
		t.Error("editing a built Config changed the provider's headers")
	}
}

func TestSetGetIsCaseInsensitive(t *testing.T) {
	s := NewSet(New(Config{ID: "DeepSeek", API: ai.APIOpenAIChat}))
	for _, id := range []string{"DeepSeek", "deepseek", "DEEPSEEK", " deepseek "} {
		if _, ok := s.Get(id); !ok {
			t.Errorf("Get(%q) missed; the catalog resolves a vendor ID without regard to case", id)
		}
	}
	if _, ok := s.Get("nobody"); ok {
		t.Error("Get returned a provider that was never added")
	}
	s.Delete("DEEPSEEK")
	if _, ok := s.Get("deepseek"); ok {
		t.Error("Delete did not remove the provider it was asked to")
	}
}

func TestSetModelSplitsOnTheFirstSlashOnly(t *testing.T) {
	s := NewSet(New(Config{
		ID:     "acme",
		API:    ai.APIOpenAIChat,
		Models: []ai.Model{{ID: "org/acme-pro", API: ai.APIOpenAIChat}},
	}))

	tests := map[string]struct {
		ref     string
		wantID  string
		wantAny bool
	}{
		// A model ID may itself contain a slash — a host that qualifies its
		// models by publisher — so only the first segment names the provider.
		"a slash-bearing model ID": {ref: "acme/org/acme-pro", wantID: "org/acme-pro", wantAny: true},
		"an unlisted model":        {ref: "acme/org/acme-new", wantID: "org/acme-new"},
		"no provider named":        {ref: "org/acme-pro"},
		"no slash at all":          {ref: "acme-pro"},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, found := s.Model(tc.ref)
			if tc.wantID == "" {
				if found || got.ID != "" {
					t.Errorf("Model(%q) = %v, %v, want nothing", tc.ref, got, found)
				}
				return
			}
			if got.ID != tc.wantID {
				t.Errorf("ID = %q, want %q", got.ID, tc.wantID)
			}
			if found != tc.wantAny {
				t.Errorf("found = %v, want %v", found, tc.wantAny)
			}
		})
	}
}

func TestSetRefreshFansOutAndReportsEachFailure(t *testing.T) {
	boom := errors.New("down")
	var mu sync.Mutex
	asked := map[string]bool{}

	fetch := func(id string, err error) func(context.Context, *Provider) ([]ai.Model, error) {
		return func(context.Context, *Provider) ([]ai.Model, error) {
			mu.Lock()
			asked[id] = true
			mu.Unlock()
			if err != nil {
				return nil, err
			}
			return []ai.Model{{ID: id + "-pro"}}, nil
		}
	}

	s := NewSet(
		New(Config{ID: "ok-one", API: ai.APIOpenAIChat, Fetch: fetch("ok-one", nil)}),
		New(Config{ID: "ok-two", API: ai.APIOpenAIChat, Fetch: fetch("ok-two", nil)}),
		New(Config{ID: "broken", API: ai.APIOpenAIChat, Fetch: fetch("broken", boom)}),
	)

	result := s.Refresh(t.Context())
	if result.OK() {
		t.Error("OK reported success although one provider failed")
	}
	if len(result.Errors) != 1 || !errors.Is(result.Errors["broken"], boom) {
		t.Errorf("Errors = %v, want the one failure keyed by its provider", result.Errors)
	}
	// One failure must not cancel the others: a set is refreshed so that what
	// can be reached still is.
	for _, id := range []string{"ok-one", "ok-two", "broken"} {
		if !asked[id] {
			t.Errorf("%s was never asked", id)
		}
	}
	if got := len(s.Models()); got != 2 {
		t.Errorf("Models = %d, want the two listings that succeeded", got)
	}
}

func TestSetAllIsSortedAndStable(t *testing.T) {
	s := NewSet(New(Config{ID: "zeta"}), New(Config{ID: "alpha"}), New(Config{ID: "mid"}))
	var ids []string
	for _, p := range s.All() {
		ids = append(ids, p.ID())
	}
	if strings.Join(ids, ",") != "alpha,mid,zeta" {
		t.Errorf("All = %v, want it sorted by ID", ids)
	}
}
