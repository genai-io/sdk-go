package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
)

func baselineProvider(fetch func(context.Context, *llm.Provider) ([]llm.Model, error)) *llm.Provider {
	return llm.NewProvider(llm.ProviderConfig{
		ID:   "acme",
		Name: "Acme",
		API:  "test",
		Models: []llm.Model{
			{ID: "acme-pro", Name: "Acme Pro", ContextWindow: 200_000, MaxOutput: 8_192,
				Pricing:   llm.Pricing{Currency: llm.USD, Input: 1, Output: 2},
				Compat:    llm.OpenAIChatCompat{Thinking: llm.ThinkingEffort},
				Reasoning: fullLadder},
			{ID: "acme-lite", Name: "Acme Lite", ContextWindow: 32_000},
		},
		Fetch: fetch,
	})
}

// Reading the list is synchronous and cannot fail, so a picker renders before
// any network call has happened.
func TestProviderModelsBeforeRefresh(t *testing.T) {
	p := baselineProvider(nil)
	models := p.Models()
	if len(models) != 2 {
		t.Fatalf("got %d models, want the 2 baseline entries", len(models))
	}
	if _, ok := p.Model("acme-pro"); !ok {
		t.Error("acme-pro not found")
	}
}

// A listing carries what the endpoint publishes — for most OpenAI-compatible
// vendors an ID and nothing else. Replacing the baseline entry wholesale would
// drop its pricing, ladder and protocol quirks, and a model stripped of its
// quirks stops working.
func TestRefreshMergesFieldByField(t *testing.T) {
	p := baselineProvider(func(context.Context, *llm.Provider) ([]llm.Model, error) {
		return []llm.Model{
			// The endpoint knows a bigger window than the catalog did, and
			// nothing else.
			{ID: "acme-pro", Name: "acme-pro", ContextWindow: 400_000},
			// A model the catalog has never heard of.
			{ID: "acme-max", Name: "Acme Max"},
		}, nil
	})
	if err := p.Refresh(context.Background()); err != nil {
		t.Fatalf("Refresh: %v", err)
	}

	pro, ok := p.Model("acme-pro")
	if !ok {
		t.Fatal("acme-pro vanished after refresh")
	}
	if pro.ContextWindow != 400_000 {
		t.Errorf("ContextWindow = %d, want the endpoint's 400000", pro.ContextWindow)
	}
	if !pro.Pricing.Known() {
		t.Error("pricing was lost: the endpoint does not report it, so the baseline must keep it")
	}
	if llm.CompatOf[llm.OpenAIChatCompat](pro).Thinking != llm.ThinkingEffort {
		t.Error("protocol quirks were lost, which would silently break the model")
	}
	if !pro.Reasons() {
		t.Error("the reasoning ladder was lost")
	}
	// A name the endpoint echoes back as the bare ID is not a real name.
	if pro.Name != "Acme Pro" {
		t.Errorf("Name = %q, want the catalog's", pro.Name)
	}

	if _, ok := p.Model("acme-max"); !ok {
		t.Error("a model only the endpoint knows should be added")
	}
	if _, ok := p.Model("acme-lite"); !ok {
		t.Error("a model the listing omitted should survive")
	}
}

// A provider that went down keeps serving what it last knew.
func TestRefreshFailureKeepsPreviousList(t *testing.T) {
	calls := 0
	p := baselineProvider(func(context.Context, *llm.Provider) ([]llm.Model, error) {
		calls++
		if calls == 1 {
			return []llm.Model{{ID: "acme-max"}}, nil
		}
		return nil, errors.New("endpoint down")
	})

	if err := p.Refresh(context.Background()); err != nil {
		t.Fatalf("first Refresh: %v", err)
	}
	if err := p.Refresh(context.Background()); err == nil {
		t.Fatal("second Refresh should have failed")
	}
	if _, ok := p.Model("acme-max"); !ok {
		t.Error("the previous listing was discarded on failure")
	}
}

// An unlisted ID is nearly always a model newer than the catalog, so it still
// resolves — carrying the provider's protocol and endpoint.
func TestProviderModelFallsBackToProtocol(t *testing.T) {
	p := baselineProvider(nil)
	m, known := p.Model("acme-v9-unreleased")
	if known {
		t.Error("an unlisted model should report as unknown")
	}
	if m.API != "test" || m.Vendor != "acme" {
		t.Errorf("unlisted model = %+v, want the provider's protocol and identity", m)
	}
	if m.ContextWindow != 0 {
		t.Errorf("ContextWindow = %d, want 0 rather than a guess", m.ContextWindow)
	}
}

// One dead endpoint must not empty the whole list, so the fan-out reports
// per-provider failures instead of returning a single error.
func TestProvidersRefreshIsBestEffort(t *testing.T) {
	good := llm.NewProvider(llm.ProviderConfig{ID: "good", API: "test",
		Fetch: func(context.Context, *llm.Provider) ([]llm.Model, error) {
			return []llm.Model{{ID: "good-1"}}, nil
		}})
	bad := llm.NewProvider(llm.ProviderConfig{ID: "bad", API: "test",
		Fetch: func(context.Context, *llm.Provider) ([]llm.Model, error) {
			return nil, errors.New("down")
		}})

	set := llm.NewProviders(good, bad)
	result := set.Refresh(context.Background())

	if result.OK() {
		t.Fatal("expected the dead provider to be reported")
	}
	if _, failed := result.Errors["bad"]; !failed {
		t.Errorf("Errors = %v, want an entry for bad", result.Errors)
	}
	if _, failed := result.Errors["good"]; failed {
		t.Errorf("good was reported as failed: %v", result.Errors["good"])
	}
	if len(set.Models()) != 1 {
		t.Errorf("models = %+v, want the healthy provider's list", set.Models())
	}
}

// A model ID may itself contain a slash, so only the first segment names the
// provider.
func TestProvidersModelReference(t *testing.T) {
	p := llm.NewProvider(llm.ProviderConfig{ID: "hub", API: "test",
		Models: []llm.Model{{ID: "vendor/model-1", ContextWindow: 1000}}})
	set := llm.NewProviders(p)

	m, ok := set.Model("hub/vendor/model-1")
	if !ok || m.ContextWindow != 1000 {
		t.Errorf("Model = %+v, %v", m, ok)
	}
	if _, ok := set.Model("nosuch/model"); ok {
		t.Error("an unknown provider should not resolve")
	}
}

func TestProviderConfigLayersOntoModel(t *testing.T) {
	p := llm.NewProvider(llm.ProviderConfig{
		ID: "acme", API: "test", APIKey: "sk-1", BaseURL: "https://acme.test",
		Headers: map[string]string{"X-Tenant": "t1"},
	})
	cfg := p.Config(llm.Model{ID: "m", API: "test"})
	if cfg.APIKey != "sk-1" || cfg.Endpoint() != "https://acme.test" {
		t.Errorf("config = %+v", cfg)
	}
	if cfg.Headers["X-Tenant"] != "t1" {
		t.Errorf("headers = %v", cfg.Headers)
	}
}
