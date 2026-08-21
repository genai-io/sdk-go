package auth_test

import (
	"errors"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/auth"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
)

func TestKeyPrefersTheFirstVariableSet(t *testing.T) {
	google, ok := catalog.Find("google")
	if !ok {
		t.Fatal("google vendor missing")
	}
	if len(google.KeyEnv) < 2 {
		t.Skip("google no longer lists a fallback variable")
	}

	t.Setenv(google.KeyEnv[1], "second")
	key, from := auth.Key(google)
	if key != "second" || from != google.KeyEnv[1] {
		t.Errorf("key = %q from %q", key, from)
	}

	t.Setenv(google.KeyEnv[0], "first")
	key, from = auth.Key(google)
	if key != "first" || from != google.KeyEnv[0] {
		t.Errorf("key = %q from %q, want the preferred variable to win", key, from)
	}
}

func TestConfigResolvesKeyAndEndpoint(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-test")
	t.Setenv("DEEPSEEK_BASE_URL", "https://proxy.internal/v1/")

	cfg, err := auth.Config("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.APIKey != "sk-test" {
		t.Errorf("APIKey = %q", cfg.APIKey)
	}
	// The trailing slash is trimmed so the SDK does not build a double-slashed
	// path.
	if cfg.Endpoint() != "https://proxy.internal/v1" {
		t.Errorf("Endpoint = %q", cfg.Endpoint())
	}
}

func TestConfigFailsLoudlyWithoutACredential(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")

	_, err := auth.Config("deepseek/deepseek-v4-pro")
	var missing *auth.MissingKeyError
	if !errors.As(err, &missing) {
		t.Fatalf("err = %v, want *MissingKeyError rather than an empty credential", err)
	}
	if len(missing.EnvVars) == 0 {
		t.Error("the error should name the variables to set")
	}
}

func TestKeylessVendorNeedsNoCredential(t *testing.T) {
	cfg, err := auth.Config("ollama/llama4")
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.Endpoint() == "" {
		t.Error("the local endpoint should still be resolved")
	}
}

func TestAvailableIncludesKeylessVendors(t *testing.T) {
	for _, v := range auth.Available() {
		if v.ID == "ollama" {
			return
		}
	}
	t.Error("a vendor needing no credential should always be available")
}

func TestProviderResolvesFromEnvironment(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-test")

	p, err := auth.Provider("deepseek")
	if err != nil {
		t.Fatalf("Provider: %v", err)
	}
	if p.ID() != "deepseek" || p.API() != llm.APIOpenAIChat {
		t.Errorf("provider = %s / %s", p.ID(), p.API())
	}
	// The catalog models are the baseline, available before any network call.
	if len(p.Models()) == 0 {
		t.Error("provider has no baseline models")
	}
	if _, ok := p.Model("deepseek-v4-pro"); !ok {
		t.Error("deepseek-v4-pro missing from the baseline")
	}
	// The credential reached the config used to open a model.
	m, _ := p.Model("deepseek-v4-pro")
	if p.Config(m).APIKey != "sk-test" {
		t.Error("the resolved credential did not reach the model config")
	}
}

func TestProviderFailsWithoutCredential(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")

	_, err := auth.Provider("deepseek")
	var missing *auth.MissingKeyError
	if !errors.As(err, &missing) {
		t.Fatalf("err = %v, want *MissingKeyError", err)
	}
}

func TestProvidersSkipsVendorsWithoutCredentials(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-test")

	set := auth.Providers()
	if _, ok := set.Get("deepseek"); !ok {
		t.Error("a vendor with a credential should be present")
	}
	// Ollama needs none, so it is always usable.
	if _, ok := set.Get("ollama"); !ok {
		t.Error("a keyless vendor should always be present")
	}
	if len(set.Models()) == 0 {
		t.Error("the set reports no models")
	}
}

func TestUnknownVendor(t *testing.T) {
	_, err := auth.Provider("nosuchvendor")
	var unknown *auth.UnknownVendorError
	if !errors.As(err, &unknown) {
		t.Fatalf("err = %v, want *UnknownVendorError", err)
	}
}
