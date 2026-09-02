package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

func vendor(t *testing.T, id string) catalog.Vendor {
	t.Helper()
	v, ok := catalog.Find(id)
	if !ok {
		t.Fatalf("no catalog vendor %q", id)
	}
	return v
}

func TestKeyTakesTheFirstVariableThatIsSet(t *testing.T) {
	v := vendor(t, "google")
	if len(v.KeyEnv) < 2 {
		t.Fatalf("this test needs a vendor with two credential variables; %s has %v", v.ID, v.KeyEnv)
	}

	t.Setenv(v.KeyEnv[0], "")
	t.Setenv(v.KeyEnv[1], "  second  ")
	key, from := Key(v)
	if key != "second" || from != v.KeyEnv[1] {
		t.Errorf("Key = %q from %q, want the second variable, trimmed", key, from)
	}

	t.Setenv(v.KeyEnv[0], "first")
	key, from = Key(v)
	if key != "first" || from != v.KeyEnv[0] {
		t.Errorf("Key = %q from %q, want the preferred variable to win", key, from)
	}

	for _, name := range v.KeyEnv {
		t.Setenv(name, "")
	}
	if key, from = Key(v); key != "" || from != "" {
		t.Errorf("Key = %q from %q, want nothing", key, from)
	}
}

func TestBaseURLAppliesTheOverrideAndItsSuffix(t *testing.T) {
	tests := map[string]struct {
		vendorID string
		given    string
		want     string
	}{
		"unset falls back to the vendor's own host": {
			"deepseek", "", "https://api.deepseek.com"},
		"an override replaces it": {
			"deepseek", "https://proxy.test/v1", "https://proxy.test/v1"},
		"a trailing slash is not part of the URL": {
			"deepseek", "https://proxy.test/v1/", "https://proxy.test/v1"},
		// The suffix is added to the bare URL people copy out of a console.
		"a suffix is added when it is missing": {
			"azure-openai", "https://my-resource.openai.azure.com", "https://my-resource.openai.azure.com/openai/v1"},
		"and not added twice": {
			"azure-openai", "https://my-resource.openai.azure.com/openai/v1", "https://my-resource.openai.azure.com/openai/v1"},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			v := vendor(t, tc.vendorID)
			t.Setenv(v.BaseURLEnv, tc.given)
			if got := BaseURL(v); got != tc.want {
				t.Errorf("BaseURL = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestGoogleCanBeRedirected is the fix for the one key-based vendor whose host
// could not be pointed anywhere else — at a gateway, a proxy, or a recorded
// endpoint in a test.
func TestGoogleCanBeRedirected(t *testing.T) {
	v := vendor(t, "google")
	if v.BaseURLEnv == "" {
		t.Fatal("google states no BaseURLEnv, so its host cannot be redirected")
	}
	t.Setenv(v.BaseURLEnv, "https://gemini-proxy.test")
	if got := BaseURL(v); got != "https://gemini-proxy.test" {
		t.Errorf("BaseURL = %q, want the override", got)
	}
}

// TestDeploymentIsTheTablesBusiness: which variables a vendor's deployment
// needs, and what shape they make, is a fact about the row. This package only
// supplies the lookup.
func TestDeploymentIsTheTablesBusiness(t *testing.T) {
	t.Run("a vendor with no deployment has none", func(t *testing.T) {
		got, err := Deployment(vendor(t, "deepseek"))
		if err != nil || got != nil {
			t.Errorf("Deployment = %v, %v, want nothing", got, err)
		}
	})

	t.Run("the missing setting is named", func(t *testing.T) {
		t.Setenv("ANTHROPIC_VERTEX_PROJECT_ID", "")
		t.Setenv("CLOUD_ML_REGION", "")
		_, err := Deployment(vendor(t, "anthropic-vertex"))
		var missing *MissingKeyError
		if !errors.As(err, &missing) {
			t.Fatalf("err = %v, want a MissingKeyError", err)
		}
		if !slices.Contains(missing.EnvVars, "ANTHROPIC_VERTEX_PROJECT_ID") {
			t.Errorf("err = %v, want it to name the variable to set", err)
		}
	})

	t.Run("what is set becomes the driver's own shape", func(t *testing.T) {
		t.Setenv("ANTHROPIC_VERTEX_PROJECT_ID", " my-project ")
		t.Setenv("CLOUD_ML_REGION", "europe-west1")
		got, err := Deployment(vendor(t, "anthropic-vertex"))
		if err != nil {
			t.Fatal(err)
		}
		cfg, ok := got.(ai.VertexConfig)
		if !ok {
			t.Fatalf("Deployment = %#v, want an ai.VertexConfig", got)
		}
		if cfg.Project != "my-project" || cfg.Region != "europe-west1" {
			t.Errorf("deployment = %+v, want the variables, trimmed", cfg)
		}
	})
}

func TestConfigResolvesCredentialAndEndpoint(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")
	t.Setenv("DEEPSEEK_BASE_URL", "")

	cfg, err := Config("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.APIKey != "sk-deepseek" {
		t.Errorf("APIKey = %q, want the one in the environment", cfg.APIKey)
	}
	if cfg.BaseURL != "https://api.deepseek.com" {
		t.Errorf("BaseURL = %q, want the vendor's own host", cfg.BaseURL)
	}
	if cfg.Model.ContextWindow == 0 {
		t.Error("the model arrived without its limits")
	}
}

func TestConfigSaysWhichVariableIsMissing(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")

	_, err := Config("deepseek/deepseek-v4-pro")
	var missing *MissingKeyError
	if !errors.As(err, &missing) {
		t.Fatalf("err = %v, want a MissingKeyError", err)
	}
	if !strings.Contains(err.Error(), "DEEPSEEK_API_KEY") {
		t.Errorf("err = %v, want it to name the variable", err)
	}
}

func TestConfigRefusesAnUnknownReference(t *testing.T) {
	if _, err := Config("nobody/nothing"); err == nil {
		t.Error("Config resolved a model no vendor lists")
	}
}

// TestConfigForAnInteractiveVendorNeedsASignIn covers the other half of the
// two-kinds-of-vendor split: there is no key to read, so the only question is
// whether somebody has signed in.
func TestConfigForAnInteractiveVendorNeedsASignIn(t *testing.T) {
	store := NewMemoryStore()
	defer withDefaultStore(t, store)()

	_, err := Config("copilot/gpt-4o")
	var notSignedIn *NotSignedInError
	if !errors.As(err, &notSignedIn) {
		t.Fatalf("err = %v, want a NotSignedInError", err)
	}
	if !strings.Contains(err.Error(), "device code") {
		t.Errorf("err = %v, want it to name the grant to run", err)
	}

	if err := store.Save(Credential{
		Vendor: "copilot", Access: "gho_1", Endpoint: "https://api.enterprise.githubcopilot.com",
	}); err != nil {
		t.Fatal(err)
	}
	cfg, err := Config("copilot/gpt-4o")
	if err != nil {
		t.Fatalf("Config after signing in: %v", err)
	}
	// The endpoint recorded at sign-in wins: an enterprise account's is not
	// the one in the table.
	if cfg.BaseURL != "https://api.enterprise.githubcopilot.com" {
		t.Errorf("BaseURL = %q, want the endpoint the sign-in resolved to", cfg.BaseURL)
	}
	if cfg.APIKey != "" {
		t.Error("a static token was baked into the config; it would go stale mid-session")
	}
	if cfg.HTTPClient == nil {
		t.Error("no client to mint a token with")
	}
}

func TestAvailableReportsWhatCanActuallyBeReached(t *testing.T) {
	store := NewMemoryStore()
	defer withDefaultStore(t, store)()

	// Start from nothing: whatever is in the developer's environment must not
	// decide the answer.
	for _, v := range catalog.All() {
		for _, name := range v.KeyEnv {
			t.Setenv(name, "")
		}
		if v.BaseURLEnv != "" {
			t.Setenv(v.BaseURLEnv, "")
		}
	}

	ids := func() []string {
		var out []string
		for _, v := range Available() {
			out = append(out, v.ID)
		}
		return out
	}

	got := ids()
	// A local server needs no credential, so it is always reachable.
	if !slices.Contains(got, "ollama") {
		t.Errorf("Available = %v, want the vendor that needs no credential", got)
	}
	if slices.Contains(got, "deepseek") {
		t.Error("a vendor with no key set was reported as available")
	}
	// A vendor whose host names a tenant's own resource is unusable until it
	// has one, key or no key.
	if slices.Contains(got, "azure-openai") {
		t.Error("a vendor with no endpoint set was reported as available")
	}
	if slices.Contains(got, "copilot") {
		t.Error("an interactive vendor was reported as available with nobody signed in")
	}
	// And one whose models live in a cloud project nobody named.
	t.Setenv("ANTHROPIC_VERTEX_PROJECT_ID", "")
	if slices.Contains(ids(), "anthropic-vertex") {
		t.Error("a vendor with no deployment set was reported as available")
	}
	t.Setenv("ANTHROPIC_VERTEX_PROJECT_ID", "my-project")
	if !slices.Contains(ids(), "anthropic-vertex") {
		t.Error("a vendor with its deployment set was left out")
	}

	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")
	if err := store.Save(Credential{Vendor: "copilot", Access: "gho_1"}); err != nil {
		t.Fatal(err)
	}
	got = ids()
	for _, want := range []string{"deepseek", "copilot"} {
		if !slices.Contains(got, want) {
			t.Errorf("Available = %v, want it to include %q", got, want)
		}
	}
	// In display order, so a picker renders the same list every time.
	if !slices.IsSortedFunc(Available(), func(a, b catalog.Vendor) int { return a.Order - b.Order }) {
		t.Error("Available is not in display order")
	}
}

func TestProviderCarriesTheCredentialAndTheCatalog(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")
	t.Setenv("DEEPSEEK_BASE_URL", "")

	p, err := Provider("deepseek")
	if err != nil {
		t.Fatalf("Provider: %v", err)
	}
	if p.ID() != "deepseek" || p.API() == "" {
		t.Errorf("provider = %s/%s, want the vendor's identity and protocol", p.ID(), p.API())
	}
	if len(p.Models()) == 0 {
		t.Error("the provider was seeded with no models")
	}
	// And an ID the table does not list still arrives resolved, which is what
	// the resolver the vendor installs is for.
	m, listed := p.Model("deepseek-v9-pro")
	if listed {
		t.Fatal("an unlisted model was reported as listed")
	}
	if m.API != p.API() || m.BaseURL == "" {
		t.Errorf("model = %+v, want the vendor's protocol and host", m)
	}
}

func TestProviderRefusesWhatItCannotReach(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, err := Provider("deepseek"); err == nil {
		t.Error("built a provider with no credential")
	}
	var unknown *UnknownVendorError
	if _, err := Provider("nobody"); !errors.As(err, &unknown) {
		t.Errorf("err = %v, want an UnknownVendorError", err)
	}
}

func TestProvidersCoversEveryVendorThatCanBeReached(t *testing.T) {
	defer withDefaultStore(t, NewMemoryStore())()
	for _, v := range catalog.All() {
		for _, name := range v.KeyEnv {
			t.Setenv(name, "")
		}
	}
	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")

	set := Providers()
	if _, ok := set.Get("deepseek"); !ok {
		t.Error("Providers left out a vendor with a usable credential")
	}
	if _, ok := set.Get("openai"); ok {
		t.Error("Providers included a vendor with no credential")
	}
}

// TestLoginStoresWhatTheGrantProduced drives a real flow — the Copilot one,
// pointed at a stub, which is what its endpoints are a struct for.
func TestLoginStoresWhatTheGrantProduced(t *testing.T) {
	var editorHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/device":
			_, _ = w.Write([]byte(`{"device_code":"dc","user_code":"UC-9","verification_uri":"https://github.test/login/device","interval":1,"expires_in":600}`))
		case "/token":
			_, _ = w.Write([]byte(`{"access_token":"gho_stub","token_type":"bearer"}`))
		case "/exchange":
			editorHeaders = r.Header.Clone()
			_, _ = w.Write([]byte(`{"token":"copilot-session","expires_at":4102444800,"endpoints":{"api":"https://api.enterprise.githubcopilot.test"}}`))
		}
	}))
	defer server.Close()

	RegisterFlow(copilotStub, newCopilotFlow(copilotEndpoints{
		device:   server.URL + "/device",
		token:    server.URL + "/token",
		exchange: server.URL + "/exchange",
		api:      "https://api.individual.githubcopilot.test",
	}))

	store := NewMemoryStore()
	var shown oauth.Prompt
	got, err := Login(t.Context(), copilotStub, LoginOptions{
		Store: store,
		Interaction: oauth.InteractionFunc(func(ctx context.Context, p oauth.Prompt) error {
			shown = p
			return nil
		}),
	})
	if err != nil {
		t.Fatalf("Login: %v", err)
	}

	if shown.UserCode != "UC-9" {
		t.Errorf("the person was shown %+v, want the code to type", shown)
	}
	// The long-lived GitHub token is what persists; storing the half-hour
	// Copilot one would mean signing in again every half hour.
	if got.Access != "gho_stub" {
		t.Errorf("Access = %q, want the GitHub token", got.Access)
	}
	if got.Endpoint != "https://api.enterprise.githubcopilot.test" {
		t.Errorf("Endpoint = %q, want the one the exchange revealed", got.Endpoint)
	}
	stored, found, err := store.Load(copilotStub)
	if err != nil || !found {
		t.Fatalf("Load after Login = %v, %v", found, err)
	}
	if stored != got {
		t.Errorf("stored %+v, want what Login returned", stored)
	}
	if editorHeaders.Get("Editor-Version") == "" || editorHeaders.Get("Copilot-Integration-Id") == "" {
		t.Errorf("the exchange was sent %v, want the editor headers the vendor row carries", editorHeaders)
	}

	// And signing out forgets it.
	if err := Logout(copilotStub, store); err != nil {
		t.Fatalf("Logout: %v", err)
	}
	if _, found, _ := store.Load(copilotStub); found {
		t.Error("Logout left the credential behind")
	}
}

func TestLoginRefusesAVendorThatTakesAKey(t *testing.T) {
	_, err := Login(t.Context(), "deepseek", LoginOptions{Store: NewMemoryStore()})
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Errorf("err = %v, want it to say to set the key variable instead", err)
	}

	var unknown *UnknownVendorError
	if _, err := Login(t.Context(), "nobody", LoginOptions{Store: NewMemoryStore()}); !errors.As(err, &unknown) {
		t.Errorf("err = %v, want an UnknownVendorError", err)
	}
}
