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
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

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
	if cfg.Model.BaseURL != "https://api.deepseek.com" {
		t.Errorf("BaseURL = %q, want the vendor's own host", cfg.Model.BaseURL)
	}
	if cfg.Model.API != ai.APIOpenAIChat {
		t.Error("the model arrived without its protocol")
	}
}

// An override in the environment replaces the host the catalog model carries.
func TestABaseURLOverrideReplacesTheVendorDefault(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")
	t.Setenv("DEEPSEEK_BASE_URL", "https://proxy.test/")

	cfg, err := Config("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.Model.BaseURL != "https://proxy.test" {
		t.Errorf("BaseURL = %q, want the override", cfg.Model.BaseURL)
	}
}

func TestConfigSaysWhichVariableIsMissing(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")

	_, err := Config("deepseek/deepseek-v4-pro")
	var missing *catalog.MissingEnvError
	if !errors.As(err, &missing) {
		t.Fatalf("err = %v, want a MissingEnvError", err)
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
	store := withDefaultStore(t)

	_, err := Config("copilot/gpt-4o")
	var notSignedIn *NotSignedInError
	if !errors.As(err, &notSignedIn) {
		t.Fatalf("err = %v, want a NotSignedInError", err)
	}
	if !strings.Contains(err.Error(), "device code") {
		t.Errorf("err = %v, want it to name the grant to run", err)
	}

	if err := store.Save(Credential{
		Vendor: "copilot", Access: "gho_1", BaseURL: "https://api.enterprise.githubcopilot.com",
	}); err != nil {
		t.Fatal(err)
	}
	cfg, err := Config("copilot/gpt-4o")
	if err != nil {
		t.Fatalf("Config after signing in: %v", err)
	}
	// The endpoint recorded at sign-in wins: an enterprise account's is not
	// the one in the table.
	if cfg.Model.BaseURL != "https://api.enterprise.githubcopilot.com" {
		t.Errorf("BaseURL = %q, want the endpoint the sign-in resolved to", cfg.Model.BaseURL)
	}
	if cfg.APIKey != "" {
		t.Error("a static token was baked into the config; it would go stale mid-session")
	}
	if cfg.HTTPClient == nil {
		t.Error("no client to mint a token with")
	}
}

func TestAvailableReportsWhatCanActuallyBeReached(t *testing.T) {
	store := withDefaultStore(t)

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
	all := catalog.All()
	pos := func(v catalog.Vendor) int {
		return slices.IndexFunc(all, func(c catalog.Vendor) bool { return c.ID == v.ID })
	}
	if !slices.IsSortedFunc(Available(), func(a, b catalog.Vendor) int { return pos(a) - pos(b) }) {
		t.Error("Available is not in display order")
	}
}

func TestEndpointCarriesTheCredentialAndTheProtocol(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")
	t.Setenv("DEEPSEEK_BASE_URL", "")

	p, err := Endpoint("deepseek")
	if err != nil {
		t.Fatalf("Endpoint: %v", err)
	}
	if p.ID() != "deepseek" || p.API() == "" {
		t.Errorf("endpoint = %s/%s, want the vendor's identity and protocol", p.ID(), p.API())
	}
	// The catalog lists no models, so every ID arrives resolved through the
	// resolver the vendor installs.
	m, listed := p.Model("deepseek-v9-pro")
	if listed {
		t.Fatal("an unlisted model was reported as listed")
	}
	if m.API != p.API() || m.BaseURL == "" {
		t.Errorf("model = %+v, want the vendor's protocol and host", m)
	}
}

func TestEndpointRefusesWhatItCannotReach(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, err := Endpoint("deepseek"); err == nil {
		t.Error("built an endpoint with no credential")
	}
	t.Setenv("AZURE_OPENAI_API_KEY", "k")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	if _, err := Endpoint("azure-openai"); err == nil {
		t.Error("built an endpoint with no host")
	}
	var unknown *catalog.UnknownVendorError
	if _, err := Endpoint("nobody"); !errors.As(err, &unknown) {
		t.Errorf("err = %v, want an UnknownVendorError", err)
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
			// Form-encoded, as GitHub answers a token request that does not ask
			// for JSON.
			w.Header().Set("Content-Type", "application/x-www-form-urlencoded")
			_, _ = w.Write([]byte(`access_token=gho_stub&token_type=bearer`))
		case "/exchange":
			editorHeaders = r.Header.Clone()
			_, _ = w.Write([]byte(`{"token":"copilot-session","expires_at":4102444800,"endpoints":{"api":"https://api.enterprise.githubcopilot.test"}}`))
		}
	}))
	defer server.Close()

	flows[copilotStub] = newCopilotFlow(copilotEndpoints{
		device:   server.URL + "/device",
		token:    server.URL + "/token",
		exchange: server.URL + "/exchange",
		api:      "https://api.individual.githubcopilot.test",
	})
	t.Cleanup(func() { delete(flows, copilotStub) })

	store := &memStore{}
	var shown Prompt
	got, err := Login(t.Context(), copilotStub, LoginOptions{
		Store: store,
		Interaction: func(ctx context.Context, p Prompt) error {
			shown = p
			return nil
		},
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
	if got.BaseURL != "https://api.enterprise.githubcopilot.test" {
		t.Errorf("BaseURL = %q, want the one the exchange revealed", got.BaseURL)
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
	if editorHeaders.Get("X-GitHub-Api-Version") != "" {
		t.Error("the exchange was sent the Copilot API's version, which api.github.com reads as its own")
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
	_, err := Login(t.Context(), "deepseek", LoginOptions{Store: &memStore{}})
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Errorf("err = %v, want it to say to set the key variable instead", err)
	}

	var unknown *catalog.UnknownVendorError
	if _, err := Login(t.Context(), "nobody", LoginOptions{Store: &memStore{}}); !errors.As(err, &unknown) {
		t.Errorf("err = %v, want an UnknownVendorError", err)
	}
}
