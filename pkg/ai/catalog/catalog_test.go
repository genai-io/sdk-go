package catalog

import (
	"errors"
	"net/url"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"
)

// protocols is what each wire protocol expects as its Compat, mirroring the
// registry inside package ai. It is written out here rather than read from
// there because a vendor row naming a protocol this SDK does not serve is the
// failure worth catching, and an empty lookup would pass silently.
var protocols = map[ai.API]string{
	ai.APIAnthropicMessages: "ai.AnthropicCompat",
	ai.APIAnthropicVertex:   "ai.AnthropicCompat",
	ai.APIOpenAIChat:        "ai.OpenAIChatCompat",
	ai.APIOpenAIResponses:   "ai.OpenAIResponsesCompat",
	ai.APIGoogleGenAI:       "ai.GoogleCompat",
	ai.APIGoogleVertex:      "ai.GoogleCompat",
}

// compatName reports which protocol a compat value belongs to, by its concrete
// type. Compat is an any, so nothing but this stops a row from carrying the
// wrong protocol's flags: the driver would read its own type out and find the
// zero value, which looks like an ordinary endpoint rather than a mistake.
func compatName(c any) string {
	switch c.(type) {
	case nil:
		return ""
	case ai.AnthropicCompat:
		return "ai.AnthropicCompat"
	case ai.OpenAIChatCompat:
		return "ai.OpenAIChatCompat"
	case ai.OpenAIResponsesCompat:
		return "ai.OpenAIResponsesCompat"
	case ai.GoogleCompat:
		return "ai.GoogleCompat"
	default:
		return "unknown"
	}
}

// TestCatalogInvariants checks the properties the hand-written table is written
// on and nothing enforces: a mistyped Compat is read as the zero value and
// ignored.
func TestCatalogInvariants(t *testing.T) {
	seenID := map[string]string{}

	for _, v := range vendors {
		t.Run(v.ID, func(t *testing.T) {
			checkVendorIdentity(t, v, seenID)
			checkVendorProtocol(t, v)
			checkVendorCredential(t, v)
			checkVendorEndpoint(t, v)
		})
	}
}

func checkVendorIdentity(t *testing.T, v Vendor, seenID map[string]string) {
	t.Helper()
	if v.ID == "" {
		t.Error("ID is empty")
	}
	if v.ID != strings.ToLower(v.ID) {
		t.Errorf("ID = %q, want it lowercase: a reference is matched case-insensitively but printed as written", v.ID)
	}
	// A slash separates the vendor from the model in a reference, so an ID
	// containing one could never be resolved back.
	if strings.ContainsAny(v.ID, "/ ") {
		t.Errorf("ID = %q, want no slash or space", v.ID)
	}
	if first, dup := seenID[v.ID]; dup {
		t.Errorf("ID %q is already used by %s", v.ID, first)
	}
	seenID[v.ID] = v.DisplayName
	if v.DisplayName == "" {
		t.Error("DisplayName is empty")
	}
}

func checkVendorProtocol(t *testing.T, v Vendor) {
	t.Helper()
	want, known := protocols[v.API]
	if !known {
		t.Fatalf("API = %q, which no driver in this SDK serves", v.API)
	}
	if got := compatName(v.Compat); got != "" && got != want {
		t.Errorf("Compat is %s but the endpoint speaks %s, which reads %s; "+
			"the mismatch is silently ignored rather than reported", got, v.API, want)
	}
}

func checkVendorCredential(t *testing.T, v Vendor) {
	t.Helper()
	// A vendor with no credential variable is local, browser-signed-in, or on a
	// cloud's own credentials. The row has to say which; nothing else can.
	if len(v.KeyEnv) == 0 && v.Note == "" && v.VertexEnv == (VertexEnv{}) {
		t.Error("no KeyEnv and no Note: a vendor that takes no API key has to say what it takes instead")
	}
	for _, name := range v.KeyEnv {
		if name != strings.ToUpper(name) || strings.TrimSpace(name) != name {
			t.Errorf("KeyEnv %q is not a plain environment variable name", name)
		}
	}
	// A Vertex protocol cannot run without a project, and nothing else reads one.
	vertex := v.API == ai.APIAnthropicVertex || v.API == ai.APIGoogleVertex
	if vertex != (v.VertexEnv.Project != "") {
		t.Errorf("VertexEnv %+v disagrees with API %s about whether this vendor runs on Vertex", v.VertexEnv, v.API)
	}
}

func checkVendorEndpoint(t *testing.T, v Vendor) {
	t.Helper()
	if v.BaseURL != "" {
		u, err := url.Parse(v.BaseURL)
		if err != nil {
			t.Errorf("BaseURL %q does not parse: %v", v.BaseURL, err)
		} else if u.Scheme == "" || u.Host == "" {
			t.Errorf("BaseURL = %q, want an absolute URL with a scheme and host", v.BaseURL)
		}
	}
	// Every vendor reached with a key has to be redirectable: a gateway, a proxy,
	// a regional host and a recorded test all depend on it.
	if len(v.KeyEnv) > 0 && v.BaseURLEnv == "" {
		t.Error("has a credential variable but no BaseURLEnv, so its host cannot be redirected")
	}
	if v.RequiresBaseURL {
		if v.BaseURLEnv == "" {
			t.Error("RequiresBaseURL with no BaseURLEnv names no variable to set")
		}
		if v.BaseURL != "" {
			t.Errorf("RequiresBaseURL with a default BaseURL %q: one of the two is wrong", v.BaseURL)
		}
	}
}

// TestAliasesPointAtRows keeps the redirection honest: an alias for a row that
// no longer exists resolves to nothing and is worse than no alias at all,
// because it reads as though the old spelling still works.
func TestAliasesPointAtRows(t *testing.T) {
	for from, to := range aliases {
		if from != strings.ToLower(from) {
			t.Errorf("alias %q is not lowercase; it is looked up lowercased", from)
		}
		if _, ok := row(from); ok {
			t.Errorf("alias %q is also a real vendor ID, so the alias is dead", from)
		}
		if _, ok := row(to); !ok {
			t.Errorf("alias %q points at %q, which is not a vendor", from, to)
		}
	}
}

// TestAModelReferenceResolves covers the ways a reference is written, including
// the vendor spelling that was corrected after it had been published.
func TestAModelReferenceResolves(t *testing.T) {
	tests := map[string]struct {
		ref        string
		wantVendor string
		wantErr    bool
	}{
		"qualified":           {ref: "minimax/MiniMax-M3", wantVendor: "minimax"},
		"qualified, any case": {ref: "MiniMax/MiniMax-M3", wantVendor: "minimax"},
		"the misspelt vendor": {ref: "minmax/MiniMax-M3", wantVendor: "minimax"},
		"unlisted, qualified": {ref: "minimax/MiniMax-M9", wantVendor: "minimax"},
		// The catalog lists no models, so nothing can be inferred from one.
		"bare":                 {ref: "deepseek-v4-pro", wantErr: true},
		"unknown vendor":       {ref: "nobody/some-model", wantErr: true},
		"empty":                {ref: "  ", wantErr: true},
		"vendor with no model": {ref: "deepseek/", wantErr: true},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			m, err := Model(tc.ref)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Model(%q) = %v, want an error", tc.ref, m)
				}
				return
			}
			if err != nil {
				t.Fatalf("Model(%q): %v", tc.ref, err)
			}
			if m.Vendor != tc.wantVendor {
				t.Errorf("Vendor = %q, want %q", m.Vendor, tc.wantVendor)
			}
		})
	}
}

// TestDecoratedModelsDoNotAliasTheTable: a caller that edits what it was
// handed must not edit the package-level table every other caller reads.
func TestDecoratedModelsDoNotAliasTheTable(t *testing.T) {
	first, err := Model("copilot/gpt-5.5")
	if err != nil {
		t.Fatal(err)
	}
	first.Headers["Editor-Version"] = "tampered"

	second, err := Model("copilot/gpt-5.5")
	if err != nil {
		t.Fatal(err)
	}
	if second.Headers["Editor-Version"] == "tampered" {
		t.Error("editing a resolved model changed the table it came from")
	}
}

// TestAVendorEndpointKeepsWhatTheCatalogKnows pins that an endpoint built from
// a vendor hands back a model already carrying the vendor's protocol facts.
func TestAVendorEndpointKeepsWhatTheCatalogKnows(t *testing.T) {
	v, ok := Find("deepseek")
	if !ok {
		t.Fatal("no deepseek vendor")
	}
	got, listed := v.Endpoint(endpoint.Config{}).Model("deepseek-v9")
	if listed {
		t.Fatal("the model was reported as listed")
	}
	if got.Vendor != "deepseek" || got.API != v.API || got.BaseURL != v.BaseURL {
		t.Errorf("model = %s/%s at %q, want the vendor's identity, protocol and host", got.Vendor, got.API, got.BaseURL)
	}
	if ai.CompatOf[ai.OpenAIChatCompat](got).Thinking != ai.ThinkingEffortOrDisable {
		t.Errorf("Compat = %#v, want the vendor's", got.Compat)
	}
}

// TestAVendorReadsItsVariables covers the three lookups and the one error
// they share, through a getenv that is not the process environment.
func TestAVendorReadsItsVariables(t *testing.T) {
	env := map[string]string{
		"AZURE_OPENAI_ENDPOINT": "https://acme.openai.azure.com/",
		"DEEPSEEK_API_KEY":      " sk-1 ",
		"GOOGLE_CLOUD_PROJECT":  "proj",
	}
	getenv := func(name string) string { return env[name] }
	find := func(id string) Vendor {
		v, ok := Find(id)
		if !ok {
			t.Fatalf("no vendor %s", id)
		}
		return v
	}
	missing := func(err error, want string) {
		t.Helper()
		var m *MissingEnvError
		if !errors.As(err, &m) || len(m.EnvVars) == 0 || m.EnvVars[0] != want {
			t.Errorf("err = %v, want a MissingEnvError naming %s", err, want)
		}
	}

	if key, err := find("deepseek").KeyFromEnv(getenv); key != "sk-1" || err != nil {
		t.Errorf("deepseek key = %q, %v", key, err)
	}
	_, err := find("openai").KeyFromEnv(getenv)
	missing(err, "OPENAI_API_KEY")
	if key, err := find("ollama").KeyFromEnv(getenv); key != "" || err != nil {
		t.Errorf("ollama key = %q, %v; want none and no error", key, err)
	}

	if url, err := find("azure-openai").BaseURLFromEnv(getenv); url != "https://acme.openai.azure.com/openai/v1" || err != nil {
		t.Errorf("azure base URL = %q, %v", url, err)
	}
	_, err = find("bedrock-openai").BaseURLFromEnv(getenv)
	missing(err, "AWS_BEDROCK_BASE_URL")
	if url, _ := find("deepseek").BaseURLFromEnv(getenv); url != "https://api.deepseek.com" {
		t.Errorf("deepseek base URL = %q, want the default", url)
	}

	if cfg, err := find("google-vertex").VertexFromEnv(getenv); err != nil || cfg.Project != "proj" {
		t.Errorf("google-vertex = %+v, %v", cfg, err)
	}
	_, err = find("anthropic-vertex").VertexFromEnv(getenv)
	missing(err, "ANTHROPIC_VERTEX_PROJECT_ID")
	if cfg, err := find("deepseek").VertexFromEnv(getenv); cfg != nil || err != nil {
		t.Errorf("deepseek vertex = %+v, %v; want none", cfg, err)
	}
}

func TestBaseURLFromEnvAppliesTheOverrideAndItsSuffix(t *testing.T) {
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
			v, _ := Find(tc.vendorID)
			got, err := v.BaseURLFromEnv(func(string) string { return tc.given })
			if err != nil || got != tc.want {
				t.Errorf("BaseURLFromEnv = %q, %v, want %q", got, err, tc.want)
			}
		})
	}
}

func TestKeyFromEnvPrefersTheFirstVariable(t *testing.T) {
	v, _ := Find("google")
	env := map[string]string{v.KeyEnv[1]: "second"}
	if key, _ := v.KeyFromEnv(func(n string) string { return env[n] }); key != "second" {
		t.Errorf("key = %q, want the second variable when the first is unset", key)
	}
	env[v.KeyEnv[0]] = "first"
	if key, _ := v.KeyFromEnv(func(n string) string { return env[n] }); key != "first" {
		t.Errorf("key = %q, want the preferred variable to win", key)
	}
}
