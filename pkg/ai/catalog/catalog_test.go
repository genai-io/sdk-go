package catalog

import (
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/provider"
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
// on and nothing enforces: a duplicated Order shuffles a picker, and a mistyped
// Compat is read as the zero value and ignored.
func TestCatalogInvariants(t *testing.T) {
	seenID := map[string]string{}
	seenOrder := map[int]string{}

	for _, v := range vendors {
		t.Run(v.ID, func(t *testing.T) {
			checkVendorIdentity(t, v, seenID, seenOrder)
			checkVendorProtocol(t, v)
			checkVendorCredential(t, v)
			checkVendorEndpoint(t, v)

			if _, err := time.Parse("2006-01-02", v.Verified); err != nil {
				t.Errorf("Verified = %q, want a YYYY-MM-DD date: %v", v.Verified, err)
			}
		})
	}
}

func checkVendorIdentity(t *testing.T, v Vendor, seenID map[string]string, seenOrder map[int]string) {
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
	if first, dup := seenOrder[v.Order]; dup {
		t.Errorf("Order %d is already used by %s; two vendors at one position sort arbitrarily", v.Order, first)
	}
	seenOrder[v.Order] = v.ID
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
	if len(v.KeyEnv) == 0 && v.Note == "" && v.Deployment == nil {
		t.Error("no KeyEnv and no Note: a vendor that takes no API key has to say what it takes instead")
	}
	for _, name := range v.KeyEnv {
		if name != strings.ToUpper(name) || strings.TrimSpace(name) != name {
			t.Errorf("KeyEnv %q is not a plain environment variable name", name)
		}
	}
	// The two halves of a deployment have to agree: DeploymentEnv is what a
	// caller is told to set and Deployment is what reads it.
	if v.NeedsDeployment() != (v.Deployment != nil) {
		t.Errorf("DeploymentEnv %v and Deployment %v disagree about whether this vendor needs one",
			v.DeploymentEnv, v.Deployment != nil)
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

// TestStaleReportsUnverifiedEntries covers the freshness check the Verified
// column exists for. Nothing calls it at runtime; it is the tool a maintainer
// runs to find rows that have gone unchecked.
func TestStaleReportsUnverifiedEntries(t *testing.T) {
	newest := ""
	for _, v := range All() {
		if v.Verified > newest {
			newest = v.Verified
		}
	}
	now, err := time.Parse("2006-01-02", newest)
	if err != nil {
		t.Fatalf("parsing the newest Verified date %q: %v", newest, err)
	}

	if got := Stale(now, 365*24*time.Hour); len(got) != 0 {
		t.Errorf("Stale reported %d entries as a year old on the day of the last sweep", len(got))
	}
	// A day past the newest sweep, with no tolerance, every entry is stale.
	all := Stale(now.AddDate(0, 0, 1), 0)
	if len(all) != len(vendors) {
		t.Errorf("Stale with no tolerance = %d entries, want all %d", len(all), len(vendors))
	}
	// Oldest first, so a maintainer reads the list top down.
	for i := 1; i < len(all); i++ {
		if all[i-1].Verified > all[i].Verified {
			t.Errorf("Stale is not sorted oldest first: %q before %q", all[i-1].Verified, all[i].Verified)
		}
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

// TestAVendorProviderKeepsWhatTheCatalogKnows pins that a provider built from a
// vendor hands back a model already carrying the vendor's protocol facts.
func TestAVendorProviderKeepsWhatTheCatalogKnows(t *testing.T) {
	v, ok := Find("deepseek")
	if !ok {
		t.Fatal("no deepseek vendor")
	}
	got, listed := v.Provider(provider.Config{}).Model("deepseek-v9")
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
