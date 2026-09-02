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
// on and nothing enforces: a duplicated Order shuffles a picker, a mistyped
// Compat is read as the zero value and ignored, a ladder with no dialect never
// reaches the wire.
func TestCatalogInvariants(t *testing.T) {
	seenID := map[string]string{}
	seenOrder := map[int]string{}

	for _, v := range vendors {
		t.Run(v.ID, func(t *testing.T) {
			checkVendorIdentity(t, v, seenID, seenOrder)
			checkVendorProtocol(t, v)
			checkVendorCredential(t, v)
			checkVendorEndpoint(t, v)
			checkVendorReasoning(t, v)
			checkVendorModels(t, v)

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
	for _, m := range v.Models {
		if got := compatName(m.Compat); got != "" && got != want {
			t.Errorf("model %s carries %s but the endpoint speaks %s, which reads %s", m.ID, got, v.API, want)
		}
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

func checkVendorReasoning(t *testing.T, v Vendor) {
	t.Helper()
	checkLadder(t, "vendor default", v.Reasoning)
	for _, m := range v.Models {
		checkLadder(t, "model "+m.ID, m.Reasoning)
	}

	// A ladder is only a vocabulary; the dialect is what puts a rung on the wire.
	// On Chat Completions that is Compat.Thinking, whose zero value means "no
	// reasoning switch" — so a ladder without one is dropped, silently.
	if v.API != ai.APIOpenAIChat {
		return
	}
	compat := ai.CompatOf[ai.OpenAIChatCompat](ai.Model{Compat: v.Compat})
	for _, ladder := range append([][]ai.ReasoningLevel{v.Reasoning}, laddersOf(v.Models)...) {
		if len(ladder) > 0 && compat.Thinking == ai.ThinkingNone {
			t.Error("declares a reasoning ladder but states no Compat.Thinking, " +
				"so every rung is dropped before the request is sent")
			return
		}
	}
}

func laddersOf(models []ai.Model) [][]ai.ReasoningLevel {
	out := make([][]ai.ReasoningLevel, 0, len(models))
	for _, m := range models {
		out = append(out, m.Reasoning)
	}
	return out
}

func checkLadder(t *testing.T, what string, ladder []ai.ReasoningLevel) {
	t.Helper()
	defaults := 0
	seen := map[ai.Effort]bool{}
	for _, rung := range ladder {
		if rung.Default {
			defaults++
		}
		if seen[rung.Effort] {
			t.Errorf("%s: repeats the %q rung; the second is unreachable", what, rung.Effort)
		}
		seen[rung.Effort] = true
	}
	if defaults > 1 {
		t.Errorf("%s: %d rungs are marked Default; only the first is ever used, "+
			"so the others say something that is not true", what, defaults)
	}
}

func checkVendorModels(t *testing.T, v Vendor) {
	t.Helper()
	seen := map[string]bool{}
	for _, m := range v.Models {
		if m.ID == "" {
			t.Error("a model row has no ID")
			continue
		}
		key := strings.ToLower(m.ID)
		if seen[key] {
			// Lookup is case-insensitive and takes the first match, so the
			// second row is dead weight that still shows up in a picker.
			t.Errorf("model %q is listed twice; only the first is reachable", m.ID)
		}
		seen[key] = true
		if m.API != "" && m.API != v.API {
			t.Errorf("model %q states API %q, which its vendor overwrites", m.ID, m.API)
		}
		if m.Stage == ai.StageRetired && m.Replacement == "" {
			t.Errorf("model %q is retired but names no replacement, which is the "+
				"only reason to keep listing it", m.ID)
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
		"qualified":            {ref: "minimax/MiniMax-M3", wantVendor: "minimax"},
		"qualified, any case":  {ref: "MiniMax/MiniMax-M3", wantVendor: "minimax"},
		"the misspelt vendor":  {ref: "minmax/MiniMax-M3", wantVendor: "minimax"},
		"unlisted, qualified":  {ref: "minimax/MiniMax-M9", wantVendor: "minimax"},
		"bare and unambiguous": {ref: "deepseek-v4-pro", wantVendor: "deepseek"},
		"bare and unknown":     {ref: "no-such-model", wantErr: true},
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

// TestABareModelNameSkipsVendorsNeedingDeployment pins the rule that keeps one
// alternate hosting vendor from making every model it serves ambiguous.
func TestABareModelNameSkipsVendorsNeedingDeployment(t *testing.T) {
	m, err := Model("claude-opus-5")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Vendor != "anthropic" {
		t.Errorf("Vendor = %q, want the first-party API rather than a deployment-scoped one", m.Vendor)
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

// TestDecoratedModelsDoNotAliasTheTable is the reason decorate clones twice: a
// caller that edits what it was handed must not edit the package-level table
// every other caller reads.
func TestDecoratedModelsDoNotAliasTheTable(t *testing.T) {
	first, err := Model("anthropic/claude-opus-5")
	if err != nil {
		t.Fatal(err)
	}
	first.Reasoning[0].Value = "tampered"
	first.Input[0] = "tampered"

	second, err := Model("anthropic/claude-opus-5")
	if err != nil {
		t.Fatal(err)
	}
	if second.Reasoning[0].Value == "tampered" || second.Input[0] == "tampered" {
		t.Error("editing a resolved model changed the table it came from")
	}
}

// TestTheClaudeLineIsServedFromOnePlace pins what sharing the rows bought: the
// two entries cannot drift apart in what they list, only in what they charge.
func TestTheClaudeLineIsServedFromOnePlace(t *testing.T) {
	first, ok := Find("anthropic")
	if !ok {
		t.Fatal("no anthropic vendor")
	}
	vertex, ok := Find("anthropic-vertex")
	if !ok {
		t.Fatal("no anthropic-vertex vendor")
	}

	live := map[string]bool{}
	for _, m := range first.Models {
		if m.Stage.Available() {
			live[m.ID] = true
		}
	}
	if len(live) != len(vertex.Models) {
		t.Errorf("the first-party API lists %d live models and Vertex %d", len(live), len(vertex.Models))
	}
	for _, m := range vertex.Models {
		if !live[m.ID] {
			t.Errorf("Vertex lists %q, which the first-party API does not", m.ID)
		}
		if m.Pricing.Known() {
			t.Errorf("Vertex model %q carries a rate card; Vertex bills through a Google contract", m.ID)
		}
	}
}

func TestInferAnthropic(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"an unlisted Claude gets the generation's shape": {
			in: ai.Model{ID: "claude-opus-6"}, window: 1_000_000, output: 128_000},
		"a snapshot ID too": {
			in: ai.Model{ID: "claude-sonnet-5@20260101"}, window: 1_000_000, output: 128_000},
		"a stated figure is never overwritten": {
			in: ai.Model{ID: "claude-opus-6", ContextWindow: 42}, window: 42, output: 128_000},
		"something that is not a Claude is left alone": {
			in: ai.Model{ID: "llama4"}, window: 0, output: 0},
	}
	runInfer(t, inferAnthropic, tests)
}

func TestInferOpenAI(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"gpt-5": {in: ai.Model{ID: "gpt-5.5"}, window: 1_050_000, output: 128_000},
		"gpt-6": {in: ai.Model{ID: "gpt-6-terra"}, window: 1_050_000, output: 128_000},
		"a fine-tune of a gpt-5": {
			// The generation is in the middle of the ID, which a prefix never
			// sees; the tuned model has the base model's shape.
			in: ai.Model{ID: "ft:gpt-5.4-2026-01-01:acme::abc123"}, window: 1_050_000, output: 128_000},
		"gpt-4.1":       {in: ai.Model{ID: "gpt-4.1-mini"}, window: 1_047_576, output: 32_768},
		"gpt-4o":        {in: ai.Model{ID: "gpt-4o-mini"}, window: 128_000, output: 16_384},
		"gpt-4 turbo":   {in: ai.Model{ID: "gpt-4-turbo-2024-04-09"}, window: 128_000, output: 4_096},
		"gpt-4":         {in: ai.Model{ID: "gpt-4-0613"}, window: 8_192, output: 8_192},
		"gpt-3.5 turbo": {in: ai.Model{ID: "gpt-3.5-turbo-0125"}, window: 16_385, output: 4_096},
		"o3":            {in: ai.Model{ID: "o3-mini"}, window: 200_000, output: 100_000},
		// A point release nobody has published must not be sized at the original
		// GPT-4's 8k, which is wrong by two orders.
		"an unpublished gpt-4 point release": {in: ai.Model{ID: "gpt-4.5-preview"}},
		"an ID from another vendor entirely": {in: ai.Model{ID: "openai.gpt-oss-120b-1:0"}},
	}
	runInfer(t, inferOpenAI, tests)

	reasoning := map[string]struct {
		id   string
		want []ai.ReasoningLevel
	}{
		"a reasoning generation gets the ladder":       {id: "gpt-5.5", want: openAIEfforts},
		"a generation that does not reason says so":    {id: "gpt-4o", want: noReasoning},
		"an unrecognised ID is left saying nothing":    {id: "some-new-thing", want: nil},
		"a stated ladder is never replaced by a guess": {id: "gpt-4o", want: nil},
	}
	for name, tc := range reasoning {
		t.Run(name, func(t *testing.T) {
			in := ai.Model{ID: tc.id}
			if tc.want == nil && strings.HasPrefix(tc.id, "gpt-4o") {
				// The stated-ladder case: hand it a ladder and expect it back.
				in.Reasoning = openAIEfforts
				got := inferOpenAI(in)
				if len(got.Reasoning) != len(openAIEfforts) {
					t.Errorf("Reasoning was replaced: %v", got.Reasoning)
				}
				return
			}
			got := inferOpenAI(in).Reasoning
			if tc.want == nil {
				if got != nil {
					t.Errorf("Reasoning = %v, want nil: an unrecognised ID knows nothing about "+
						"reasoning, which is not the same as knowing there is none", got)
				}
				return
			}
			if len(got) != len(tc.want) {
				t.Errorf("Reasoning = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestInferGoogle(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"gemini 3":                  {in: ai.Model{ID: "gemini-3.9-pro"}, window: 1_048_576, output: 65_536},
		"gemini 2.5":                {in: ai.Model{ID: "gemini-2.5-pro"}, window: 1_048_576, output: 65_536},
		"a generation nobody sized": {in: ai.Model{ID: "gemini-1.0-pro"}},
		"something else entirely":   {in: ai.Model{ID: "text-bison"}},
	}
	runInfer(t, inferGoogle, tests)

	// 2.5 predates the thinking level and still takes a budget, which is a
	// different request field, not a different number.
	old := inferGoogle(ai.Model{ID: "gemini-2.5-flash"})
	if _, ok := old.Compat.(ai.GoogleCompat); !ok {
		t.Errorf("Compat = %#v, want a GoogleCompat with no thinking level", old.Compat)
	}
	if ai.CompatOf[ai.GoogleCompat](old).ThinkingLevel {
		t.Error("gemini-2.5 was given the thinking level, which it does not take")
	}
	if len(old.Reasoning) != len(budgetLadder) {
		t.Errorf("Reasoning = %v, want the budget ladder", old.Reasoning)
	}
	// A Gemini 3 row states its own dialect on the vendor, so Infer leaves it.
	if got := inferGoogle(ai.Model{ID: "gemini-3.9-pro"}); got.Compat != nil || got.Reasoning != nil {
		t.Errorf("gemini-3 was given %v/%v, which the vendor already states", got.Compat, got.Reasoning)
	}
}

func TestInferMoonshot(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"a generation":   {in: ai.Model{ID: "kimi-k3-turbo"}, window: 1_048_576},
		"the one before": {in: ai.Model{ID: "kimi-k2-0905"}, window: 262_144},
		// The suffix is a whole token: "8k" lives inside "128k", so a substring
		// test sizes a 128k model at 8k.
		"a 128k suffix":             {in: ai.Model{ID: "moonshot-v1-128k"}, window: 131_072, output: 8_192},
		"a 32k suffix":              {in: ai.Model{ID: "moonshot-v1-32k"}, window: 32_768, output: 8_192},
		"an 8k suffix":              {in: ai.Model{ID: "moonshot-v1-8k"}, window: 8_192, output: 3_000},
		"a size nobody has checked": {in: ai.Model{ID: "moonshot-v1-512k"}},
		"no size at all":            {in: ai.Model{ID: "moonshot-v1"}},
	}
	runInfer(t, inferMoonshot, tests)
}

func TestInferMiniMax(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"m3": {in: ai.Model{ID: "MiniMax-M3"}, window: 1_000_000, output: 8_192},
		"m2": {in: ai.Model{ID: "MiniMax-M2.1"}, window: 204_800, output: 8_192},
		// The M2 and M3 windows differ by a factor of five, so a generation
		// nobody has checked reports nothing rather than a neighbour's figure.
		"a generation nobody has checked": {in: ai.Model{ID: "MiniMax-M9"}},
	}
	runInfer(t, inferMiniMax, tests)
}

func TestInferMiMo(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"pro":   {in: ai.Model{ID: "mimo-v2.5-pro"}, window: 1_048_576, output: 131_072},
		"flash": {in: ai.Model{ID: "mimo-v2-flash"}, window: 262_144, output: 65_536},
		"omni":  {in: ai.Model{ID: "mimo-v2-omni"}, window: 262_144, output: 65_536},
		// MiMo's own listing returns the vendor-qualified name; both have to
		// size the same or the same model has two shapes.
		"vendor-qualified": {in: ai.Model{ID: "xiaomi/mimo-v2.5-pro"}, window: 1_048_576, output: 131_072},
		"another line":     {in: ai.Model{ID: "mimo-v1-pro"}},
	}
	runInfer(t, inferMiMo, tests)
}

func TestInferBigModel(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"glm-5":                        {in: ai.Model{ID: "glm-5-air"}, window: 200_000, output: 128_000},
		"glm-4.7":                      {in: ai.Model{ID: "glm-4.7-flashx"}, window: 200_000, output: 128_000},
		"glm-4.6":                      {in: ai.Model{ID: "glm-4.6"}, window: 200_000, output: 128_000},
		"a generation not on the docs": {in: ai.Model{ID: "glm-4.5"}},
	}
	runInfer(t, inferBigModel, tests)
}

func TestInferVolcengine(t *testing.T) {
	tests := map[string]struct {
		in             ai.Model
		window, output int
	}{
		"the seed generation": {in: ai.Model{ID: "doubao-seed-1-6"}, window: 1_024_000, output: 256_000},
		"a 256k suffix":       {in: ai.Model{ID: "doubao-pro-256k"}, window: 256_000, output: 8_000},
		"a 128k suffix":       {in: ai.Model{ID: "doubao-pro-128k"}, window: 128_000, output: 8_000},
		"a 32k suffix":        {in: ai.Model{ID: "doubao-pro-32k"}, window: 32_000, output: 4_000},
		"no suffix":           {in: ai.Model{ID: "doubao-pro"}},
	}
	runInfer(t, inferVolcengine, tests)
}

// runInfer drives one Infer function over a table of IDs. An expectation of
// zero means the function must report unknown rather than guess: a window is
// acted on silently, so a wrong one fails in both directions.
func runInfer(t *testing.T, infer func(ai.Model) ai.Model, tests map[string]struct {
	in             ai.Model
	window, output int
}) {
	t.Helper()
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got := infer(tc.in)
			if got.ContextWindow != tc.window {
				t.Errorf("ContextWindow = %d, want %d", got.ContextWindow, tc.window)
			}
			if got.MaxOutput != tc.output {
				t.Errorf("MaxOutput = %d, want %d", got.MaxOutput, tc.output)
			}
			if got.ID != tc.in.ID {
				t.Errorf("ID = %q, want it untouched", got.ID)
			}
		})
	}
}

// TestAVendorProviderKeepsWhatTheCatalogKnows pins that a provider built from a
// vendor hands back live-listed models already carrying what the catalog knows.
func TestAVendorProviderKeepsWhatTheCatalogKnows(t *testing.T) {
	v, ok := Find("openai")
	if !ok {
		t.Fatal("no openai vendor")
	}
	p := v.Provider(provider.Config{})

	// An ID the table does not list, which the vendor's Infer can still size.
	got, listed := p.Model("gpt-5.9-nova")
	if listed {
		t.Fatal("the model was reported as listed")
	}
	if got.ContextWindow == 0 || got.MaxOutput == 0 {
		t.Errorf("limits = %d/%d, want the generation's; the provider dropped the vendor's Infer",
			got.ContextWindow, got.MaxOutput)
	}
	if len(got.Reasoning) == 0 {
		t.Error("no reasoning ladder: a GPT-5 opened through a provider could not be asked to think")
	}
	if got.Vendor != "openai" || got.API != v.API {
		t.Errorf("model = %s/%s, want the vendor's identity and protocol", got.Vendor, got.API)
	}
}
