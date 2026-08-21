package catalog_test

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
)

func TestModelQualifiedReference(t *testing.T) {
	m, err := catalog.Model("deepseek/deepseek-v4-pro")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Vendor != "deepseek" || m.ID != "deepseek-v4-pro" {
		t.Errorf("model = %+v", m)
	}
	if m.API != llm.APIOpenAIChat {
		t.Errorf("API = %q, want %q", m.API, llm.APIOpenAIChat)
	}
	if m.BaseURL == "" {
		t.Error("vendor base URL was not inherited")
	}
	if got := llm.CompatOf[llm.OpenAIChatCompat](m).Thinking; got != llm.ThinkingEffortOrDisable {
		t.Errorf("reasoning dialect = %q", got)
	}
	if m.Accepts(llm.ModalityImage) {
		t.Error("DeepSeek's chat endpoint is text-only")
	}
}

// A model ID may itself contain a slash — aggregators routinely namespace
// them — so only the first segment names the vendor.
func TestModelReferenceWithSlashInID(t *testing.T) {
	m, err := catalog.Model("agnesai/moonshotai/kimi-k3")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Vendor != "agnesai" || m.ID != "moonshotai/kimi-k3" {
		t.Errorf("model = %+v", m)
	}
}

func TestMimoModelHasCatalogLimits(t *testing.T) {
	m, err := catalog.Model("mimo/mimo-v2.5-pro")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.ContextWindow == 0 || !m.Pricing.Known() {
		t.Errorf("catalog data was not applied: %+v", m)
	}
}

func TestModelBareReference(t *testing.T) {
	m, err := catalog.Model("claude-opus-4-6")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Vendor != "anthropic" || m.API != llm.APIAnthropicMessages {
		t.Errorf("model = %+v", m)
	}
}

func TestModelAmbiguousReference(t *testing.T) {
	// deepseek-v4-flash is served both directly and through SenseNova.
	_, err := catalog.Model("deepseek-v4-flash")
	var ambiguous *catalog.AmbiguousModelError
	if !errors.As(err, &ambiguous) {
		t.Fatalf("err = %v, want *AmbiguousModelError", err)
	}
	if len(ambiguous.Candidates) < 2 {
		t.Errorf("candidates = %v", ambiguous.Candidates)
	}
}

func TestModelUnknownBareReference(t *testing.T) {
	_, err := catalog.Model("totally-made-up")
	var unknown *catalog.UnknownModelError
	if !errors.As(err, &unknown) {
		t.Fatalf("err = %v, want *UnknownModelError", err)
	}
	// The message has to say how to proceed, since qualifying it works.
	if !strings.Contains(err.Error(), "vendor/") {
		t.Errorf("unhelpful message: %v", err)
	}
}

// A vendor ships models faster than this table; an unlisted ID must still
// resolve to something usable.
func TestUnlistedModelInheritsVendor(t *testing.T) {
	m, err := catalog.Model("deepseek/deepseek-v9-unreleased")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.API != llm.APIOpenAIChat || m.BaseURL == "" {
		t.Errorf("vendor defaults were not inherited: %+v", m)
	}
	if m.ContextWindow != 0 {
		t.Errorf("ContextWindow = %d, want 0 (unknown, not guessed)", m.ContextWindow)
	}
}

func TestNoReasoningIsNotOverwritten(t *testing.T) {
	gpt4o, err := catalog.Model("openai/gpt-4o")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if gpt4o.Reasons() {
		t.Error("a model marked NoReasoning inherited a reasoning ladder")
	}

	opus, err := catalog.Model("anthropic/claude-opus-4-6")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if !opus.Reasons() {
		t.Error("a model that omits Reasoning should inherit the vendor default")
	}
}

// Claude 4.6 and later reject a budget_tokens request; the 4.5 generation only
// accepts one. Which dialect a model speaks has to survive the vendor default.
func TestAnthropicThinkingDialectIsPerModel(t *testing.T) {
	tests := map[string]bool{
		"anthropic/claude-opus-5":     true,
		"anthropic/claude-sonnet-4-6": true,
		"anthropic/claude-opus-4-5":   false,
		"anthropic/claude-haiku-4-5":  false,
		// An Anthropic-compatible vendor implements the older shape only.
		"mimo/mimo-v2.5-pro": false,
	}
	for ref, wantAdaptive := range tests {
		m, err := catalog.Model(ref)
		if err != nil {
			t.Fatalf("Model(%q): %v", ref, err)
		}
		got := llm.CompatOf[llm.AnthropicCompat](m).ForceAdaptiveThinking
		if got != wantAdaptive {
			t.Errorf("%s ForceAdaptiveThinking = %v, want %v", ref, got, wantAdaptive)
		}
		// The rungs have to match the shape: adaptive carries effort literals,
		// budget carries token counts.
		level, ok := m.ResolveLevel(llm.EffortHigh)
		if !ok {
			t.Fatalf("%s offers no high rung", ref)
		}
		if wantAdaptive && level.Value == "" {
			t.Errorf("%s is adaptive but its high rung carries no effort literal", ref)
		}
		if !wantAdaptive && level.Budget == 0 {
			t.Errorf("%s is budget-based but its high rung carries no budget", ref)
		}
	}
}

// Fable 5 reasons unconditionally — an explicit "off" is rejected — so the
// ladder must not offer one.
func TestFable5CannotTurnThinkingOff(t *testing.T) {
	m, err := catalog.Model("anthropic/claude-fable-5")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	for _, e := range m.Efforts() {
		if e == llm.EffortOff {
			t.Fatal("claude-fable-5 must not advertise EffortOff")
		}
	}
	level, ok := m.ResolveLevel(llm.EffortOff)
	if !ok || level.Effort == llm.EffortOff {
		t.Errorf("ResolveLevel(off) = %+v, want it snapped onto a supported rung", level)
	}
}

// Gemini 3 replaced the thinking budget with a level; 2.5 still takes a budget.
func TestGeminiThinkingDialectIsPerGeneration(t *testing.T) {
	three, _ := catalog.Model("google/gemini-3.7-flash")
	if !llm.CompatOf[llm.GoogleCompat](three).ThinkingLevel {
		t.Error("gemini-3.7-flash should take a thinking level")
	}
	if level, ok := three.ResolveLevel(llm.EffortHigh); !ok || level.Value != "HIGH" {
		t.Errorf("gemini-3.7-flash high rung = %+v", level)
	}

	twoFive, _ := catalog.Model("google/gemini-2.5-pro")
	if llm.CompatOf[llm.GoogleCompat](twoFive).ThinkingLevel {
		t.Error("gemini-2.5-pro should take a thinking budget, not a level")
	}
	if level, ok := twoFive.ResolveLevel(llm.EffortHigh); !ok || level.Budget == 0 {
		t.Errorf("gemini-2.5-pro high rung = %+v", level)
	}
	if twoFive.ContextWindow == 0 {
		t.Error("gemini-2.5-pro was not sized")
	}
}

func TestInferSizesUnlistedModels(t *testing.T) {
	tests := []struct {
		ref  string
		want int
	}{
		{"openai/gpt-5.9-unreleased", 1_050_000},
		{"openai/gpt-4o-mini", 128_000},
		{"bigmodel/glm-5.2-air", 200_000},
		{"bigmodel/glm-4.6", 200_000},
		{"moonshot/kimi-k2-turbo", 262_144},
		{"moonshot/kimi-k3-preview", 1_048_576},
		{"volcengine/doubao-pro-256k", 256_000},
		{"volcengine/doubao-seed-evolving-latest-version", 1_024_000},
		{"anthropic/claude-opus-9", 1_000_000},
	}
	for _, tc := range tests {
		m, err := catalog.Model(tc.ref)
		if err != nil {
			t.Fatalf("Model(%q): %v", tc.ref, err)
		}
		if m.ContextWindow != tc.want {
			t.Errorf("%s ContextWindow = %d, want %d", tc.ref, m.ContextWindow, tc.want)
		}
	}
}

func TestOpenAIReasoningIsPerFamily(t *testing.T) {
	gpt5, _ := catalog.Model("openai/gpt-5.5")
	if !gpt5.Reasons() {
		t.Error("gpt-5 should reason")
	}
	// reasoning.effort accepts "none" on these models, so "off" is a real
	// rung rather than something the driver has to fake.
	level, ok := gpt5.ResolveLevel(llm.EffortOff)
	if !ok || level.Value != "none" {
		t.Errorf("gpt-5 off rung = %+v, want value \"none\"", level)
	}
	gpt4o, _ := catalog.Model("openai/gpt-4o")
	if gpt4o.Reasons() {
		t.Error("gpt-4o does not reason")
	}
}

// Every entry states when it was last checked, so a stale window or price is
// visible rather than silent.
func TestEveryVendorRecordsWhenItWasVerified(t *testing.T) {
	for _, v := range catalog.All() {
		if _, err := time.Parse("2006-01-02", v.Verified); err != nil {
			t.Errorf("vendor %q has an unusable Verified date %q: %v", v.ID, v.Verified, err)
		}
	}
	// Nothing should be stale relative to its own recorded date.
	if stale := catalog.Stale(mustDate(t, "2026-08-21"), 24*time.Hour); len(stale) != 0 {
		t.Errorf("unexpectedly stale: %v", vendorIDs(stale))
	}
	// ...and everything is stale once enough time passes.
	if stale := catalog.Stale(mustDate(t, "2028-01-01"), 24*time.Hour); len(stale) != len(catalog.All()) {
		t.Errorf("Stale reported %d of %d vendors", len(stale), len(catalog.All()))
	}
}

func mustDate(t *testing.T, s string) time.Time {
	t.Helper()
	d, err := time.Parse("2006-01-02", s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return d
}

func vendorIDs(vs []catalog.Vendor) []string {
	out := make([]string, len(vs))
	for i, v := range vs {
		out[i] = v.ID
	}
	return out
}

func TestOllamaBaseURLSuffix(t *testing.T) {
	v, ok := catalog.Find("ollama")
	if !ok {
		t.Fatal("ollama vendor missing")
	}
	tests := map[string]string{
		"":                        v.BaseURL,
		"http://box:11434":        "http://box:11434/v1",
		"http://box:11434/":       "http://box:11434/v1",
		"http://box:11434/v1":     "http://box:11434/v1",
		"https://ollama.internal": "https://ollama.internal/v1",
	}
	for in, want := range tests {
		if got := v.ResolveBaseURL(in); got != want {
			t.Errorf("ResolveBaseURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEnrichKeepsEndpointFacts(t *testing.T) {
	// A listing that reports its own window must win; the catalog only fills
	// what the endpoint left out.
	listed := []llm.Model{
		{ID: "deepseek-v4-pro", ContextWindow: 512_000},
		{ID: "deepseek-v4-flash"},
	}
	got := catalog.Enrich("deepseek", listed)
	if got[0].ContextWindow != 512_000 {
		t.Errorf("endpoint's own figure was overwritten: %d", got[0].ContextWindow)
	}
	if got[1].ContextWindow != 1_000_000 {
		t.Errorf("catalog did not fill the gap: %d", got[1].ContextWindow)
	}
	if !got[1].Pricing.Known() {
		t.Error("pricing was not merged in")
	}
}

func TestMissingReportsUnlistedCatalogEntries(t *testing.T) {
	missing := catalog.Missing("deepseek", []llm.Model{{ID: "deepseek-v4-pro"}})
	if len(missing) != 1 || missing[0].ID != "deepseek-v4-flash" {
		t.Errorf("missing = %+v", missing)
	}
}

func TestPricingCarriesCurrency(t *testing.T) {
	m, err := catalog.Model("minmax/MiniMax-M2.7")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Pricing.Currency != llm.CNY {
		t.Errorf("currency = %q, want %q", m.Pricing.Currency, llm.CNY)
	}
	cost := m.Pricing.Cost(llm.Usage{Input: 1_000_000, Output: 1_000_000})
	if cost.Currency != llm.CNY {
		t.Errorf("cost currency = %q", cost.Currency)
	}
	if want := 2.1 + 8.4; cost.Total != want {
		t.Errorf("total = %v, want %v", cost.Total, want)
	}
}

func TestEveryVendorHasARegisteredProtocol(t *testing.T) {
	known := map[llm.API]bool{
		llm.APIAnthropicMessages: true,
		llm.APIAnthropicVertex:   true,
		llm.APIOpenAIChat:        true,
		llm.APIOpenAIResponses:   true,
		llm.APIGoogleGenAI:       true,
	}
	for _, v := range catalog.All() {
		if v.ID == "" || v.DisplayName == "" {
			t.Errorf("vendor %+v is missing an identifier", v)
		}
		if !known[v.API] {
			t.Errorf("vendor %q speaks unknown protocol %q", v.ID, v.API)
		}
	}
}

// The nine gateway vendors are data alone — no new protocol code — which is
// the whole claim of splitting protocol from vendor. This guards the claim.
func TestGatewayVendorsAreDataOnly(t *testing.T) {
	gateways := []string{
		"openrouter", "xai", "zai", "groq", "cerebras",
		"together", "fireworks", "nvidia", "huggingface",
	}
	for _, id := range gateways {
		v, ok := catalog.Find(id)
		if !ok {
			t.Errorf("vendor %q missing", id)
			continue
		}
		if v.API != llm.APIOpenAIChat {
			t.Errorf("%s speaks %q; a new protocol would need code", id, v.API)
		}
		if v.BaseURL == "" {
			t.Errorf("%s states no base URL", id)
		}
		if len(v.KeyEnv) == 0 {
			t.Errorf("%s names no credential variable", id)
		}
	}
}

// An aggregator serves models from many upstreams with different reasoning
// controls, so a vendor-wide ladder would be wrong for most of them. Stating
// none is the honest position, and the Note has to say so.
func TestAggregatorsStateNoLadder(t *testing.T) {
	for _, id := range []string{"groq", "cerebras", "together", "fireworks", "nvidia", "huggingface"} {
		v, _ := catalog.Find(id)
		if len(v.Reasoning) != 0 {
			t.Errorf("%s states a vendor-wide ladder: %+v", id, v.Reasoning)
		}
		if v.Note == "" {
			t.Errorf("%s states no ladder and no reason why", id)
		}
	}

	// OpenRouter is the exception: it normalizes reasoning itself, so one
	// ladder is correct across everything it serves.
	m, err := catalog.Model("openrouter/anthropic/claude-opus-5")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.ID != "anthropic/claude-opus-5" {
		t.Errorf("ID = %q, want the slash-bearing upstream ID intact", m.ID)
	}
	if llm.CompatOf[llm.OpenAIChatCompat](m).Thinking != llm.ThinkingReasoningObject {
		t.Error("openrouter should use the nested reasoning object")
	}
	if level, ok := m.ResolveLevel(llm.EffortHigh); !ok || level.Value != "high" {
		t.Errorf("openrouter high rung = %+v", level)
	}
}

// Adding one alternate hosting vendor must not make every model it serves
// ambiguous by bare name: nobody typing "claude-opus-5" means the Vertex
// deployment.
func TestBareReferencePrefersDirectVendor(t *testing.T) {
	m, err := catalog.Model("claude-opus-5")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Vendor != "anthropic" {
		t.Errorf("vendor = %q, want the first-party API", m.Vendor)
	}

	// Naming the vendor is how the deployment is chosen.
	vertex, err := catalog.Model("anthropic-vertex/claude-opus-5")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if vertex.API != llm.APIAnthropicVertex {
		t.Errorf("API = %q", vertex.API)
	}

	// Genuine ambiguity between two direct vendors still reports rather than
	// silently picking one.
	if _, err := catalog.Model("deepseek-v4-flash"); err == nil {
		t.Error("a model served by two direct vendors should stay ambiguous")
	}
}

func TestVertexModelIDsKeepSnapshotForm(t *testing.T) {
	// Pre-4.6 generations use the @-versioned form on Vertex.
	if _, err := catalog.Model("anthropic-vertex/claude-opus-4-5@20251101"); err != nil {
		t.Errorf("Model: %v", err)
	}
	v, _ := catalog.Find("anthropic-vertex")
	if !v.NeedsDeployment() {
		t.Error("the Vertex vendor should report that it needs a deployment")
	}
	// Models retired on the first-party API remain available on Google Cloud.
	if _, err := catalog.Model("anthropic-vertex/claude-opus-4-1@20250805"); err != nil {
		t.Errorf("Opus 4.1 should still be listed on Vertex: %v", err)
	}
	// It is retired on the first-party API, but the entry stays so a caller
	// still pointing at it gets told what happened.
	opus41, err := catalog.Model("anthropic/claude-opus-4-1-20250805")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if opus41.Stage != llm.StageRetired {
		t.Errorf("Stage = %q, want retired", opus41.Stage)
	}
	if opus41.Replacement == "" {
		t.Error("a retired model should name its replacement")
	}
}
