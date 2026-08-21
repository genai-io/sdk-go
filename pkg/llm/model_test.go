package llm_test

import (
	"math"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// switchLadder is a two-state endpoint: reasoning is on or it is off.
var switchLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff, Default: true},
	{Effort: llm.EffortHigh, Value: "enabled"},
}

// fullLadder spans the ladder with a level string per rung.
var fullLadder = []llm.ReasoningLevel{
	{Effort: llm.EffortOff, Value: "none"},
	{Effort: llm.EffortLow, Value: "low"},
	{Effort: llm.EffortMedium, Value: "medium", Default: true},
	{Effort: llm.EffortHigh, Value: "high"},
}

func TestResolveLevelExactAndDefault(t *testing.T) {
	m := llm.Model{ID: "m", Reasoning: fullLadder}

	if level, ok := m.ResolveLevel(llm.EffortLow); !ok || level.Value != "low" {
		t.Errorf("exact rung = %+v, %v", level, ok)
	}
	// An unset effort takes the rung the model marks default.
	if level, ok := m.ResolveLevel(llm.EffortDefault); !ok || level.Value != "medium" {
		t.Errorf("default rung = %+v, %v", level, ok)
	}
}

// The search runs upward first: quietly reasoning less than asked is the more
// surprising failure, so a request the model cannot meet exactly rounds up.
func TestResolveLevelRoundsUp(t *testing.T) {
	m := llm.Model{ID: "m", Reasoning: switchLadder}

	for _, ask := range []llm.Effort{llm.EffortMinimal, llm.EffortLow, llm.EffortMedium} {
		level, ok := m.ResolveLevel(ask)
		if !ok || level.Effort != llm.EffortHigh {
			t.Errorf("ResolveLevel(%q) = %+v, want the high rung", ask, level)
		}
	}
}

// Only when nothing above exists does it fall back downward.
func TestResolveLevelFallsBackDown(t *testing.T) {
	m := llm.Model{ID: "m", Reasoning: fullLadder}

	for _, ask := range []llm.Effort{llm.EffortXHigh, llm.EffortMax} {
		level, ok := m.ResolveLevel(ask)
		if !ok || level.Effort != llm.EffortHigh {
			t.Errorf("ResolveLevel(%q) = %+v, want the highest rung offered", ask, level)
		}
	}
}

func TestResolveLevelOnNonReasoningModel(t *testing.T) {
	m := llm.Model{ID: "m"}
	if _, ok := m.ResolveLevel(llm.EffortHigh); ok {
		t.Error("a model with no ladder must yield no rung")
	}
}

// A ladder with no default rung sends nothing when the caller says nothing,
// leaving the provider's own default in place.
func TestNoDefaultRungYieldsNothing(t *testing.T) {
	m := llm.Model{ID: "m", Reasoning: []llm.ReasoningLevel{
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortHigh, Value: "high"},
	}}
	if _, ok := m.ResolveLevel(llm.EffortDefault); ok {
		t.Error("a ladder with no default should resolve to no rung")
	}
	// An explicit ask still works.
	if level, ok := m.ResolveLevel(llm.EffortHigh); !ok || level.Value != "high" {
		t.Errorf("explicit ask = %+v, %v", level, ok)
	}
}

func TestAcceptsModality(t *testing.T) {
	// No stated modalities means text only.
	bare := llm.Model{ID: "m"}
	if !bare.Accepts(llm.ModalityText) || bare.Accepts(llm.ModalityImage) {
		t.Error("a model stating no modalities should be text-only")
	}

	vision := llm.Model{ID: "m", Input: []llm.Modality{llm.ModalityText, llm.ModalityImage}}
	if !vision.Accepts(llm.ModalityImage) || vision.Accepts(llm.ModalityAudio) {
		t.Errorf("Accepts is wrong for %+v", vision.Input)
	}
}

func TestCompatOfYieldsZeroValueWhenAbsent(t *testing.T) {
	// A model carrying no compat behaves as all-defaults, which is what makes
	// "not stated" and "everything default" the same thing.
	bare := llm.CompatOf[llm.AnthropicCompat](llm.Model{ID: "m"})
	if bare.ForceAdaptiveThinking || bare.BearerAuth {
		t.Errorf("zero value expected, got %+v", bare)
	}

	// A model carrying another protocol's compat must not leak through.
	wrong := llm.CompatOf[llm.AnthropicCompat](llm.Model{ID: "m", Compat: llm.GoogleCompat{ThinkingLevel: true}})
	if wrong.ForceAdaptiveThinking {
		t.Error("compat of the wrong protocol was read as this one's")
	}

	right := llm.CompatOf[llm.AnthropicCompat](llm.Model{ID: "m", Compat: llm.AnthropicCompat{BearerAuth: true}})
	if !right.BearerAuth {
		t.Error("compat was not read back")
	}
}

func TestPricingTiers(t *testing.T) {
	p := llm.Pricing{
		Currency: llm.CNY, Input: 2.10, Output: 8.40,
		Tiers: []llm.PricingTier{{AboveInputTokens: 512_000, Input: 4.20, Output: 16.80}},
	}

	// Below the threshold, the base card applies.
	base := p.Cost(llm.Usage{Input: 100_000, Output: 1_000_000})
	if want := 0.1*2.10 + 8.40; math.Abs(base.Total-want) > 1e-9 {
		t.Errorf("base total = %v, want %v", base.Total, want)
	}
	if base.Currency != llm.CNY {
		t.Errorf("currency = %q", base.Currency)
	}

	// The threshold counts the whole prompt, cached portion included.
	tiered := p.Cost(llm.Usage{Input: 600_000, CacheRead: 1_000, Output: 1_000_000})
	if got := tiered.Input; math.Abs(got-600_000*4.20/1_000_000) > 1e-9 {
		t.Errorf("tiered input = %v, want the above-512k rate", got)
	}
	if got := tiered.Output; got != 16.80 {
		t.Errorf("tiered output = %v, want the above-512k rate", got)
	}
}
