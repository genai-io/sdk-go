package llm_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
)

// A session file remembers which model was selected. Reading it back must give
// the same model — a reloaded one that lost its protocol quirks looks fine and
// misbehaves later, which is the worse of the two failures.
func TestModelSurvivesJSONRoundTrip(t *testing.T) {
	for _, ref := range []string{
		"deepseek/deepseek-v4-pro",   // OpenAIChatCompat, a thinking dialect
		"anthropic/claude-opus-5",    // AnthropicCompat, adaptive thinking
		"anthropic/claude-haiku-4-5", // AnthropicCompat, the budget shape
		"google/gemini-3.7-flash",    // GoogleCompat, a thinking level
		"openai/gpt-5.6-terra",       // OpenAIResponsesCompat
		"volcengine/doubao-seed-1.6", // AnthropicCompat, bearer auth
	} {
		t.Run(ref, func(t *testing.T) {
			original, err := catalog.Model(ref)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}
			blob, err := json.Marshal(original)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var back llm.Model
			if err := json.Unmarshal(blob, &back); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}

			if back.ID != original.ID || back.API != original.API {
				t.Fatalf("identity changed: %+v", back)
			}
			if back.ContextWindow != original.ContextWindow || back.MaxOutput != original.MaxOutput {
				t.Errorf("limits changed: %d/%d", back.ContextWindow, back.MaxOutput)
			}
			if len(back.Reasoning) != len(original.Reasoning) {
				t.Errorf("ladder rungs %d, want %d", len(back.Reasoning), len(original.Reasoning))
			}
			if back.Pricing.Input != original.Pricing.Input ||
				back.Pricing.Currency != original.Pricing.Currency ||
				len(back.Pricing.Tiers) != len(original.Pricing.Tiers) {
				t.Errorf("pricing changed: %+v", back.Pricing)
			}
			// The part that used to be lost.
			if back.Compat != original.Compat {
				t.Errorf("compat = %#v, want %#v", back.Compat, original.Compat)
			}
		})
	}
}

// The specific regression: a dialect that decoded into a map made CompatOf
// return the zero value, so DeepSeek stopped sending its "reasoning off" field
// and reasoning stayed on with nothing reporting it.
func TestReloadedModelKeepsItsThinkingDialect(t *testing.T) {
	original, _ := catalog.Model("deepseek/deepseek-v4-pro")
	blob, _ := json.Marshal(original)
	var back llm.Model
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := llm.CompatOf[llm.OpenAIChatCompat](back).Thinking; got != llm.ThinkingEffortOrDisable {
		t.Errorf("Thinking = %q, want %q", got, llm.ThinkingEffortOrDisable)
	}
}

// A model whose protocol nothing can rebuild fails loudly rather than loading
// with its quirks silently dropped.
func TestUnknownProtocolCompatIsAnError(t *testing.T) {
	blob := []byte(`{"id":"m","api":"some-custom-protocol","compat":{"x":1}}`)
	var m llm.Model
	if err := json.Unmarshal(blob, &m); err == nil {
		t.Fatal("expected an error for an unregistered protocol's compat")
	}

	// With no compat at all there is nothing to lose, so it loads.
	if err := json.Unmarshal([]byte(`{"id":"m","api":"some-custom-protocol"}`), &m); err != nil {
		t.Errorf("a model with no compat should load: %v", err)
	}
}

func TestRegisterCompatForACustomProtocol(t *testing.T) {
	type myCompat struct {
		Quirk string `json:"quirk"`
	}
	llm.RegisterCompat[myCompat]("test-custom-protocol")

	var m llm.Model
	blob := []byte(`{"id":"m","api":"test-custom-protocol","compat":{"quirk":"yes"}}`)
	if err := json.Unmarshal(blob, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := llm.CompatOf[myCompat](m).Quirk; got != "yes" {
		t.Errorf("quirk = %q", got)
	}
}

// A compat value belonging to another protocol is the one mistake CompatOf
// cannot report: the failed assertion yields the zero value, so the model runs
// with first-party defaults and nothing says its dialect was ignored.
func TestValidateRefusesAMismatchedCompat(t *testing.T) {
	m := llm.Model{ID: "m", API: llm.APIAnthropicMessages, Compat: llm.OpenAIChatCompat{}}
	err := m.Validate(nil, llm.Options{})
	if err == nil {
		t.Fatal("a model carrying another protocol's compat should not validate")
	}
	if !llm.IsUnsupported(err) {
		t.Errorf("err = %v, want KindUnsupported", err)
	}
	for _, want := range []string{"anthropic-messages", "OpenAIChatCompat", "AnthropicCompat"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("err = %v, want it to name %s", err, want)
		}
	}

	// The matching case, and the no-compat case, both pass.
	ok := llm.Model{ID: "m", API: llm.APIAnthropicMessages, Compat: llm.AnthropicCompat{}}
	if err := ok.Validate(nil, llm.Options{}); err != nil {
		t.Errorf("a matching compat should validate: %v", err)
	}
	bare := llm.Model{ID: "m", API: llm.APIAnthropicMessages}
	if err := bare.Validate(nil, llm.Options{}); err != nil {
		t.Errorf("no compat should validate: %v", err)
	}
}

// A protocol this package does not define has nothing to compare against, so
// the check stays out of the way rather than guessing.
func TestValidateLeavesAnUnregisteredProtocolAlone(t *testing.T) {
	type otherCompat struct{ X string }
	m := llm.Model{ID: "m", API: "some-unregistered-protocol", Compat: otherCompat{}}
	if err := m.Validate(nil, llm.Options{}); err != nil {
		t.Errorf("an unregistered protocol should not be second-guessed: %v", err)
	}
}
