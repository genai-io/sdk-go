package catalog

import (
	"regexp"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// A Vendor.Infer function runs on every resolved model, after the vendor
// defaults, and by convention only fills fields that are still unset — so a
// row that states a ladder or a dialect is never overwritten by a guess. It
// decides protocol behavior only: token limits are the application's to set.

// openAIGenerations says whether an OpenAI generation reasons, most specific
// first.
//
// The patterns match a generation as a whole token, not as a prefix: "gpt-4"
// as a prefix would also claim an unpublished gpt-4.5, so the 4-series
// patterns stop at a following digit or dot. They match anywhere in the ID,
// which reads a fine-tune ("ft:gpt-5.4-2026-01-01:acme::abc") as the model it
// was tuned from.
var openAIGenerations = []struct {
	generation *regexp.Regexp
	reasons    bool
}{
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-[56](\.[0-9]+)?([^.0-9]|$)`), true},
	// The generations before GPT-5. Not listed as rows, but /v1/models still
	// serves them, so naming one gets the right ladder rather than none.
	{regexp.MustCompile(`(^|[^a-z0-9])o[134]([^.0-9]|$)`), true},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4\.1([^.0-9]|$)`), false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4o([^.0-9]|$)`), false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4-turbo([^.0-9]|$)`), false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4([^.0-9o]|$)`), false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-3\.5-turbo([^.0-9]|$)`), false},
}

// inferOpenAI picks the reasoning ladder for a model ID the table does not
// list, by generation.
//
// An ID whose generation is not recognised is left alone: a nil Reasoning says
// nothing is known, not "this model does not reason".
func inferOpenAI(m ai.Model) ai.Model {
	if m.Reasoning != nil {
		return m
	}
	id := strings.ToLower(strings.TrimSpace(m.ID))
	for _, g := range openAIGenerations {
		if !g.generation.MatchString(id) {
			continue
		}
		if g.reasons {
			m.Reasoning = openAIEfforts
		} else {
			m.Reasoning = noReasoning
		}
		return m
	}
	return m
}

// inferGoogle picks a Gemini model's thinking dialect. Gemini 3 takes a
// thinking level, the vendor default; 2.5 still takes a budget.
func inferGoogle(m ai.Model) ai.Model {
	if !strings.HasPrefix(strings.ToLower(m.ID), "gemini-2.5") {
		return m
	}
	// Only fill what the row left blank: Infer is a fallback for an unlisted
	// ID, never a correction of stated data.
	if m.Compat == nil {
		m.Compat = ai.GoogleCompat{}
	}
	if m.Reasoning == nil {
		m.Reasoning = budgetLadder
	}
	return m
}
