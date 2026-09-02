package catalog

import (
	"regexp"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// A Vendor.Infer function runs on every resolved model, after the vendor
// defaults, and by convention only fills fields that are still zero — so a
// figure the table did state is never overwritten by a guess. Several vendors
// encode a model's context window in its ID ("kimi-...-128k", "glm-5.2-...")
// while publishing nothing through their API.

// inferAnthropic sizes a Claude model the table does not list. Claude IDs are
// stable and long-lived, so an unlisted one is nearly always a model newer
// than this catalog: assume the current generation's shape rather than
// reporting nothing.
func inferAnthropic(m ai.Model) ai.Model {
	if !strings.HasPrefix(strings.ToLower(m.ID), "claude-") {
		return m
	}
	m.ContextWindow = orDefault(m.ContextWindow, 1_000_000)
	m.MaxOutput = orDefault(m.MaxOutput, 128_000)
	return m
}

// openAIGenerations sizes an OpenAI model by its generation, most specific
// first.
//
// Each pattern matches the generation as a whole token rather than as a
// prefix, and that is the whole point of them being patterns:
//
//   - a prefix of "gpt-4" also matches a gpt-4.5 nobody has published yet, and
//     would size it at the original GPT-4's 8k. The 4-series patterns below
//     stop at a following digit or dot, so an unrecognised point release
//     reports unknown instead of a figure that is off by two orders.
//   - a fine-tune is named for what it was tuned from and wrapped in a job
//     ("ft:gpt-5.4-2026-01-01:acme::abc"), so the generation is not at the
//     front. Matching anywhere in the ID sizes it as its base model.
var openAIGenerations = []struct {
	generation     *regexp.Regexp
	window, output int
	reasons        bool
}{
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-[56](\.[0-9]+)?([^.0-9]|$)`), 1_050_000, 128_000, true},
	// The generations before GPT-5. They are not listed as rows — nobody
	// starts a project on one — but /v1/models still serves them, and a caller
	// who names one deserves a window rather than silence.
	{regexp.MustCompile(`(^|[^a-z0-9])o[134]([^.0-9]|$)`), 200_000, 100_000, true},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4\.1([^.0-9]|$)`), 1_047_576, 32_768, false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4o([^.0-9]|$)`), 128_000, 16_384, false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4-turbo([^.0-9]|$)`), 128_000, 4_096, false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-4([^.0-9o]|$)`), 8_192, 8_192, false},
	{regexp.MustCompile(`(^|[^a-z0-9])gpt-3\.5-turbo([^.0-9]|$)`), 16_385, 4_096, false},
}

// inferOpenAI fills limits and reasoning for a model ID the table does not
// list. The /v1/models listing carries neither, so both come from the
// published per-model pages.
//
// An ID whose generation is not recognised is left alone, ladder included. A
// nil Reasoning says nothing is known about this model, which is what is
// true; stamping it "does not reason" would answer a question nobody here can
// answer, and the caller would believe it.
func inferOpenAI(m ai.Model) ai.Model {
	id := strings.ToLower(strings.TrimSpace(m.ID))
	for _, g := range openAIGenerations {
		if !g.generation.MatchString(id) {
			continue
		}
		m.ContextWindow = orDefault(m.ContextWindow, g.window)
		m.MaxOutput = orDefault(m.MaxOutput, g.output)
		if m.Reasoning == nil {
			if g.reasons {
				m.Reasoning = openAIEfforts
			} else {
				m.Reasoning = noReasoning
			}
		}
		return m
	}
	return m
}

// inferGoogle sizes a Gemini model and picks its thinking dialect. Gemini 3
// takes a thinking level; 2.5 still takes a budget.
func inferGoogle(m ai.Model) ai.Model {
	id := strings.ToLower(m.ID)
	switch {
	case strings.HasPrefix(id, "gemini-3"):
		m.ContextWindow = orDefault(m.ContextWindow, 1_048_576)
		m.MaxOutput = orDefault(m.MaxOutput, 65_536)
	case strings.HasPrefix(id, "gemini-2.5"):
		m.ContextWindow = orDefault(m.ContextWindow, 1_048_576)
		m.MaxOutput = orDefault(m.MaxOutput, 65_536)
		// Only fill what the row left blank: Infer is a fallback for an
		// unlisted ID, never a correction of stated data.
		if m.Compat == nil {
			m.Compat = ai.GoogleCompat{}
		}
		if m.Reasoning == nil {
			m.Reasoning = budgetLadder
		}
	}
	return m
}

// moonshotWindow reads the window suffix out of a Kimi model ID as a whole
// number rather than as a substring. "8k" occurs inside "128k", so a
// substring test only worked because of the order the cases happened to be
// written in — one reordering away from sizing a 128k model at 8k.
var moonshotWindow = regexp.MustCompile(`(^|[^0-9])([0-9]+)k($|[^a-z0-9])`)

// inferMoonshot reads the window out of a Kimi model ID, which is where
// Moonshot puts it when /v1/models omits context_length.
func inferMoonshot(m ai.Model) ai.Model {
	id := strings.ToLower(m.ID)
	// The current line names its generation instead of its size, and the
	// generation is the more specific fact, so it is read first.
	switch {
	case strings.Contains(id, "k3"):
		m.ContextWindow = orDefault(m.ContextWindow, 1_048_576)
		return m
	case strings.Contains(id, "k2"):
		m.ContextWindow = orDefault(m.ContextWindow, 262_144)
		return m
	}
	match := moonshotWindow.FindStringSubmatch(id)
	if match == nil {
		return m
	}
	switch match[2] {
	case "128":
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 131_072), orDefault(m.MaxOutput, 8_192)
	case "32":
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 32_768), orDefault(m.MaxOutput, 8_192)
	case "8":
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 8_192), orDefault(m.MaxOutput, 3_000)
	}
	return m
}

// inferMiniMax sizes a MiniMax model from its generation. MiniMax publishes
// prices per model but no windows at all, and its Anthropic-compatible
// endpoint lists none either, so the generation in the ID is the only thing
// there is to read: the whole M2 line shares one window, and M3 another.
//
// A generation this does not name reports unknown rather than borrowing a
// neighbour's figure — the M2 and M3 windows differ by a factor of five, so a
// wrong guess here is not a small one.
func inferMiniMax(m ai.Model) ai.Model {
	id := strings.ToLower(m.ID)
	switch {
	case strings.Contains(id, "minimax-m3"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 1_000_000), orDefault(m.MaxOutput, 8_192)
	case strings.Contains(id, "minimax-m2"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 204_800), orDefault(m.MaxOutput, 8_192)
	}
	return m
}

// inferMiMo sizes a MiMo model from its generation.
//
// It matches on the generation rather than the whole ID because MiMo serves
// the same model under two names — "mimo-v2.5-pro" and the vendor-qualified
// "xiaomi/mimo-v2.5-pro" its own listing returns — and both have to size the
// same. Within the v2 line the pro and flash tiers differ, so the tier is read
// too.
func inferMiMo(m ai.Model) ai.Model {
	id := strings.ToLower(m.ID)
	if !strings.Contains(id, "mimo-v2") {
		return m
	}
	switch {
	case strings.Contains(id, "flash"), strings.Contains(id, "omni"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 262_144), orDefault(m.MaxOutput, 65_536)
	default:
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 1_048_576), orDefault(m.MaxOutput, 131_072)
	}
	return m
}

// inferBigModel sizes a GLM model. BigModel's /v1/models follows the bare
// OpenAI shape — id, object, owned_by — and publishes no window at all. Only
// the generations whose docs state 200K are sized; anything else reports
// unknown rather than inheriting a figure that was never checked.
func inferBigModel(m ai.Model) ai.Model {
	id := strings.ToLower(m.ID)
	if strings.HasPrefix(id, "glm-5") || strings.HasPrefix(id, "glm-4.7") || strings.HasPrefix(id, "glm-4.6") {
		m.ContextWindow = orDefault(m.ContextWindow, 200_000)
		m.MaxOutput = orDefault(m.MaxOutput, 128_000)
	}
	return m
}

// inferVolcengine sizes a Doubao model. Ark encodes the window in the model ID
// suffix; the seed generation publishes a 1024k window and 256k output.
func inferVolcengine(m ai.Model) ai.Model {
	id := strings.ToLower(m.ID)
	switch {
	case strings.Contains(id, "seed"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 1_024_000), orDefault(m.MaxOutput, 256_000)
	case strings.Contains(id, "256k"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 256_000), orDefault(m.MaxOutput, 8_000)
	case strings.Contains(id, "128k"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 128_000), orDefault(m.MaxOutput, 8_000)
	case strings.Contains(id, "32k"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 32_000), orDefault(m.MaxOutput, 4_000)
	}
	return m
}

// orDefault keeps an Infer from overwriting a figure the catalog states.
func orDefault(stated, inferred int) int {
	if stated > 0 {
		return stated
	}
	return inferred
}
