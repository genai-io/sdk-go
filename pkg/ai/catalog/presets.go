package catalog

import (
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ─── reasoning ladders ───

var (
	// claudeAdaptive is output_config.effort for Claude 4.7 and later, where
	// xhigh sits between high and max.
	claudeAdaptive = []ai.ReasoningLevel{
		{Effort: ai.EffortOff},
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortMedium, Value: "medium"},
		{Effort: ai.EffortHigh, Value: "high", Default: true},
		{Effort: ai.EffortXHigh, Value: "xhigh"},
		{Effort: ai.EffortMax, Value: "max"},
	}
	// claudeAdaptive46 is the same ladder for Claude 4.6, which predates xhigh.
	claudeAdaptive46 = []ai.ReasoningLevel{
		{Effort: ai.EffortOff},
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortMedium, Value: "medium"},
		{Effort: ai.EffortHigh, Value: "high", Default: true},
		{Effort: ai.EffortMax, Value: "max"},
	}
	// claudeAlwaysOn drops the off rung: Fable 5 reasons unconditionally and
	// rejects an explicit thinking: {"type": "disabled"} with a 400.
	claudeAlwaysOn = []ai.ReasoningLevel{
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortMedium, Value: "medium"},
		{Effort: ai.EffortHigh, Value: "high", Default: true},
		{Effort: ai.EffortXHigh, Value: "xhigh"},
		{Effort: ai.EffortMax, Value: "max"},
	}
	// budgetLadder is the same four rungs expressed as token budgets. Three
	// unrelated protocols land on it — Anthropic's older thinking.budget_tokens
	// (which no listed Claude still takes, but every Anthropic-compatible
	// third party does), Gemini 2.5's thinkingConfig.thinkingBudget, and
	// DashScope's thinking_budget — because the rungs are the same decision
	// and only the field name differs. Which field it lands in is the driver's
	// business, decided by Compat.
	budgetLadder = []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Default: true},
		{Effort: ai.EffortLow, Budget: 5_000},
		{Effort: ai.EffortMedium, Budget: 32_000},
		{Effort: ai.EffortHigh, Budget: 128_000},
	}

	// openAIEfforts is reasoning.effort. "none" is a real value here, so "off"
	// is expressible rather than something the catalog has to hide.
	openAIEfforts = []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Value: "none"},
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortMedium, Value: "medium", Default: true},
		{Effort: ai.EffortHigh, Value: "high"},
		{Effort: ai.EffortXHigh, Value: "xhigh"},
	}
	// openAIEffortsMax adds the top rung the 5.6 family accepts.
	openAIEffortsMax = append(append([]ai.ReasoningLevel{}, openAIEfforts...),
		ai.ReasoningLevel{Effort: ai.EffortMax, Value: "max"})

	// geminiLevels is thinkingConfig.thinkingLevel — Gemini 3. MINIMAL exists
	// in the enum but Gemini 3 rejects it, so there is no minimal rung.
	geminiLevels = []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Default: true},
		{Effort: ai.EffortLow, Value: "LOW"},
		{Effort: ai.EffortMedium, Value: "MEDIUM"},
		{Effort: ai.EffortHigh, Value: "HIGH"},
	}
	// deepseekEfforts is reasoning_effort, with "off" sent as a thinking
	// object instead — see ai.ThinkingEffortOrDisable.
	deepseekEfforts = []ai.ReasoningLevel{
		{Effort: ai.EffortOff},
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortHigh, Value: "high", Default: true},
		{Effort: ai.EffortXHigh, Value: "xhigh"},
		{Effort: ai.EffortMax, Value: "max"},
	}

	// thinkingSwitch is for endpoints whose reasoning is a boolean, sent as
	// thinking: {"type": "enabled"}. ResolveLevel snaps an in-between request
	// onto one of these.
	thinkingSwitch = []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Default: true},
		{Effort: ai.EffortHigh, Value: "enabled"},
	}

	// effortLadder is the same four rungs as plain level strings, for every
	// endpoint that takes OpenAI's spelling — including OpenRouter, which
	// normalizes its upstreams onto reasoning: {"effort": …}.
	effortLadder = []ai.ReasoningLevel{
		{Effort: ai.EffortOff, Default: true},
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortMedium, Value: "medium"},
		{Effort: ai.EffortHigh, Value: "high"},
	}

	// gptOSSEfforts is reasoning_effort for the open-weight gpt-oss models,
	// which reason unconditionally: there is no rung that turns it off, and
	// ResolveLevel snaps a request for one onto low.
	gptOSSEfforts = []ai.ReasoningLevel{
		{Effort: ai.EffortLow, Value: "low"},
		{Effort: ai.EffortMedium, Value: "medium", Default: true},
		{Effort: ai.EffortHigh, Value: "high"},
	}
)

// ─── protocol behavior ───

var (
	// Claude 4.7 and later also reject a non-default temperature.
	claudeAdaptiveNoTemp = ai.AnthropicCompat{ForceAdaptiveThinking: true, NoTemperature: true}
	claudeAdaptiveCompat = ai.AnthropicCompat{ForceAdaptiveThinking: true}
)

// ─── modalities ───

var (
	textOnly  = []ai.Modality{ai.ModalityText}
	textImage = []ai.Modality{ai.ModalityText, ai.ModalityImage}
)

// usd and cny build a rate card from per-million-token prices.
func usd(input, output, cacheWrite, cacheRead float64) ai.Pricing {
	return ai.Pricing{Currency: ai.USD, Input: input, Output: output, CacheWrite: cacheWrite, CacheRead: cacheRead}
}

func cny(input, output, cacheWrite, cacheRead float64) ai.Pricing {
	return ai.Pricing{Currency: ai.CNY, Input: input, Output: output, CacheWrite: cacheWrite, CacheRead: cacheRead}
}

// gpt5 builds an OpenAI reasoning-model entry. The whole GPT-5 family shares
// one window and output cap, and inferOpenAI already knows what they are —
// restating them here is how a row and an inference drift apart.
func gpt5(id, name string, pricing ai.Pricing) ai.Model {
	return inferOpenAI(ai.Model{
		ID: id, Name: name,
		Reasoning: openAIEfforts,
		Pricing:   pricing,
	})
}

// gpt56 is gpt5 with the extra top rung the 5.6 family accepts.
func gpt56(id, name string, pricing ai.Pricing) ai.Model {
	m := gpt5(id, name, pricing)
	m.Reasoning = openAIEffortsMax
	return m
}

// retired marks a model the vendor no longer serves, naming what replaces it.
// The entry exists so that a stale configuration produces a sentence rather
// than a 404.
func retired(id, name, replacement string) ai.Model {
	return ai.Model{
		ID: id, Name: name,
		Stage:       ai.StageRetired,
		Replacement: replacement,
		Reasoning:   noReasoning,
	}
}

// preview marks a model whose behavior may change without notice.
func preview(m ai.Model) ai.Model {
	m.Stage = ai.StagePreview
	return m
}

// gemini builds a Gemini 3 entry. The family shares one window and output cap,
// which inferGoogle states already. Google publishes no per-model rate card in
// the API docs, so no pricing is given rather than a guessed one.
func gemini(id, name string) ai.Model {
	return inferGoogle(ai.Model{ID: id, Name: name})
}

// priced attaches a rate card, by model ID, to a line of models two vendors
// share. The rows themselves say nothing about price because the two bill
// differently — the same Claude costs one thing from Anthropic and another
// through a Google contract — and that is the only thing they differ in.
func priced(models []ai.Model, prices map[string]ai.Pricing) []ai.Model {
	out := make([]ai.Model, len(models))
	for i, m := range models {
		m.Pricing = prices[m.ID]
		out[i] = m
	}
	return out
}

// ─── deployments ───

// The variables a Vertex deployment is named by. They are constants because
// the row lists them for a caller to read and vertexDeployment reads them for
// real, and the two saying different things would be undetectable.
const (
	vertexProjectEnv = "ANTHROPIC_VERTEX_PROJECT_ID"
	vertexRegionEnv  = "CLOUD_ML_REGION"
)

// vertexDeployment reads the project and region a Vertex-served model lives
// in. The project has no default worth guessing: without one the request goes
// to nobody's project, so it is refused here rather than 400 later. The
// credential is not among these — Vertex takes Application Default
// Credentials, not a variable.
func vertexDeployment(env func(string) string) (ai.ProtocolConfig, error) {
	project := strings.TrimSpace(env(vertexProjectEnv))
	if project == "" {
		return nil, &MissingDeploymentError{
			EnvVars: []string{vertexProjectEnv},
			Note: "Vertex needs a Google Cloud project. Credentials themselves come from " +
				"Application Default Credentials, not from a variable.",
		}
	}
	return ai.VertexConfig{Project: project, Region: strings.TrimSpace(env(vertexRegionEnv))}, nil
}
