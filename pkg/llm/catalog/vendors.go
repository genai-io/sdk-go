package catalog

import (
	"strings"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// verified is the date this file's figures were last checked against the
// vendors' own documentation. Every entry carries its own Verified field so
// they can drift apart; this is the date of the sweep that set them.
const verified = "2026-08-20"

// verifiedGateways is the sweep that added the OpenAI-compatible gateways and
// single-vendor endpoints below.
const verifiedGateways = "2026-08-21"

// ─── reasoning ladders ───
//
// A ladder is ordered least to most effort, and each rung carries what its
// endpoint actually wants: Value for a level-taking field, Budget for a
// token-taking one. Nothing here is translated in driver code — adding a
// vendor's spelling is adding rungs.

var (
	// claudeAdaptive is output_config.effort for Claude 4.7 and later, where
	// xhigh sits between high and max.
	claudeAdaptive = []llm.ReasoningLevel{
		{Effort: llm.EffortOff},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium"},
		{Effort: llm.EffortHigh, Value: "high", Default: true},
		{Effort: llm.EffortXHigh, Value: "xhigh"},
		{Effort: llm.EffortMax, Value: "max"},
	}
	// claudeAdaptive46 is the same ladder for Claude 4.6, which predates xhigh.
	claudeAdaptive46 = []llm.ReasoningLevel{
		{Effort: llm.EffortOff},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium"},
		{Effort: llm.EffortHigh, Value: "high", Default: true},
		{Effort: llm.EffortMax, Value: "max"},
	}
	// claudeAlwaysOn drops the off rung: Fable 5 reasons unconditionally and
	// rejects an explicit thinking: {"type": "disabled"} with a 400.
	claudeAlwaysOn = []llm.ReasoningLevel{
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium"},
		{Effort: llm.EffortHigh, Value: "high", Default: true},
		{Effort: llm.EffortXHigh, Value: "xhigh"},
		{Effort: llm.EffortMax, Value: "max"},
	}
	// claudeBudget is thinking.budget_tokens — the 4.5 generation, and every
	// Anthropic-compatible third-party endpoint.
	claudeBudget = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Budget: 5_000},
		{Effort: llm.EffortMedium, Budget: 32_000},
		{Effort: llm.EffortHigh, Budget: 128_000},
	}

	// openAIEfforts is reasoning.effort. "none" is a real value here, so "off"
	// is expressible rather than something the catalog has to hide.
	openAIEfforts = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Value: "none"},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium", Default: true},
		{Effort: llm.EffortHigh, Value: "high"},
		{Effort: llm.EffortXHigh, Value: "xhigh"},
	}
	// openAIEffortsMax adds the top rung the 5.6 family accepts.
	openAIEffortsMax = append(append([]llm.ReasoningLevel{}, openAIEfforts...),
		llm.ReasoningLevel{Effort: llm.EffortMax, Value: "max"})

	// geminiLevels is thinkingConfig.thinkingLevel — Gemini 3. MINIMAL exists
	// in the enum but Gemini 3 rejects it, so there is no minimal rung.
	geminiLevels = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Value: "LOW"},
		{Effort: llm.EffortMedium, Value: "MEDIUM"},
		{Effort: llm.EffortHigh, Value: "HIGH"},
	}
	// geminiBudgets is thinkingConfig.thinkingBudget — Gemini 2.5.
	geminiBudgets = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Budget: 5_000},
		{Effort: llm.EffortMedium, Budget: 32_000},
		{Effort: llm.EffortHigh, Budget: 128_000},
	}

	// deepseekEfforts is reasoning_effort, with "off" sent as a thinking
	// object instead — see llm.ThinkingEffortOrDisable.
	deepseekEfforts = []llm.ReasoningLevel{
		{Effort: llm.EffortOff},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortHigh, Value: "high", Default: true},
		{Effort: llm.EffortXHigh, Value: "xhigh"},
		{Effort: llm.EffortMax, Value: "max"},
	}

	// thinkingSwitch is for endpoints whose reasoning is a boolean, sent as
	// thinking: {"type": "enabled"}. ResolveLevel snaps an in-between request
	// onto one of these.
	thinkingSwitch = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortHigh, Value: "enabled"},
	}

	// openRouterEfforts is reasoning: {"effort": …}, which OpenRouter
	// normalizes every upstream onto.
	openRouterEfforts = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium"},
		{Effort: llm.EffortHigh, Value: "high"},
	}

	// effortLadder is plain reasoning_effort, for endpoints that take OpenAI's
	// own spelling.
	effortLadder = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Value: "low"},
		{Effort: llm.EffortMedium, Value: "medium"},
		{Effort: llm.EffortHigh, Value: "high"},
	}

	// qwenBudgets is enable_thinking plus thinking_budget.
	qwenBudgets = []llm.ReasoningLevel{
		{Effort: llm.EffortOff, Default: true},
		{Effort: llm.EffortLow, Budget: 5_000},
		{Effort: llm.EffortMedium, Budget: 32_000},
		{Effort: llm.EffortHigh, Budget: 128_000},
	}
)

// ─── protocol behavior ───

var (
	// Claude 4.7 and later also reject a non-default temperature.
	claudeAdaptiveNoTemp = llm.AnthropicCompat{ForceAdaptiveThinking: true, NoTemperature: true}
	claudeAdaptiveCompat = llm.AnthropicCompat{ForceAdaptiveThinking: true}
	claudeBudgetCompat   = llm.AnthropicCompat{}
)

// ─── modalities ───

var (
	textOnly  = []llm.Modality{llm.ModalityText}
	textImage = []llm.Modality{llm.ModalityText, llm.ModalityImage}
)

// usd and cny build a rate card from per-million-token prices.
func usd(input, output, cacheWrite, cacheRead float64) llm.Pricing {
	return llm.Pricing{Currency: llm.USD, Input: input, Output: output, CacheWrite: cacheWrite, CacheRead: cacheRead}
}

func cny(input, output, cacheWrite, cacheRead float64) llm.Pricing {
	return llm.Pricing{Currency: llm.CNY, Input: input, Output: output, CacheWrite: cacheWrite, CacheRead: cacheRead}
}

// vendors is the directory. Order is the display order; the numbering leaves
// gaps so a vendor can be slotted in without renumbering the rest.
var vendors = []Vendor{
	{
		ID:          "anthropic",
		DisplayName: "Anthropic",
		Order:       10,
		Verified:    verified,
		API:         llm.APIAnthropicMessages,
		BaseURLEnv:  "ANTHROPIC_BASE_URL",
		KeyEnv:      []string{"ANTHROPIC_API_KEY"},
		Input:       textImage,
		// Claude 4.6 and later take adaptive thinking with the level in
		// output_config.effort. The three 4.5-generation entries below opt
		// back to budget_tokens, which is all they accept.
		Compat:    claudeAdaptiveNoTemp,
		Reasoning: claudeAdaptive,
		Models: []llm.Model{
			{ID: "claude-fable-5", Name: "Claude Fable 5", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Reasoning: claudeAlwaysOn, Pricing: usd(10, 50, 12.50, 1.00)},
			{ID: "claude-opus-5", Name: "Claude Opus 5", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Pricing: usd(5, 25, 6.25, 0.50)},
			{ID: "claude-opus-4-8", Name: "Claude Opus 4.8", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Pricing: usd(5, 25, 6.25, 0.50)},
			{ID: "claude-opus-4-7", Name: "Claude Opus 4.7", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Pricing: usd(5, 25, 6.25, 0.50)},
			{ID: "claude-opus-4-6", Name: "Claude Opus 4.6", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Compat: claudeAdaptiveCompat, Reasoning: claudeAdaptive46, Pricing: usd(5, 25, 6.25, 0.50)},
			{ID: "claude-sonnet-5", Name: "Claude Sonnet 5", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Pricing: usd(2, 10, 2.50, 0.20)},
			{ID: "claude-sonnet-4-6", Name: "Claude Sonnet 4.6", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Compat: claudeAdaptiveCompat, Reasoning: claudeAdaptive46, Pricing: usd(3, 15, 3.75, 0.30)},
			// The 4.5 generation predates adaptive thinking.
			{ID: "claude-opus-4-5", Name: "Claude Opus 4.5", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget, Pricing: usd(5, 25, 6.25, 0.50)},
			{ID: "claude-sonnet-4-5", Name: "Claude Sonnet 4.5", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget, Pricing: usd(3, 15, 3.75, 0.30)},
			{ID: "claude-haiku-4-5", Name: "Claude Haiku 4.5", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget, Pricing: usd(1, 5, 1.25, 0.10)},

			// Retired on the first-party API. They stay listed so a caller
			// still pointing at one is told what happened and what to move to,
			// instead of meeting an opaque rejection from the endpoint. Filter
			// them out of a picker with Stage.Available.
			retired("claude-opus-4-1-20250805", "Claude Opus 4.1", "claude-opus-5"),
			retired("claude-opus-4-20250514", "Claude Opus 4", "claude-opus-5"),
			retired("claude-sonnet-4-20250514", "Claude Sonnet 4", "claude-sonnet-5"),
			retired("claude-3-7-sonnet-20250219", "Claude Sonnet 3.7", "claude-sonnet-5"),
			retired("claude-3-5-haiku-20241022", "Claude Haiku 3.5", "claude-haiku-4-5"),
		},
		Infer: inferAnthropic,
	},
	{
		ID:          "anthropic-vertex",
		DisplayName: "Anthropic (Vertex AI)",
		Order:       15,
		Verified:    verifiedGateways,
		API:         llm.APIAnthropicVertex,
		// Vertex resolves its own endpoint from the deployment region, and
		// authenticates with Google credentials rather than a key. The two
		// variables below name the deployment, not a credential.
		KeyEnv: nil,
		DeploymentEnv: map[string]string{
			"project": "ANTHROPIC_VERTEX_PROJECT_ID",
			"region":  "CLOUD_ML_REGION",
		},
		Input:     textImage,
		Compat:    claudeAdaptiveNoTemp,
		Reasoning: claudeAdaptive,
		Note: "Authenticates with Google Application Default Credentials; set ANTHROPIC_VERTEX_PROJECT_ID and, optionally, CLOUD_ML_REGION. " +
			"The list is longer than the first-party one: Opus 4.1, Opus 4 and Sonnet 4 are retired on the Claude API but remain on Google Cloud.",
		Models: []llm.Model{
			// Claude 4.6 and later use dateless IDs on Vertex too.
			{ID: "claude-fable-5", Name: "Claude Fable 5", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Reasoning: claudeAlwaysOn},
			{ID: "claude-opus-5", Name: "Claude Opus 5", ContextWindow: 1_000_000, MaxOutput: 128_000},
			{ID: "claude-opus-4-8", Name: "Claude Opus 4.8", ContextWindow: 1_000_000, MaxOutput: 128_000},
			{ID: "claude-opus-4-7", Name: "Claude Opus 4.7", ContextWindow: 1_000_000, MaxOutput: 128_000},
			{ID: "claude-opus-4-6", Name: "Claude Opus 4.6", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Compat: claudeAdaptiveCompat, Reasoning: claudeAdaptive46},
			{ID: "claude-sonnet-5", Name: "Claude Sonnet 5", ContextWindow: 1_000_000, MaxOutput: 128_000},
			{ID: "claude-sonnet-4-6", Name: "Claude Sonnet 4.6", ContextWindow: 1_000_000, MaxOutput: 128_000,
				Compat: claudeAdaptiveCompat, Reasoning: claudeAdaptive46},
			// Earlier generations keep the @-versioned snapshot form.
			{ID: "claude-opus-4-5@20251101", Name: "Claude Opus 4.5", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget},
			{ID: "claude-sonnet-4-5@20250929", Name: "Claude Sonnet 4.5", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget},
			{ID: "claude-haiku-4-5@20251001", Name: "Claude Haiku 4.5", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget},
			{ID: "claude-opus-4-1@20250805", Name: "Claude Opus 4.1", ContextWindow: 200_000, MaxOutput: 32_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget},
			{ID: "claude-opus-4@20250514", Name: "Claude Opus 4", ContextWindow: 200_000, MaxOutput: 32_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget},
			{ID: "claude-sonnet-4@20250514", Name: "Claude Sonnet 4", ContextWindow: 200_000, MaxOutput: 64_000,
				Compat: claudeBudgetCompat, Reasoning: claudeBudget},
		},
		Infer: inferAnthropic,
	},
	{
		ID:          "openai",
		DisplayName: "OpenAI",
		Order:       20,
		Verified:    verified,
		API:         llm.APIOpenAIResponses,
		BaseURLEnv:  "OPENAI_BASE_URL",
		KeyEnv:      []string{"OPENAI_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIResponsesCompat{},
		Models: []llm.Model{
			gpt56("gpt-5.6-sol", "GPT-5.6 Sol", usd(5, 30, 6.25, 0.50)),
			gpt56("gpt-5.6-terra", "GPT-5.6 Terra", usd(2, 12, 2.50, 0.20)),
			gpt56("gpt-5.6-luna", "GPT-5.6 Luna", usd(0.20, 1.20, 0.25, 0.02)),
			gpt5("gpt-5.5", "GPT-5.5", usd(5, 30, 6.25, 0.50)),
			gpt5("gpt-5.4", "GPT-5.4", usd(2.50, 15, 3.125, 0.25)),
			{ID: "gpt-4.1", Name: "GPT-4.1", ContextWindow: 1_047_576, MaxOutput: 32_768,
				Reasoning: NoReasoning, Pricing: usd(2, 8, 2.50, 0.50)},
			{ID: "gpt-4o", Name: "GPT-4o", ContextWindow: 128_000, MaxOutput: 16_384,
				Reasoning: NoReasoning, Pricing: usd(2.50, 10, 3.125, 1.25)},
		},
		// Reasoning is per model family, not per vendor: gpt-4o does not
		// reason and gpt-5 does, so Infer decides rather than a default.
		Infer: inferOpenAI,
	},
	{
		ID:          "google",
		DisplayName: "Google Gemini",
		Order:       30,
		Verified:    verified,
		API:         llm.APIGoogleGenAI,
		KeyEnv:      []string{"GOOGLE_API_KEY", "GEMINI_API_KEY"},
		Input:       textImage,
		// Gemini 3 replaced the thinking budget with a level; the 2.5 entries
		// opt back to a budget through Infer.
		Compat:    llm.GoogleCompat{ThinkingLevel: true},
		Reasoning: geminiLevels,
		Models: []llm.Model{
			gemini("gemini-3.7-flash", "Gemini 3.7 Flash"),
			gemini("gemini-3.6-flash", "Gemini 3.6 Flash"),
			gemini("gemini-3.5-flash", "Gemini 3.5 Flash"),
			gemini("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
			gemini("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"),
			preview(gemini("gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)")),
			preview(gemini("gemini-3-flash-preview", "Gemini 3 Flash (preview)")),
		},
		Infer: inferGoogle,
	},
	{
		ID:          "deepseek",
		DisplayName: "DeepSeek",
		Order:       40,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.deepseek.com",
		BaseURLEnv:  "DEEPSEEK_BASE_URL",
		KeyEnv:      []string{"DEEPSEEK_API_KEY"},
		// The Chat Completions endpoint rejects image_url content parts.
		Input: textOnly,
		// DeepSeek reasons unless told not to, so "off" has to be sent.
		Reasoning: deepseekEfforts,
		Compat:    llm.OpenAIChatCompat{Thinking: llm.ThinkingEffortOrDisable},
		Note: "Prices are the standard (peak) USD rate; DeepSeek bills 50% of it off-peak " +
			"(outside 09:00-12:00 and 14:00-18:00 Beijing time), which Pricing cannot express. " +
			"The Chinese-language docs state the same card in CNY.",
		Models: []llm.Model{
			// CacheWrite mirrors Input: a cache miss is the write, and
			// DeepSeek bills no separate write.
			{ID: "deepseek-v4-flash", Name: "DeepSeek V4 Flash", ContextWindow: 1_000_000, MaxOutput: 384_000,
				Pricing: usd(0.44, 1.32, 0.44, 0.014)},
			{ID: "deepseek-v4-pro", Name: "DeepSeek V4 Pro", ContextWindow: 1_000_000, MaxOutput: 384_000,
				Pricing: usd(1.32, 3.96, 1.32, 0.044)},
		},
	},
	{
		ID:          "sensenova",
		DisplayName: "SenseNova",
		Order:       50,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://token.sensenova.cn/v1",
		BaseURLEnv:  "SENSENOVA_BASE_URL",
		KeyEnv:      []string{"SENSENOVA_API_KEY"},
		// SenseNova also publishes an Anthropic-compatible endpoint, but as of
		// 2026-06 that one reports zero tokens in every SSE event, which makes
		// context tracking impossible. The OpenAI endpoint honours
		// stream_options.include_usage and returns real counts.
		Input:  textImage,
		Compat: llm.OpenAIChatCompat{},
		Note:   "Model IDs are confirmed; SenseNova publishes no per-model token limits, so the windows below are carried over and unverified.",
		Models: []llm.Model{
			{ID: "sensenova-6.7-flash-lite", Name: "SenseNova 6.7 Flash Lite", ContextWindow: 256_000, MaxOutput: 65_536},
			{ID: "deepseek-v4-flash", Name: "DeepSeek V4 Flash (via SenseNova)", ContextWindow: 1_000_000, MaxOutput: 384_000},
		},
	},
	{
		ID:          "minmax",
		DisplayName: "MiniMax",
		Order:       60,
		Verified:    verified,
		API:         llm.APIAnthropicMessages,
		BaseURL:     "https://api.minimaxi.com/anthropic",
		BaseURLEnv:  "MINIMAX_BASE_URL",
		KeyEnv:      []string{"MINIMAX_API_KEY"},
		Input:       textImage,
		Compat:      claudeBudgetCompat,
		Reasoning:   claudeBudget,
		Note:        "M2.7 context windows are carried over and unverified; MiniMax publishes prices but not per-model limits.",
		Models: []llm.Model{
			{ID: "MiniMax-M3", Name: "MiniMax M3", ContextWindow: 1_000_000,
				Pricing: llm.Pricing{
					Currency: llm.CNY, Input: 2.10, Output: 8.40, CacheRead: 0.42,
					// MiniMax bills double above 512k input tokens.
					Tiers: []llm.PricingTier{{
						AboveInputTokens: 512_000,
						Input:            4.20, Output: 16.80, CacheRead: 0.84,
					}},
				}},
			{ID: "MiniMax-M2.7", Name: "MiniMax M2.7", ContextWindow: 204_800, MaxOutput: 8_192,
				Pricing: cny(2.1, 8.4, 2.625, 0.42)},
			{ID: "MiniMax-M2.7-highspeed", Name: "MiniMax M2.7 Highspeed", ContextWindow: 204_800, MaxOutput: 8_192,
				Pricing: cny(4.2, 16.8, 2.625, 0.42)},
		},
	},
	{
		ID:          "moonshot",
		DisplayName: "Moonshot (Kimi)",
		Order:       70,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.moonshot.cn/v1",
		BaseURLEnv:  "MOONSHOT_BASE_URL",
		KeyEnv:      []string{"MOONSHOT_API_KEY"},
		Input:       textImage,
		Reasoning:   thinkingSwitch,
		Compat: llm.OpenAIChatCompat{
			Thinking: llm.ThinkingType,
			// Moonshot rejects a thinking-enabled request whose assistant
			// messages lack reasoning_content, even when it is empty.
			ReasoningContent: true,
		},
		Models: []llm.Model{
			{ID: "kimi-k3", Name: "Kimi K3", ContextWindow: 1_048_576, Pricing: cny(20, 100, 0, 2)},
		},
		Infer: inferMoonshot,
	},
	{
		ID:          "alibaba",
		DisplayName: "Alibaba (Qwen)",
		Order:       80,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://dashscope.aliyuncs.com/compatible-mode/v1",
		BaseURLEnv:  "DASHSCOPE_BASE_URL",
		KeyEnv:      []string{"DASHSCOPE_API_KEY"},
		Input:       textImage,
		Reasoning:   qwenBudgets,
		Compat:      llm.OpenAIChatCompat{Thinking: llm.ThinkingEnableFlag},
		Note:        "Model IDs are confirmed; Model Studio publishes limits and prices per model behind its console rather than in the docs, so no windows are stated here.",
		Models: []llm.Model{
			{ID: "qwen3.8-max", Name: "Qwen3.8 Max"},
			{ID: "qwen3.7-plus", Name: "Qwen3.7 Plus"},
			{ID: "qwen3.7-flash", Name: "Qwen3.7 Flash"},
		},
	},
	{
		ID:          "bigmodel",
		DisplayName: "Z.ai (GLM)",
		Order:       90,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://open.bigmodel.cn/api/paas/v4",
		BaseURLEnv:  "BIGMODEL_BASE_URL",
		KeyEnv:      []string{"BIGMODEL_API_KEY"},
		Input:       textImage,
		Reasoning:   thinkingSwitch,
		Compat:      llm.OpenAIChatCompat{Thinking: llm.ThinkingType, ReasoningContent: true},
		Models: []llm.Model{
			{ID: "glm-5", Name: "GLM-5", ContextWindow: 200_000, MaxOutput: 128_000},
			{ID: "glm-4.7", Name: "GLM-4.7", ContextWindow: 200_000, MaxOutput: 128_000},
			{ID: "glm-4.7-flashx", Name: "GLM-4.7-FlashX", ContextWindow: 200_000, MaxOutput: 128_000},
		},
		Infer: inferBigModel,
	},
	{
		ID:          "ollama",
		DisplayName: "Ollama (local)",
		Order:       100,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "http://localhost:11434/v1",
		BaseURLEnv:  "OLLAMA_BASE_URL",
		// Ollama needs no credential. The driver still sends an Authorization
		// header, which Ollama ignores.
		BaseURLSuffix: "/v1",
		Input:         textImage,
		Compat:        llm.OpenAIChatCompat{},
		Note: "Windows depend on the pulled model file and the server's num_ctx, not on a published catalog; " +
			"the entries below are common defaults. The live listing is authoritative.",
		Models: []llm.Model{
			{ID: "llama4", Name: "Llama 4", ContextWindow: 131_072, MaxOutput: 16_384},
			{ID: "qwq", Name: "QwQ", ContextWindow: 131_072, MaxOutput: 16_384},
			{ID: "gemma3", Name: "Gemma 3", ContextWindow: 131_072, MaxOutput: 16_384},
			{ID: "mistral", Name: "Mistral", ContextWindow: 131_072, MaxOutput: 16_384},
		},
	},
	{
		ID:          "mimo",
		DisplayName: "Xiaomi MiMo",
		Order:       110,
		Verified:    verified,
		API:         llm.APIAnthropicMessages,
		BaseURL:     "https://api.xiaomimimo.com/anthropic",
		BaseURLEnv:  "MIMO_BASE_URL",
		KeyEnv:      []string{"MIMO_API_KEY"},
		Input:       textImage,
		Compat:      claudeBudgetCompat,
		Reasoning:   claudeBudget,
		Note:        "Prices are the overseas USD card; the mainland card is in CNY. Cache writes are currently free.",
		Models: []llm.Model{
			{ID: "mimo-v2.5-pro", Name: "MiMo V2.5 Pro", ContextWindow: 1_048_576,
				Pricing: usd(0.435, 0.87, 0, 0.0036)},
			{ID: "mimo-v2.5", Name: "MiMo V2.5", ContextWindow: 1_048_576,
				Pricing: usd(0.14, 0.28, 0, 0.0028)},
		},
	},
	{
		ID:          "volcengine",
		DisplayName: "Volcengine Ark",
		Order:       120,
		Verified:    verified,
		API:         llm.APIAnthropicMessages,
		BaseURL:     "https://ark.cn-beijing.volces.com/api/coding",
		BaseURLEnv:  "VOLCENGINE_BASE_URL",
		KeyEnv:      []string{"VOLCENGINE_API_KEY"},
		// Ark takes the key as a bearer token, not in x-api-key.
		Compat:    llm.AnthropicCompat{BearerAuth: true},
		Input:     textImage,
		Reasoning: claudeBudget,
		Note:      "Ark serves models through per-account endpoints, so there is no fixed catalog; windows are inferred from the model ID.",
		Infer:     inferVolcengine,
	},
	{
		ID:          "agnesai",
		DisplayName: "Agnes-AI",
		Order:       130,
		Verified:    verified,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://apihub.agnes-ai.com/v1",
		BaseURLEnv:  "AGNESAI_BASE_URL",
		KeyEnv:      []string{"AGNESAI_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "An aggregator: the models it serves and their windows come from its live listing, not from a fixed catalog.",
	},
	// ── OpenAI-compatible gateways and single-vendor endpoints ──
	//
	// Every entry below is data alone: no new protocol code, because they all
	// speak Chat Completions. That is the point of splitting protocol from
	// vendor — reaching a new endpoint is a row.
	//
	// The aggregators state no reasoning ladder. They serve models from many
	// upstreams with different reasoning controls, so one vendor-wide ladder
	// would be wrong for most of them; a caller who knows their model states
	// it on the Model. Windows and prices are likewise per-model and come from
	// the live listing.
	{
		ID:          "openrouter",
		DisplayName: "OpenRouter",
		Order:       150,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://openrouter.ai/api/v1",
		BaseURLEnv:  "OPENROUTER_BASE_URL",
		KeyEnv:      []string{"OPENROUTER_API_KEY"},
		Input:       textImage,
		Reasoning:   openRouterEfforts,
		Compat:      llm.OpenAIChatCompat{Thinking: llm.ThinkingReasoningObject},
		Note:        "A gateway over many upstreams. It normalizes reasoning onto reasoning:{effort}, so the ladder is vendor-wide; windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "xai",
		DisplayName: "xAI (Grok)",
		Order:       160,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.x.ai/v1",
		BaseURLEnv:  "XAI_BASE_URL",
		KeyEnv:      []string{"XAI_API_KEY"},
		Input:       textImage,
		Reasoning:   effortLadder,
		Compat:      llm.OpenAIChatCompat{Thinking: llm.ThinkingEffort},
	},
	{
		ID:          "zai",
		DisplayName: "Z.ai (international)",
		Order:       170,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.z.ai/api/paas/v4",
		BaseURLEnv:  "ZAI_BASE_URL",
		KeyEnv:      []string{"ZAI_API_KEY"},
		Input:       textImage,
		Reasoning:   thinkingSwitch,
		Compat:      llm.OpenAIChatCompat{Thinking: llm.ThinkingType, ReasoningContent: true},
		Note:        "The same GLM models as the bigmodel vendor, on Z.ai's international endpoint. A Coding Plan subscription uses a different path — set ZAI_BASE_URL to https://api.z.ai/api/coding/paas/v4.",
		Infer:       inferBigModel,
	},
	{
		ID:          "groq",
		DisplayName: "Groq",
		Order:       180,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.groq.com/openai/v1",
		BaseURLEnv:  "GROQ_BASE_URL",
		KeyEnv:      []string{"GROQ_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "Reasoning support varies by hosted model, so no vendor-wide ladder is stated; set one on the Model for a model you know reasons.",
	},
	{
		ID:          "cerebras",
		DisplayName: "Cerebras",
		Order:       190,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.cerebras.ai/v1",
		BaseURLEnv:  "CEREBRAS_BASE_URL",
		KeyEnv:      []string{"CEREBRAS_API_KEY"},
		Input:       textOnly,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "Reasoning support varies by hosted model, so no vendor-wide ladder is stated.",
	},
	{
		ID:          "together",
		DisplayName: "Together AI",
		Order:       200,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.together.ai/v1",
		BaseURLEnv:  "TOGETHER_BASE_URL",
		KeyEnv:      []string{"TOGETHER_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "A host for many open models; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "fireworks",
		DisplayName: "Fireworks AI",
		Order:       210,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://api.fireworks.ai/inference/v1",
		BaseURLEnv:  "FIREWORKS_BASE_URL",
		KeyEnv:      []string{"FIREWORKS_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "A host for many open models; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "nvidia",
		DisplayName: "NVIDIA NIM",
		Order:       220,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://integrate.api.nvidia.com/v1",
		BaseURLEnv:  "NVIDIA_BASE_URL",
		KeyEnv:      []string{"NVIDIA_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "A host for many open models; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "huggingface",
		DisplayName: "Hugging Face",
		Order:       230,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		BaseURL:     "https://router.huggingface.co/v1",
		BaseURLEnv:  "HF_BASE_URL",
		KeyEnv:      []string{"HF_TOKEN", "HUGGINGFACE_API_KEY"},
		Input:       textImage,
		Compat:      llm.OpenAIChatCompat{},
		Note:        "A router over many inference providers; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "copilot",
		DisplayName: "GitHub Copilot",
		Order:       140,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIChat,
		// A fallback only: Copilot reveals the endpoint the account actually
		// talks to during sign-in, and an enterprise account's is not this
		// one. auth records it on the credential and prefers it.
		BaseURL: "https://api.individual.githubcopilot.com",
		Input:   textImage,
		Compat:  llm.OpenAIChatCompat{},
		Headers: CopilotHeaders,
		Note: "Copilot authenticates a person, not a service: there is no API key. " +
			"Sign in with auth.Login(ctx, \"copilot\", ...), which runs GitHub's device-code " +
			"grant and stores the result. The token it issues lasts about half an hour and " +
			"is renewed for you.",
	},
	{
		ID:          "openai-codex",
		DisplayName: "ChatGPT (Codex)",
		Order:       145,
		Verified:    verifiedGateways,
		API:         llm.APIOpenAIResponses,
		BaseURL:     "https://chatgpt.com/backend-api/codex",
		Input:       textImage,
		Compat:      llm.OpenAIResponsesCompat{Stateless: true},
		Note: "A ChatGPT subscription rather than an API key. Sign in with " +
			"auth.Login(ctx, \"openai-codex\", ...), which runs the PKCE browser grant. " +
			"The endpoint and client identifier come from OpenAI's Codex CLI, not from " +
			"published API documentation, and are not covered by the API's compatibility " +
			"promises.",
	},
}

// CopilotHeaders identify the caller as an editor integration. The Copilot API
// refuses requests that do not.
//
// They live here rather than in auth because both the token exchange and every
// subsequent request need them, and catalog is the package auth already
// depends on.
var CopilotHeaders = map[string]string{
	"Editor-Version":         "vscode/1.107.0",
	"Editor-Plugin-Version":  "copilot-chat/0.35.0",
	"Copilot-Integration-Id": "vscode-chat",
}

// gpt5 builds an OpenAI reasoning-model entry. The whole GPT-5 family shares
// one window and output cap.
func gpt5(id, name string, pricing llm.Pricing) llm.Model {
	return llm.Model{
		ID: id, Name: name,
		ContextWindow: 1_050_000, MaxOutput: 128_000,
		Reasoning: openAIEfforts,
		Pricing:   pricing,
	}
}

// gpt56 is gpt5 with the extra top rung the 5.6 family accepts.
func gpt56(id, name string, pricing llm.Pricing) llm.Model {
	m := gpt5(id, name, pricing)
	m.Reasoning = openAIEffortsMax
	return m
}

// retired marks a model the vendor no longer serves, naming what replaces it.
// The entry exists so that a stale configuration produces a sentence rather
// than a 404.
func retired(id, name, replacement string) llm.Model {
	return llm.Model{
		ID: id, Name: name,
		Stage:       llm.StageRetired,
		Replacement: replacement,
		Reasoning:   NoReasoning,
	}
}

// preview marks a model whose behavior may change without notice.
func preview(m llm.Model) llm.Model {
	m.Stage = llm.StagePreview
	return m
}

// gemini builds a Gemini 3 entry. The family shares one window and output cap;
// Google publishes no per-model rate card in the API docs, so no pricing is
// stated rather than a guessed one.
func gemini(id, name string) llm.Model {
	return llm.Model{ID: id, Name: name, ContextWindow: 1_048_576, MaxOutput: 65_536}
}

// inferAnthropic sizes a Claude model the table does not list. Claude IDs are
// stable and long-lived, so an unlisted one is nearly always a model newer
// than this catalog: assume the current generation's shape rather than
// reporting nothing.
func inferAnthropic(m llm.Model) llm.Model {
	if !strings.HasPrefix(strings.ToLower(m.ID), "claude-") {
		return m
	}
	m.ContextWindow = orDefault(m.ContextWindow, 1_000_000)
	m.MaxOutput = orDefault(m.MaxOutput, 128_000)
	return m
}

// inferOpenAI fills limits and reasoning for a model ID the table does not
// list. The /v1/models listing carries neither, so both come from the
// published per-model pages.
func inferOpenAI(m llm.Model) llm.Model {
	id := strings.ToLower(strings.TrimSpace(m.ID))
	reasons := false
	switch {
	case strings.HasPrefix(id, "gpt-6"), strings.HasPrefix(id, "gpt-5"):
		m.ContextWindow, m.MaxOutput, reasons = orDefault(m.ContextWindow, 1_050_000), orDefault(m.MaxOutput, 128_000), true
	case strings.HasPrefix(id, "gpt-4.1"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 1_047_576), orDefault(m.MaxOutput, 32_768)
	case strings.HasPrefix(id, "gpt-4o"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 128_000), orDefault(m.MaxOutput, 16_384)
	}
	if m.Reasoning != nil {
		return m
	}
	if reasons {
		m.Reasoning = openAIEfforts
		return m
	}
	m.Reasoning = NoReasoning
	return m
}

// inferGoogle sizes a Gemini model and picks its thinking dialect. Gemini 3
// takes a thinking level; 2.5 still takes a budget.
func inferGoogle(m llm.Model) llm.Model {
	id := strings.ToLower(m.ID)
	switch {
	case strings.HasPrefix(id, "gemini-3"):
		m.ContextWindow = orDefault(m.ContextWindow, 1_048_576)
		m.MaxOutput = orDefault(m.MaxOutput, 65_536)
	case strings.HasPrefix(id, "gemini-2.5"):
		m.ContextWindow = orDefault(m.ContextWindow, 1_048_576)
		m.MaxOutput = orDefault(m.MaxOutput, 65_536)
		m.Compat = llm.GoogleCompat{}
		m.Reasoning = geminiBudgets
	}
	return m
}

// inferMoonshot reads the window out of a Kimi model ID, which is where
// Moonshot puts it when /v1/models omits context_length.
func inferMoonshot(m llm.Model) llm.Model {
	id := strings.ToLower(m.ID)
	switch {
	case strings.Contains(id, "k3"):
		m.ContextWindow = orDefault(m.ContextWindow, 1_048_576)
	case strings.Contains(id, "k2"):
		m.ContextWindow = orDefault(m.ContextWindow, 262_144)
	case strings.Contains(id, "128k"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 131_072), orDefault(m.MaxOutput, 8_192)
	case strings.Contains(id, "32k"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 32_768), orDefault(m.MaxOutput, 8_192)
	case strings.Contains(id, "8k"):
		m.ContextWindow, m.MaxOutput = orDefault(m.ContextWindow, 8_192), orDefault(m.MaxOutput, 3_000)
	}
	return m
}

// inferBigModel sizes a GLM model. BigModel's /v1/models follows the bare
// OpenAI shape — id, object, owned_by — and publishes no window at all. Only
// the generations whose docs state 200K are sized; anything else reports
// unknown rather than inheriting a figure that was never checked.
func inferBigModel(m llm.Model) llm.Model {
	id := strings.ToLower(m.ID)
	if strings.HasPrefix(id, "glm-5") || strings.HasPrefix(id, "glm-4.7") || strings.HasPrefix(id, "glm-4.6") {
		m.ContextWindow = orDefault(m.ContextWindow, 200_000)
		m.MaxOutput = orDefault(m.MaxOutput, 128_000)
	}
	return m
}

// inferVolcengine sizes a Doubao model. Ark encodes the window in the model ID
// suffix; the seed generation publishes a 1024k window and 256k output.
func inferVolcengine(m llm.Model) llm.Model {
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
