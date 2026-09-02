package catalog

import (
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Adding an OpenAI-compatible endpoint is an entry here and nothing else.

// verified is the date this file's figures were last checked against the
// vendors' own documentation. Every entry carries its own Verified field so
// they can drift apart; this is the date of the sweep that set them.
const verified = "2026-08-20"

// verifiedGateways is the sweep that added the OpenAI-compatible gateways and
// single-vendor endpoints below.
const verifiedGateways = "2026-08-21"

// verifiedHyperscalers is the sweep that added the two hyperscaler-hosted
// OpenAI endpoints below. Neither states limits or prices in the catalog: on
// Azure the deployment decides them, and on Bedrock the model card does.
const verifiedHyperscalers = "2026-08-21"

// claudeLine is the live Claude generation as both entries below serve it: the
// same IDs, the same ladders, the same limits.
//
// The rows state no window: inferAnthropic fills the generation's shape, and a
// row restating it is a second place for the figure to be wrong.
var claudeLine = []ai.Model{
	{ID: "claude-fable-5", Name: "Claude Fable 5", Reasoning: claudeAlwaysOn},
	{ID: "claude-opus-5", Name: "Claude Opus 5"},
	{ID: "claude-opus-4-8", Name: "Claude Opus 4.8"},
	{ID: "claude-opus-4-7", Name: "Claude Opus 4.7"},
	{ID: "claude-opus-4-6", Name: "Claude Opus 4.6",
		Compat: claudeAdaptiveCompat, Reasoning: claudeAdaptive46},
	{ID: "claude-sonnet-5", Name: "Claude Sonnet 5"},
	{ID: "claude-sonnet-4-6", Name: "Claude Sonnet 4.6",
		Compat: claudeAdaptiveCompat, Reasoning: claudeAdaptive46},
}

// anthropicModels is claudeLine on the first-party API: the same rows with
// Anthropic's own rate card, plus the generations only this endpoint ever
// served.
var anthropicModels = append(priced(claudeLine, map[string]ai.Pricing{
	"claude-fable-5":    usd(10, 50, 12.50, 1.00),
	"claude-opus-5":     usd(5, 25, 6.25, 0.50),
	"claude-opus-4-8":   usd(5, 25, 6.25, 0.50),
	"claude-opus-4-7":   usd(5, 25, 6.25, 0.50),
	"claude-opus-4-6":   usd(5, 25, 6.25, 0.50),
	"claude-sonnet-5":   usd(2, 10, 2.50, 0.20),
	"claude-sonnet-4-6": usd(3, 15, 3.75, 0.30),
}),
	// Retired on the first-party API. They stay listed so a caller pointing at
	// one is told what to move to; filter them out with Stage.Available.
	retired("claude-opus-4-1-20250805", "Claude Opus 4.1", "claude-opus-5"),
	retired("claude-opus-4-20250514", "Claude Opus 4", "claude-opus-5"),
	retired("claude-sonnet-4-20250514", "Claude Sonnet 4", "claude-sonnet-5"),
	retired("claude-3-7-sonnet-20250219", "Claude Sonnet 3.7", "claude-sonnet-5"),
	retired("claude-3-5-haiku-20241022", "Claude Haiku 3.5", "claude-sonnet-5"),
)

// vendors is the directory. Order is the display order; the numbering leaves
// gaps so a vendor can be slotted in without renumbering the rest.
var vendors = []Vendor{
	{
		ID:          "anthropic",
		DisplayName: "Anthropic",
		Order:       10,
		Verified:    verified,
		API:         ai.APIAnthropicMessages,
		BaseURLEnv:  "ANTHROPIC_BASE_URL",
		KeyEnv:      []string{"ANTHROPIC_API_KEY"},
		Input:       textImage,
		// Every listed model takes adaptive thinking with the level in
		// output_config.effort. The 4.5 generation, which accepted only
		// thinking.budget_tokens, is no longer served from here.
		Compat:    claudeAdaptiveNoTemp,
		Reasoning: claudeAdaptive,
		Models:    anthropicModels,
		Infer:     inferAnthropic,
	},
	{
		ID:          "anthropic-vertex",
		DisplayName: "Anthropic (Vertex AI)",
		Order:       15,
		Verified:    verifiedGateways,
		API:         ai.APIAnthropicVertex,
		// Vertex resolves its own endpoint from the deployment region, and
		// authenticates with Google credentials rather than a key. The two
		// variables below name the deployment, not a credential.
		KeyEnv: nil,
		DeploymentEnv: map[string]string{
			"project": vertexProjectEnv,
			"region":  vertexRegionEnv,
		},
		Deployment: vertexDeployment,
		Input:      textImage,
		Compat:     claudeAdaptiveNoTemp,
		Reasoning:  claudeAdaptive,
		Note: "Authenticates with Google Application Default Credentials; set ANTHROPIC_VERTEX_PROJECT_ID and, optionally, CLOUD_ML_REGION. " +
			"Vertex still serves earlier generations under @-versioned snapshot IDs; they are not listed, and Infer sizes one if you name it.",
		// Claude 4.6 and later use dateless IDs on Vertex too, so this is the
		// first-party line — unpriced, as Vertex bills through Google.
		Models: claudeLine,
		Infer:  inferAnthropic,
	},
	{
		ID:          "openai",
		DisplayName: "OpenAI",
		Order:       20,
		Verified:    verified,
		API:         ai.APIOpenAIResponses,
		BaseURLEnv:  "OPENAI_BASE_URL",
		KeyEnv:      []string{"OPENAI_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIResponsesCompat{},
		// Only the 5.6 family is billed for cache writes, at 1.25x input; on
		// every earlier model a write is free and the rate below is zero.
		// Those writes are not yet counted: the endpoint reports them in
		// input_tokens_details.cache_write_tokens, which the pinned openai-go
		// release does not expose, so they arrive folded into Input and Cost
		// prices them at 1x. It under-reports a 5.6 cache write by a quarter.
		Note: "Cache writes are billed on GPT-5.6 and later only. This SDK cannot yet read the endpoint's " +
			"cache-write token count, so those tokens are priced as ordinary input — a quarter under the real figure.",
		Models: []ai.Model{
			gpt56("gpt-5.6-sol", "GPT-5.6 Sol", usd(5, 30, 6.25, 0.50)),
			gpt56("gpt-5.6-terra", "GPT-5.6 Terra", usd(2, 12, 2.50, 0.20)),
			gpt56("gpt-5.6-luna", "GPT-5.6 Luna", usd(0.20, 1.20, 0.25, 0.02)),
			gpt5("gpt-5.5", "GPT-5.5", usd(5, 30, 0, 0.50)),
			gpt5("gpt-5.4", "GPT-5.4", usd(2.50, 15, 0, 0.25)),
			{ID: "gpt-4.1", Name: "GPT-4.1", ContextWindow: 1_047_576, MaxOutput: 32_768,
				Reasoning: noReasoning, Pricing: usd(2, 8, 0, 0.50)},
			{ID: "gpt-4o", Name: "GPT-4o", ContextWindow: 128_000, MaxOutput: 16_384,
				Reasoning: noReasoning, Pricing: usd(2.50, 10, 0, 1.25)},
		},
		// Reasoning is per model family, not per vendor: gpt-4o does not
		// reason and gpt-5 does, so Infer decides rather than a default.
		Infer: inferOpenAI,
	},
	{
		ID:          "azure-openai",
		DisplayName: "OpenAI (Azure)",
		Order:       22,
		Verified:    verifiedHyperscalers,
		// Azure serves the Responses protocol itself under its v1 surface, so
		// this is the same driver as first-party OpenAI pointed somewhere
		// else — the endpoint and the credential header are what differ.
		API: ai.APIOpenAIResponses,
		// The host names a tenant's own resource, so there is no default to
		// fall back to and the variable has to be set. The suffix is added to
		// the bare resource URL people copy out of the portal.
		BaseURLEnv:      "AZURE_OPENAI_ENDPOINT",
		BaseURLSuffix:   "/openai/v1",
		RequiresBaseURL: true,
		KeyEnv:          []string{"AZURE_OPENAI_API_KEY"},
		Input:           textImage,
		// Azure lags the first-party API on newly added request fields, and
		// prompt_cache_retention is one of them. Declaring it unsupported
		// costs a longer cache lifetime; assuming it is supported costs a 400
		// on every request that asks for one, so the entry takes the first.
		Compat: ai.OpenAIResponsesCompat{NoLongCacheRetention: true},
		Note: "Set AZURE_OPENAI_ENDPOINT to the resource URL (https://YOUR-RESOURCE.openai.azure.com); " +
			"the /openai/v1 suffix is added for you. A model ID here is a deployment name chosen by " +
			"whoever created the resource, not an OpenAI model ID, so the catalog lists none and states " +
			"no limits — Infer recognizes a deployment left named after its model and fills them in then. " +
			"Which models and API versions a resource serves depends on its region.",
		Infer: inferOpenAI,
	},
	{
		ID:          "bedrock-openai",
		DisplayName: "OpenAI (Amazon Bedrock)",
		Order:       25,
		Verified:    verifiedHyperscalers,
		// Chat Completions, not Responses: Bedrock fronts the open-weight
		// OpenAI models with an OpenAI-compatible endpoint that offers
		// /chat/completions only. There is no /responses to point at, so the
		// reasoning-item reuse the first-party entry gets is not available
		// here — that is a property of the endpoint, not a choice.
		API: ai.APIOpenAIChat,
		// Regional host, so no default: bedrock-runtime.REGION.amazonaws.com.
		BaseURLEnv:      "AWS_BEDROCK_BASE_URL",
		BaseURLSuffix:   "/openai/v1",
		RequiresBaseURL: true,
		// A Bedrock API key, presented as a bearer token. SigV4 request
		// signing is a different credential flow and is not what this entry
		// describes.
		KeyEnv:    []string{"AWS_BEARER_TOKEN_BEDROCK"},
		Input:     textOnly,
		Reasoning: gptOSSEfforts,
		// The ladder above only reaches the wire once the dialect is stated:
		// gpt-oss takes OpenAI's own reasoning_effort.
		Compat: ai.OpenAIChatCompat{Thinking: ai.ThinkingEffort},
		Note: "Set AWS_BEDROCK_BASE_URL to https://bedrock-runtime.REGION.amazonaws.com and " +
			"AWS_BEARER_TOKEN_BEDROCK to a Bedrock API key; the /openai/v1 suffix is added for you. " +
			"This endpoint speaks Chat Completions only — Bedrock publishes no /responses — so the " +
			"server-side tools and reasoning-item reuse of the openai entry are not reachable through it. " +
			"Output caps and prices are per model card and per region, which no static table tracks, so " +
			"none are stated.",
		Models: []ai.Model{
			{ID: "openai.gpt-oss-120b-1:0", Name: "GPT-OSS 120B", ContextWindow: 128_000},
			{ID: "openai.gpt-oss-20b-1:0", Name: "GPT-OSS 20B", ContextWindow: 128_000},
		},
	},
	{
		ID:          "google",
		DisplayName: "Google Gemini",
		Order:       30,
		Verified:    verified,
		API:         ai.APIGoogleGenAI,
		// Google's own SDKs take a base URL in code, not from the environment, so
		// there is no name of theirs to follow; this is the Gemini CLI's.
		BaseURLEnv: "GEMINI_BASE_URL",
		KeyEnv:     []string{"GOOGLE_API_KEY", "GEMINI_API_KEY"},
		Input:      textImage,
		// Gemini 3 replaced the thinking budget with a level; the 2.5 entries
		// opt back to a budget through Infer.
		Compat:    ai.GoogleCompat{ThinkingLevel: true},
		Reasoning: geminiLevels,
		Models: []ai.Model{
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
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.deepseek.com",
		BaseURLEnv:  "DEEPSEEK_BASE_URL",
		KeyEnv:      []string{"DEEPSEEK_API_KEY"},
		// The Chat Completions endpoint rejects image_url content parts.
		Input: textOnly,
		// DeepSeek reasons unless told not to, so "off" has to be sent.
		Reasoning: deepseekEfforts,
		Compat: ai.OpenAIChatCompat{
			Thinking: ai.ThinkingEffortOrDisable,
			// DeepSeek takes its own reasoning back on an assistant message.
			// Without this a reasoning turn cannot be replayed at all, which
			// ends any conversation that continues past the model's first
			// thinking turn.
			ReasoningContent: true,
		},
		Note: "Prices are the standard (peak) USD rate; DeepSeek bills 50% of it off-peak " +
			"(outside 09:00-12:00 and 14:00-18:00 Beijing time), which Pricing cannot express. " +
			"The Chinese-language docs state the same card in CNY. Cache writes are billed at the ordinary input rate — a cache miss is the write — so the figure below matches Input and the cost comes out right even though Chat Completions reports no cache-write token count.",
		Models: []ai.Model{
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
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://token.sensenova.cn/v1",
		BaseURLEnv:  "SENSENOVA_BASE_URL",
		KeyEnv:      []string{"SENSENOVA_API_KEY"},
		// SenseNova also publishes an Anthropic-compatible endpoint, but as of
		// 2026-06 that one reports zero tokens in every SSE event, which makes
		// context tracking impossible. The OpenAI endpoint honours
		// stream_options.include_usage and returns real counts.
		Input:  textImage,
		Compat: ai.OpenAIChatCompat{},
		Note:   "Model IDs are confirmed; SenseNova publishes no per-model token limits, so the windows below are carried over and unverified.",
		Models: []ai.Model{
			{ID: "sensenova-6.7-flash-lite", Name: "SenseNova 6.7 Flash Lite", ContextWindow: 256_000, MaxOutput: 65_536},
			{ID: "deepseek-v4-flash", Name: "DeepSeek V4 Flash (via SenseNova)", ContextWindow: 1_000_000, MaxOutput: 384_000},
		},
	},
	{
		ID:          "minimax",
		DisplayName: "MiniMax",
		Order:       60,
		Verified:    verified,
		API:         ai.APIAnthropicMessages,
		BaseURL:     "https://api.minimaxi.com/anthropic",
		BaseURLEnv:  "MINIMAX_BASE_URL",
		KeyEnv:      []string{"MINIMAX_API_KEY"},
		Input:       textImage,
		Reasoning:   budgetLadder,
		Note: "M2.7 context windows are carried over and unverified; MiniMax publishes prices but not per-model limits. " +
			"The M-series card is a standing 50% discount off list (\"永久五折\"), and M3 bills double above 512k input tokens.",
		Models: []ai.Model{
			{ID: "MiniMax-M3", Name: "MiniMax M3", ContextWindow: 1_000_000,
				Pricing: ai.Pricing{
					Currency: ai.CNY, Input: 2.10, Output: 8.40, CacheRead: 0.42,
					// MiniMax bills double above 512k input tokens.
					Tiers: []ai.PricingTier{{
						AboveInputTokens: 512_000,
						Input:            4.20, Output: 16.80, CacheRead: 0.84,
					}},
				}},
			{ID: "MiniMax-M2.7", Name: "MiniMax M2.7", ContextWindow: 204_800, MaxOutput: 8_192,
				Pricing: cny(2.1, 8.4, 2.625, 0.42)},
			{ID: "MiniMax-M2.7-highspeed", Name: "MiniMax M2.7 Highspeed", ContextWindow: 204_800, MaxOutput: 8_192,
				Pricing: cny(4.2, 16.8, 2.625, 0.42)},
		},
		Infer: inferMiniMax,
	},
	{
		ID:          "moonshot",
		DisplayName: "Moonshot (Kimi)",
		Order:       70,
		Verified:    verified,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.moonshot.cn/v1",
		BaseURLEnv:  "MOONSHOT_BASE_URL",
		KeyEnv:      []string{"MOONSHOT_API_KEY"},
		Input:       textImage,
		Reasoning:   thinkingSwitch,
		Compat: ai.OpenAIChatCompat{
			Thinking: ai.ThinkingType,
			// Moonshot rejects a thinking-enabled request whose assistant
			// messages lack reasoning_content, even when it is empty.
			ReasoningContent: true,
		},
		Note: "Prices are the mainland CNY card served by api.moonshot.cn; the international endpoint " +
			"(api.moonshot.ai, MOONSHOT_BASE_URL) publishes the same rates in USD. Cache writes are free; " +
			"a cache hit costs a tenth of ordinary input.",
		Models: []ai.Model{
			{ID: "kimi-k3", Name: "Kimi K3", ContextWindow: 1_048_576, Pricing: cny(20, 100, 0, 2)},
		},
		Infer: inferMoonshot,
	},
	{
		ID:          "alibaba",
		DisplayName: "Alibaba (Qwen)",
		Order:       80,
		Verified:    verified,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://dashscope.aliyuncs.com/compatible-mode/v1",
		BaseURLEnv:  "DASHSCOPE_BASE_URL",
		KeyEnv:      []string{"DASHSCOPE_API_KEY"},
		Input:       textImage,
		Reasoning:   budgetLadder,
		Compat: ai.OpenAIChatCompat{
			Thinking: ai.ThinkingEnableFlag,
			// Model Studio takes its own reasoning back on an assistant
			// message, as DeepSeek does.
			ReasoningContent: true,
		},
		Note: "Model IDs are confirmed; Model Studio publishes limits and prices per model behind its console rather than in the docs, so no windows are stated here.",
		Models: []ai.Model{
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
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://open.bigmodel.cn/api/paas/v4",
		BaseURLEnv:  "BIGMODEL_BASE_URL",
		KeyEnv:      []string{"BIGMODEL_API_KEY"},
		Input:       textImage,
		Reasoning:   thinkingSwitch,
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingType, ReasoningContent: true},
		Models: []ai.Model{
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
		API:         ai.APIOpenAIChat,
		BaseURL:     "http://localhost:11434/v1",
		BaseURLEnv:  "OLLAMA_BASE_URL",
		// Ollama needs no credential. The driver still sends an Authorization
		// header, which Ollama ignores.
		BaseURLSuffix: "/v1",
		Input:         textImage,
		Compat:        ai.OpenAIChatCompat{},
		Note: "A local server, so there is no credential to set. Windows depend on the pulled model file " +
			"and the server's num_ctx, not on a published catalog; the entries below are common defaults. " +
			"The live listing is authoritative.",
		Models: []ai.Model{
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
		API:         ai.APIAnthropicMessages,
		BaseURL:     "https://api.xiaomimimo.com/anthropic",
		BaseURLEnv:  "MIMO_BASE_URL",
		KeyEnv:      []string{"MIMO_API_KEY"},
		Input:       textImage,
		Reasoning:   budgetLadder,
		Note: "Prices are MiMo's own published card, which states USD only — mimo.mi.com/docs/pricing. " +
			"A cache hit costs a fraction of a percent of a miss; cache writes are not charged.",
		Models: []ai.Model{
			{ID: "mimo-v2.5-pro", Name: "MiMo V2.5 Pro", ContextWindow: 1_048_576,
				Pricing: usd(0.435, 0.87, 0, 0.0036)},
			{ID: "mimo-v2.5", Name: "MiMo V2.5", ContextWindow: 1_048_576,
				Pricing: usd(0.14, 0.28, 0, 0.0028)},
		},
		Infer: inferMiMo,
	},
	{
		ID:          "volcengine",
		DisplayName: "Volcengine Ark",
		Order:       120,
		Verified:    verified,
		API:         ai.APIAnthropicMessages,
		BaseURL:     "https://ark.cn-beijing.volces.com/api/coding",
		BaseURLEnv:  "VOLCENGINE_BASE_URL",
		KeyEnv:      []string{"VOLCENGINE_API_KEY"},
		// Ark takes the key as a bearer token, not in x-api-key.
		Compat:    ai.AnthropicCompat{BearerAuth: true},
		Input:     textImage,
		Reasoning: budgetLadder,
		Note:      "Ark serves models through per-account endpoints, so there is no fixed catalog; windows are inferred from the model ID.",
		Infer:     inferVolcengine,
	},
	{
		ID:          "agnesai",
		DisplayName: "Agnes-AI",
		Order:       130,
		Verified:    verified,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://apihub.agnes-ai.com/v1",
		BaseURLEnv:  "AGNESAI_BASE_URL",
		KeyEnv:      []string{"AGNESAI_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "An aggregator: the models it serves and their windows come from its live listing, not from a fixed catalog.",
	},
	// ── OpenAI-compatible gateways and single-vendor endpoints ──
	{
		ID:          "openrouter",
		DisplayName: "OpenRouter",
		Order:       150,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://openrouter.ai/api/v1",
		BaseURLEnv:  "OPENROUTER_BASE_URL",
		KeyEnv:      []string{"OPENROUTER_API_KEY"},
		Input:       textImage,
		Reasoning:   effortLadder,
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingReasoningObject},
		Note:        "A gateway over many upstreams. It normalizes reasoning onto reasoning:{effort}, so the ladder is vendor-wide; windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "xai",
		DisplayName: "xAI (Grok)",
		Order:       160,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.x.ai/v1",
		BaseURLEnv:  "XAI_BASE_URL",
		KeyEnv:      []string{"XAI_API_KEY"},
		Input:       textImage,
		Reasoning:   effortLadder,
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingEffort},
	},
	{
		ID:          "zai",
		DisplayName: "Z.ai (international)",
		Order:       170,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.z.ai/api/paas/v4",
		BaseURLEnv:  "ZAI_BASE_URL",
		KeyEnv:      []string{"ZAI_API_KEY"},
		Input:       textImage,
		Reasoning:   thinkingSwitch,
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingType, ReasoningContent: true},
		Note:        "The same GLM models as the bigmodel vendor, on Z.ai's international endpoint. A Coding Plan subscription uses a different path — set ZAI_BASE_URL to https://api.z.ai/api/coding/paas/v4.",
		Infer:       inferBigModel,
	},
	{
		ID:          "groq",
		DisplayName: "Groq",
		Order:       180,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.groq.com/openai/v1",
		BaseURLEnv:  "GROQ_BASE_URL",
		KeyEnv:      []string{"GROQ_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "Reasoning support varies by hosted model, so no vendor-wide ladder is stated; set one on the Model for a model you know reasons.",
	},
	{
		ID:          "cerebras",
		DisplayName: "Cerebras",
		Order:       190,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.cerebras.ai/v1",
		BaseURLEnv:  "CEREBRAS_BASE_URL",
		KeyEnv:      []string{"CEREBRAS_API_KEY"},
		Input:       textOnly,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "Reasoning support varies by hosted model, so no vendor-wide ladder is stated.",
	},
	{
		ID:          "together",
		DisplayName: "Together AI",
		Order:       200,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.together.ai/v1",
		BaseURLEnv:  "TOGETHER_BASE_URL",
		KeyEnv:      []string{"TOGETHER_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "A host for many open models; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "fireworks",
		DisplayName: "Fireworks AI",
		Order:       210,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://api.fireworks.ai/inference/v1",
		BaseURLEnv:  "FIREWORKS_BASE_URL",
		KeyEnv:      []string{"FIREWORKS_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "A host for many open models; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "nvidia",
		DisplayName: "NVIDIA NIM",
		Order:       220,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://integrate.api.nvidia.com/v1",
		BaseURLEnv:  "NVIDIA_BASE_URL",
		KeyEnv:      []string{"NVIDIA_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "A host for many open models; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "huggingface",
		DisplayName: "Hugging Face",
		Order:       230,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		BaseURL:     "https://router.huggingface.co/v1",
		BaseURLEnv:  "HF_BASE_URL",
		KeyEnv:      []string{"HF_TOKEN", "HUGGINGFACE_API_KEY"},
		Input:       textImage,
		Compat:      ai.OpenAIChatCompat{},
		Note:        "A router over many inference providers; reasoning, windows and prices are per-model and come from the live listing.",
	},
	{
		ID:          "copilot",
		DisplayName: "GitHub Copilot",
		Order:       140,
		Verified:    verifiedGateways,
		API:         ai.APIOpenAIChat,
		// A fallback only: Copilot reveals the endpoint the account actually
		// talks to during sign-in, and an enterprise account's is not this
		// one. auth records it on the credential and prefers it.
		BaseURL: "https://api.individual.githubcopilot.com",
		Input:   textImage,
		Compat:  ai.OpenAIChatCompat{},
		Headers: copilotHeaders,
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
		API:         ai.APIOpenAIResponses,
		BaseURL:     "https://chatgpt.com/backend-api/codex",
		Input:       textImage,
		Compat:      ai.OpenAIResponsesCompat{Stateless: true},
		Note: "A ChatGPT subscription rather than an API key. Sign in with " +
			"auth.Login(ctx, \"openai-codex\", ...), which runs the PKCE browser grant. " +
			"The endpoint and client identifier come from OpenAI's Codex CLI, not from " +
			"published API documentation, and are not covered by the API's compatibility " +
			"promises.",
	},
}

// aliases keep a vendor ID that is already written into somebody's
// configuration resolving after the table changed its own spelling. A row is
// the truth and an alias only a redirection to one; Find is where they meet.
var aliases = map[string]string{
	// The MiniMax row was keyed "minmax" — a misspelling of the brand that
	// every other field on it, and the vendor's own host, spells correctly.
	"minmax": "minimax",
}

// copilotHeaders identify the caller as an editor integration. The Copilot API
// refuses requests that do not. It is not exported: a package-level map that
// anyone can write to is a process-wide setting nobody declared, and the entry
// below hands out a copy of it to whoever needs one.
var copilotHeaders = map[string]string{
	"Editor-Version":         "vscode/1.107.0",
	"Editor-Plugin-Version":  "copilot-chat/0.35.0",
	"Copilot-Integration-Id": "vscode-chat",
}
