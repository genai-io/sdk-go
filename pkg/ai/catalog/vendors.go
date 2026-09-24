package catalog

import (
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Adding an OpenAI-compatible endpoint is an entry here and nothing else.
//
// A row states how to reach a vendor and speak its protocol: the endpoint, the
// credential variables, the protocol and its quirks. Which models a vendor
// serves, and everything about them, is the application's to set on ai.Model.

// verified is the date this file's figures were last checked against the
// vendors' own documentation. Every entry carries its own Verified field so
// they can drift apart; this is the date of the sweep that set them.
const verified = "2026-08-20"

// verifiedGateways is the sweep that added the OpenAI-compatible gateways and
// single-vendor endpoints below.
const verifiedGateways = "2026-08-21"

// verifiedGoogleVertex is the sweep that added Gemini through Vertex AI.
const verifiedGoogleVertex = "2026-09-17"

// verifiedHyperscalers is the sweep that added the two hyperscaler-hosted
// OpenAI endpoints below.
const verifiedHyperscalers = "2026-08-21"

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
		// Current Claude takes adaptive thinking with the level in
		// output_config.effort; an older model states its own Compat.
		Compat: claudeAdaptiveNoTemp,
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
			"project": "ANTHROPIC_VERTEX_PROJECT_ID",
			"region":  "CLOUD_ML_REGION",
		},
		Deployment: vertexDeployment,
		Compat:     claudeAdaptiveNoTemp,
		Note:       "Authenticates with Google Application Default Credentials; set ANTHROPIC_VERTEX_PROJECT_ID and, optionally, CLOUD_ML_REGION.",
	},
	{
		ID:          "openai",
		DisplayName: "OpenAI",
		Order:       20,
		Verified:    verified,
		API:         ai.APIOpenAIResponses,
		BaseURLEnv:  "OPENAI_BASE_URL",
		KeyEnv:      []string{"OPENAI_API_KEY"},
		Compat:      ai.OpenAIResponsesCompat{},
		// The endpoint reports cache writes in
		// input_tokens_details.cache_write_tokens, which the pinned openai-go
		// release does not expose, so they arrive folded into Usage.Input.
		Note: "This SDK cannot yet read the endpoint's cache-write token count; those tokens are reported as ordinary input.",
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
		// Azure lags the first-party API on newly added request fields, and
		// prompt_cache_retention is one of them. Declaring it unsupported
		// costs a longer cache lifetime; assuming it is supported costs a 400
		// on every request that asks for one, so the entry takes the first.
		Compat: ai.OpenAIResponsesCompat{NoLongCacheRetention: true},
		Note: "Set AZURE_OPENAI_ENDPOINT to the resource URL (https://YOUR-RESOURCE.openai.azure.com); " +
			"the /openai/v1 suffix is added for you. A model ID here is a deployment name chosen by " +
			"whoever created the resource, not an OpenAI model ID. " +
			"Which API versions a resource serves depends on its region.",
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
		KeyEnv: []string{"AWS_BEARER_TOKEN_BEDROCK"},
		// gpt-oss takes OpenAI's own reasoning_effort.
		Compat: ai.OpenAIChatCompat{Thinking: ai.ThinkingEffort},
		Note: "Set AWS_BEDROCK_BASE_URL to https://bedrock-runtime.REGION.amazonaws.com and " +
			"AWS_BEARER_TOKEN_BEDROCK to a Bedrock API key; the /openai/v1 suffix is added for you. " +
			"This endpoint speaks Chat Completions only — Bedrock publishes no /responses — so the " +
			"server-side tools and reasoning-item reuse of the openai entry are not reachable through it.",
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
		// Gemini 3 replaced the thinking budget with a level; a 2.5 model
		// states GoogleCompat{} to go back to a budget.
		Compat: ai.GoogleCompat{ThinkingLevel: true},
	},
	{
		ID:          "google-vertex",
		DisplayName: "Google Gemini (Vertex AI)",
		Order:       35,
		Verified:    verifiedGoogleVertex,
		API:         ai.APIGoogleVertex,
		// Vertex resolves its own endpoint from the deployment region, and
		// authenticates with Google credentials rather than a key. The two
		// variables below name the deployment, not a credential; they are the
		// ones Google's own Gen AI SDK reads.
		KeyEnv: nil,
		DeploymentEnv: map[string]string{
			"project": "GOOGLE_CLOUD_PROJECT",
			"region":  "GOOGLE_CLOUD_LOCATION",
		},
		Deployment: vertexDeployment,
		Compat:     ai.GoogleCompat{ThinkingLevel: true},
		Note:       "Authenticates with Google Application Default Credentials; set GOOGLE_CLOUD_PROJECT and, optionally, GOOGLE_CLOUD_LOCATION (default global).",
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
		// DeepSeek reasons unless told not to, so "off" has to be sent.
		Compat: ai.OpenAIChatCompat{
			Thinking: ai.ThinkingEffortOrDisable,
			// DeepSeek takes its own reasoning back on an assistant message.
			// Without this a reasoning turn cannot be replayed at all, which
			// ends any conversation that continues past the model's first
			// thinking turn.
			ReasoningContent: true,
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
		Compat: ai.OpenAIChatCompat{},
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
		Compat: ai.OpenAIChatCompat{
			Thinking: ai.ThinkingType,
			// Moonshot rejects a thinking-enabled request whose assistant
			// messages lack reasoning_content, even when it is empty.
			ReasoningContent: true,
		},
		Note: "The mainland endpoint is api.moonshot.cn; point MOONSHOT_BASE_URL at api.moonshot.ai for the international one.",
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
		Compat: ai.OpenAIChatCompat{
			Thinking: ai.ThinkingEnableFlag,
			// Model Studio takes its own reasoning back on an assistant
			// message, as DeepSeek does.
			ReasoningContent: true,
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
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingType, ReasoningContent: true},
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
		Compat:        ai.OpenAIChatCompat{},
		Note:          "A local server, so there is no credential to set.",
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
		Compat: ai.AnthropicCompat{BearerAuth: true},
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
		Compat:      ai.OpenAIChatCompat{},
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
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingReasoningObject},
		Note:        "A gateway over many upstreams; it normalizes reasoning onto reasoning:{effort}.",
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
		Compat:      ai.OpenAIChatCompat{Thinking: ai.ThinkingType, ReasoningContent: true},
		Note:        "Z.ai's international endpoint. A Coding Plan subscription uses a different path — set ZAI_BASE_URL to https://api.z.ai/api/coding/paas/v4.",
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
		Compat:      ai.OpenAIChatCompat{},
		Note:        "No reasoning switch is stated; set Compat on a model you know reasons.",
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
		Compat:      ai.OpenAIChatCompat{},
		Note:        "No reasoning switch is stated; set Compat on a model you know reasons.",
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
		Compat:      ai.OpenAIChatCompat{},
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
		Compat:      ai.OpenAIChatCompat{},
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
		Compat:      ai.OpenAIChatCompat{},
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
		Compat:      ai.OpenAIChatCompat{},
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
