package llm

import "slices"

// API is a wire protocol — the request/response shape an endpoint speaks.
//
// This, not the vendor name, is what decides which driver handles a model.
// Most vendors ship an endpoint that speaks somebody else's protocol
// (DeepSeek, Moonshot and Ollama speak OpenAI Chat Completions; MiniMax,
// Xiaomi MiMo and Volcengine speak Anthropic Messages), so a vendor is a row
// in the catalog and only a protocol needs code.
type API string

const (
	// APIAnthropicMessages is the Anthropic /v1/messages protocol.
	APIAnthropicMessages API = "anthropic-messages"
	// APIOpenAIChat is the OpenAI /v1/chat/completions protocol, the de-facto
	// interchange format.
	APIOpenAIChat API = "openai-chat-completions"
	// APIOpenAIResponses is the OpenAI /v1/responses protocol.
	APIOpenAIResponses API = "openai-responses"
	// APIGoogleGenAI is the Google Gemini generateContent protocol.
	APIGoogleGenAI API = "google-genai"
	// APIAnthropicVertex is the Anthropic Messages protocol served through
	// Google Cloud Vertex AI. The wire format is identical to
	// APIAnthropicMessages; what differs is how the client authenticates and
	// where it points, which is why it is a separate protocol only so far as
	// routing is concerned — the driver reuses the same conversion code.
	APIAnthropicVertex API = "anthropic-vertex"
)

// VertexConfig is the deployment a Vertex-served model lives in. It is passed
// as Config.Native to the anthropicvertex driver.
//
// It lives here rather than beside that driver so a caller can fill it in —
// from the environment, from a settings file — without importing the driver
// and its Google Cloud auth dependencies.
type VertexConfig struct {
	// Project is the GCP project ID serving the model.
	Project string
	// Region is the serving region. "global" is the usual choice and is what
	// an empty value resolves to.
	Region string
}

// Modality is a kind of content a model accepts as input.
//
// A list rather than a set of booleans: providers keep adding kinds, and a
// caller checking Accepts(ModalityAudio) against a model that predates audio
// gets a correct "no" without the model needing a field for it.
type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
	ModalityVideo Modality = "video"
	ModalityPDF   Modality = "pdf"
)

// ReasoningLevel is one rung of a model's reasoning ladder: what a caller asks
// for, and what this particular endpoint wants sent for it.
//
// Both halves have to be here. A rung that carried only the endpoint's literal
// ("think+", "high", "enabled") would force every caller to learn each
// vendor's vocabulary, and the same Request would stop running across
// providers — which is the entire point of the normalized ladder. A rung that
// carried only the normalized Effort would push the mapping back into driver
// code, which is where it was before.
//
// It is a slice on Model rather than a map because the order is meaningful
// (clamping walks it), because a rung needs more than one value (Anthropic's
// 4.5 generation wants a token budget where 4.6+ wants an effort string, and
// one type covers both), and because a slice serializes deterministically.
type ReasoningLevel struct {
	// Effort is the normalized rung — the caller's vocabulary.
	Effort Effort `json:"effort"`
	// Value is the literal this endpoint wants. An empty string means the
	// parameter is omitted entirely for this rung, which is how "off" is
	// expressed on an endpoint that reasons only when asked.
	Value string `json:"value,omitempty"`
	// Budget is the token budget for endpoints that take one instead of, or
	// alongside, a level. Zero means none.
	Budget int `json:"budget,omitempty"`
	// Default marks the rung used when a request leaves Effort unset. At most
	// one rung should set it; the first one wins.
	Default bool `json:"default,omitempty"`
}

// Pricing is a published rate card, per million tokens. A zero field means the
// rate is unknown or not charged, not that it is free of consequence — Cost
// simply contributes nothing for it.
//
// Currency is carried rather than assumed: several vendors publish in CNY, and
// silently summing those with USD figures would produce a number that looks
// authoritative and is meaningless.
type Pricing struct {
	Currency   string  `json:"currency,omitempty"`
	Input      float64 `json:"input,omitempty"`
	Output     float64 `json:"output,omitempty"`
	CacheWrite float64 `json:"cache_write,omitempty"`
	CacheRead  float64 `json:"cache_read,omitempty"`

	// Tiers are request-wide rate switches. The highest tier whose threshold
	// the prompt exceeds replaces the base rates for the whole request —
	// MiniMax bills M3 at double above 512k input tokens, and a flat card
	// cannot say so.
	Tiers []PricingTier `json:"tiers,omitempty"`
}

// PricingTier is a rate card that takes over above a prompt-size threshold.
type PricingTier struct {
	// AboveInputTokens is the total input token count this tier applies past.
	AboveInputTokens int     `json:"above_input_tokens"`
	Input            float64 `json:"input,omitempty"`
	Output           float64 `json:"output,omitempty"`
	CacheWrite       float64 `json:"cache_write,omitempty"`
	CacheRead        float64 `json:"cache_read,omitempty"`
}

// Currency codes used by the catalog.
const (
	USD = "USD"
	CNY = "CNY"
)

// Cost is the money breakdown of one call, in Pricing's currency.
type Cost struct {
	Currency   string  `json:"currency,omitempty"`
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheWrite float64 `json:"cache_write"`
	CacheRead  float64 `json:"cache_read"`
	Total      float64 `json:"total"`
}

// Cost prices a usage record, applying the highest matching tier. It reports
// zero when no rates are known, which callers should render as "unknown"
// rather than as free.
func (p Pricing) Cost(u Usage) Cost {
	const perMillion = 1_000_000.0

	input, output := p.Input, p.Output
	cacheWrite, cacheRead := p.CacheWrite, p.CacheRead
	// Tiers switch on the whole prompt, cached portion included: that is what
	// the vendors bill on.
	matched := -1
	for _, tier := range p.Tiers {
		if u.TotalInput() > tier.AboveInputTokens && tier.AboveInputTokens > matched {
			matched = tier.AboveInputTokens
			input, output = tier.Input, tier.Output
			cacheWrite, cacheRead = tier.CacheWrite, tier.CacheRead
		}
	}

	// A long-lifetime cache write is billed at twice the input rate, where a
	// short one costs the CacheWrite rate. Splitting them here is what keeps a
	// long-cache turn from being understated.
	long := min(max(u.CacheWrite1h, 0), u.CacheWrite)
	short := u.CacheWrite - long

	c := Cost{
		Currency:   p.Currency,
		Input:      float64(u.Input) * input / perMillion,
		Output:     float64(u.Output) * output / perMillion,
		CacheWrite: (float64(short)*cacheWrite + float64(long)*input*2) / perMillion,
		CacheRead:  float64(u.CacheRead) * cacheRead / perMillion,
	}
	c.Total = c.Input + c.Output + c.CacheWrite + c.CacheRead
	return c
}

// Known reports whether any rate is set.
func (p Pricing) Known() bool {
	return p.Input > 0 || p.Output > 0 || p.CacheWrite > 0 || p.CacheRead > 0
}

// Model is everything the SDK needs to talk to one model: which protocol
// serves it, where, and what it can do.
//
// Values come from the catalog, from a provider's live model listing, or from
// a caller who wants neither — a hand-built Model with ID, API and BaseURL set
// is enough to reach any endpoint.
type Model struct {
	ID   string `json:"id"`
	API  API    `json:"api"`
	Name string `json:"name,omitempty"`

	// Vendor is the catalog vendor this model came from, e.g. "deepseek". It
	// is informational: routing keys off API.
	Vendor string `json:"vendor,omitempty"`

	// BaseURL overrides the driver's default endpoint. Empty means the
	// protocol's own default host.
	BaseURL string `json:"base_url,omitempty"`

	// ContextWindow is the maximum input tokens, and MaxOutput the maximum
	// tokens the model will generate in one turn. Zero means unknown: callers
	// must treat that as "cannot size the window" rather than substituting a
	// guess, because acting on a guessed limit fails silently in both
	// directions.
	ContextWindow int `json:"context_window,omitempty"`
	MaxOutput     int `json:"max_output,omitempty"`

	// Input lists the content kinds this model accepts. An empty list means
	// text only.
	Input []Modality `json:"input,omitempty"`

	// Reasoning is the model's ladder, ordered from least to most effort. Nil
	// means the model does not reason.
	Reasoning []ReasoningLevel `json:"reasoning,omitempty"`

	Pricing Pricing `json:"pricing,omitempty"`

	// Unsupported records what this model cannot do. Its zero value is a
	// fully capable model.
	Unsupported Unsupported `json:"unsupported,omitempty"`

	// Stage is where the model sits in its vendor's lifecycle. The zero value
	// is a stable, generally available model.
	Stage Stage `json:"stage,omitempty"`

	// Replacement names the model to move to, for one that is deprecated or
	// retired.
	Replacement string `json:"replacement,omitempty"`

	// SamplingParams are merged into the request body verbatim, after the
	// named fields, so a custom OpenAI-compatible server (llama.cpp, vLLM,
	// SGLang) can receive parameters this SDK does not model — top_p, top_k,
	// min_p, repetition_penalty. Per-request Options.SamplingParams override
	// these key by key. Only the OpenAI-family drivers apply them.
	SamplingParams map[string]any `json:"sampling_params,omitempty"`

	// Headers are added to every request for this model, on top of whatever
	// the Config carries.
	Headers map[string]string `json:"headers,omitempty"`

	// Compat holds the protocol-specific behavior flags this endpoint needs —
	// one of AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat or
	// GoogleCompat, by value. Read it with CompatOf, which yields the zero
	// value when the model carries none, so "not stated" and "all defaults"
	// are the same thing.
	//
	// It is `any` rather than a type parameter because a []Model has to hold
	// models of different protocols; a generic Model[T] could not.
	Compat any `json:"compat,omitempty"`
}

// CompatOf returns a model's protocol compatibility flags, or the zero value
// when it carries none or carries a different protocol's.
//
//	compat := llm.CompatOf[llm.AnthropicCompat](model)
//	if compat.ForceAdaptiveThinking { … }
func CompatOf[T any](m Model) T {
	c, _ := m.Compat.(T)
	return c
}

// AnthropicCompat is the behavior an Anthropic Messages endpoint needs, beyond
// what its model ID says. Every field's zero value is the first-party
// Anthropic behavior, so a third-party endpoint states only its differences.
type AnthropicCompat struct {
	// ForceAdaptiveThinking sends thinking: {"type": "adaptive"} with the
	// level in output_config.effort, instead of a budget.
	//
	// It is a flag on the model rather than a guess from the ID because
	// nothing in an ID is reliable: Claude 4.6 and later need the adaptive
	// shape, Opus 4.7 and later reject a budget with a 400, and a corporate
	// proxy will happily serve Opus under an ID like
	// "vendor--claude-opus-latest" that no substring match would catch.
	ForceAdaptiveThinking bool `json:"force_adaptive_thinking,omitempty"`

	// BearerAuth sends the credential as Authorization: Bearer rather than in
	// x-api-key. Volcengine Ark wants this.
	BearerAuth bool `json:"bearer_auth,omitempty"`

	// NoPromptCache omits the cache_control breakpoint on the system prompt.
	// Set it for an endpoint that rejects the field.
	NoPromptCache bool `json:"no_prompt_cache,omitempty"`

	// NoTemperature omits the temperature field. Claude Opus 4.7 and later
	// reject a non-default value.
	NoTemperature bool `json:"no_temperature,omitempty"`

	// NoLongCacheRetention marks an endpoint that rejects cache_control.ttl.
	// A request asking for CacheLong falls back to the short lifetime rather
	// than failing.
	NoLongCacheRetention bool `json:"no_long_cache_retention,omitempty"`
}

// ThinkingFormat is which request field an OpenAI Chat Completions endpoint
// puts its reasoning switch in.
//
// This is a separate axis from ReasoningLevel.Value: the format says where the
// value goes, the rung says what the value is. DeepSeek needs both — "on"
// means a reasoning_effort string and "off" means a thinking object, two
// different fields, which no single value could express.
type ThinkingFormat string

const (
	// ThinkingNone means the endpoint has no reasoning switch.
	ThinkingNone ThinkingFormat = ""
	// ThinkingEffort sends reasoning_effort: <value>, omitting it when the
	// rung's value is empty. OpenAI's own spelling.
	ThinkingEffort ThinkingFormat = "effort"
	// ThinkingEffortOrDisable is ThinkingEffort, except that an empty value
	// sends thinking: {"type": "disabled"} — DeepSeek reasons unless told not
	// to, so omitting the field would leave it on.
	ThinkingEffortOrDisable ThinkingFormat = "effort_or_disable"
	// ThinkingType sends thinking: {"type": <value>}. Moonshot and Z.ai.
	ThinkingType ThinkingFormat = "thinking_type"
	// ThinkingEnableFlag sends enable_thinking: true plus thinking_budget from
	// the rung. Alibaba DashScope.
	ThinkingEnableFlag ThinkingFormat = "enable_thinking"
	// ThinkingReasoningObject sends reasoning: {"effort": <value>}, the shape
	// OpenRouter normalizes every upstream onto.
	ThinkingReasoningObject ThinkingFormat = "reasoning_object"
)

// OpenAIChatCompat is the behavior an OpenAI Chat Completions endpoint needs.
// Zero values are OpenAI's own behavior.
type OpenAIChatCompat struct {
	// Thinking is which field carries the reasoning switch.
	Thinking ThinkingFormat `json:"thinking,omitempty"`

	// ReasoningContent sends a reasoning_content field on every assistant
	// message, empty when there is none. Moonshot and Z.ai reject a
	// thinking-enabled request whose history omits it.
	ReasoningContent bool `json:"reasoning_content,omitempty"`

	// MaxTokensField names the output-cap field. Empty means
	// max_completion_tokens; set it to "max_tokens" for a server that only
	// understands the older name.
	MaxTokensField string `json:"max_tokens_field,omitempty"`

	// NoUsageInStream skips stream_options.include_usage for endpoints that
	// reject it. Token counts are then whatever the final chunk carries.
	NoUsageInStream bool `json:"no_usage_in_stream,omitempty"`

	// NoFinishReason marks an endpoint that never sends finish_reason. The
	// driver then infers tool_use or end_turn from what the turn produced.
	NoFinishReason bool `json:"no_finish_reason,omitempty"`
}

// OpenAIResponsesCompat is the behavior an OpenAI Responses endpoint needs.
type OpenAIResponsesCompat struct {
	// NoLongCacheRetention marks an endpoint that rejects
	// prompt_cache_retention. A request asking for CacheLong falls back to the
	// default lifetime rather than failing.
	NoLongCacheRetention bool `json:"no_long_cache_retention,omitempty"`

	// Stateless marks a backend that refuses server-side state, such as the
	// ChatGPT/Codex subscription endpoint: the driver sends store=false and
	// asks for encrypted reasoning content so reasoning can be replayed on the
	// next turn rather than kept by the server.
	Stateless bool `json:"stateless,omitempty"`
}

// GoogleCompat is the behavior a Google GenAI endpoint needs.
type GoogleCompat struct {
	// ThinkingLevel sends thinkingConfig.thinkingLevel — Gemini 3's control —
	// instead of the thinkingBudget that Gemini 2.5 takes.
	ThinkingLevel bool `json:"thinking_level,omitempty"`
}

// Accepts reports whether the model takes the given input kind. A model that
// lists no modalities is text-only.
func (m Model) Accepts(kind Modality) bool {
	if len(m.Input) == 0 {
		return kind == ModalityText
	}
	return slices.Contains(m.Input, kind)
}

// Reasons reports whether the model has a reasoning ladder.
func (m Model) Reasons() bool { return len(m.Reasoning) > 0 }

// Efforts returns the rungs the model advertises, least to most.
func (m Model) Efforts() []Effort {
	out := make([]Effort, len(m.Reasoning))
	for i, level := range m.Reasoning {
		out[i] = level.Effort
	}
	return out
}

// DefaultLevel returns the rung used when a request leaves Effort unset, and
// whether the model states one. A model with a ladder but no default sends
// nothing, leaving the provider's own default in place.
func (m Model) DefaultLevel() (ReasoningLevel, bool) {
	for _, level := range m.Reasoning {
		if level.Default {
			return level, true
		}
	}
	return ReasoningLevel{}, false
}

// ResolveLevel picks the rung to send for a requested effort.
//
// An exact rung wins. Otherwise the search runs *upward* first and only falls
// back downward when nothing above exists: quietly reasoning less than asked
// is the more surprising failure, so a request for "low" against an off/high
// endpoint turns reasoning on rather than off.
//
// EffortDefault yields the model's default rung, or no rung at all when it
// states none. A model with no ladder never yields a rung.
func (m Model) ResolveLevel(want Effort) (ReasoningLevel, bool) {
	if !m.Reasons() {
		return ReasoningLevel{}, false
	}
	if want == EffortDefault {
		return m.DefaultLevel()
	}
	for _, level := range m.Reasoning {
		if level.Effort == want {
			return level, true
		}
	}

	rank, ok := effortRank(want)
	if !ok {
		return m.DefaultLevel()
	}
	// Upward first.
	for _, level := range m.Reasoning {
		if r, ok := effortRank(level.Effort); ok && r > rank {
			return level, true
		}
	}
	// Nothing above: take the highest rung below.
	for i := len(m.Reasoning) - 1; i >= 0; i-- {
		if r, ok := effortRank(m.Reasoning[i].Effort); ok && r < rank {
			return m.Reasoning[i], true
		}
	}
	return m.DefaultLevel()
}

// String returns "vendor/id", or just the ID for a model with no vendor.
func (m Model) String() string {
	if m.Vendor == "" {
		return m.ID
	}
	return m.Vendor + "/" + m.ID
}
