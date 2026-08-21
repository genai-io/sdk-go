package ai

import (
	"fmt"
	"reflect"
	"sync"
)

// Protocol dialects: where one endpoint states how it differs from the
// protocol owner's own behaviour.
//
// Two endpoints can speak the same wire protocol and still disagree about the
// details — which field carries the reasoning switch, whether temperature is
// accepted, whether a cache TTL is understood. A Compat value is where one
// endpoint states its differences, so the driver stays one implementation
// rather than a tree of vendor special cases. Which type belongs to which
// protocol is registered below, so a model that carries the wrong one is
// reported rather than quietly ignored.

// CompatOf returns a model's protocol compatibility flags, or the zero value
// when it carries none or carries a different protocol's.
//
//	compat := ai.CompatOf[ai.AnthropicCompat](model)
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

// compatRegistry maps a protocol to the compat type it expects.
//
// The type, not a decoder: it is what both jobs need. Reading a model back
// needs to rebuild the value as that type, and validating one needs to catch a
// model carrying another protocol's compat — which CompatOf cannot report,
// because a type assertion that fails yields the zero value and the model
// simply behaves as though it had no dialect at all.
var compatRegistry = struct {
	mu sync.RWMutex
	m  map[API]reflect.Type
}{m: map[API]reflect.Type{
	APIAnthropicMessages: reflect.TypeFor[AnthropicCompat](),
	APIAnthropicVertex:   reflect.TypeFor[AnthropicCompat](),
	APIOpenAIChat:        reflect.TypeFor[OpenAIChatCompat](),
	APIOpenAIResponses:   reflect.TypeFor[OpenAIResponsesCompat](),
	APIGoogleGenAI:       reflect.TypeFor[GoogleCompat](),
}}

// RegisterCompat declares the compat type a protocol this package does not
// define uses. A driver package for a custom protocol calls it from init,
// alongside RegisterAPI.
//
//	ai.RegisterCompat[MyCompat](myAPI)
//
// It takes the type rather than a decode function because every caller wants
// the same decoding — and a type is the thing that also makes a mismatched
// compat detectable, which a decode function is not.
func RegisterCompat[T any](api API) {
	compatRegistry.mu.Lock()
	defer compatRegistry.mu.Unlock()
	compatRegistry.m[api] = reflect.TypeFor[T]()
}

// compatType reports the compat type registered for a protocol.
func compatType(api API) (reflect.Type, bool) {
	compatRegistry.mu.RLock()
	defer compatRegistry.mu.RUnlock()
	t, ok := compatRegistry.m[api]
	return t, ok
}

// checkCompat reports a compat value that does not belong to the model's
// protocol.
//
// This is the one failure CompatOf cannot surface. Setting an OpenAIChatCompat
// on an Anthropic model leaves CompatOf[AnthropicCompat] returning the zero
// value, so the model runs with first-party defaults and nothing says the
// dialect was ignored — the same silent-downgrade shape that UnmarshalJSON
// refuses. A protocol with no registered type is left alone: there is nothing
// to compare against.
func (m Model) checkCompat() error {
	if m.Compat == nil {
		return nil
	}
	want, ok := compatType(m.API)
	if !ok {
		return nil
	}
	if got := reflect.TypeOf(m.Compat); got != want {
		return &Error{Kind: KindUnsupported, Message: fmt.Sprintf(
			"model %s speaks %s, but carries %s as its compat; %s expects %s, "+
				"and the mismatch would be ignored rather than reported",
			m, m.API, got, m.API, want)}
	}
	return nil
}
