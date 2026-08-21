package llm

// StopReason says why the model stopped generating.
type StopReason string

const (
	// StopEndTurn is a natural completion.
	StopEndTurn StopReason = "end_turn"
	// StopToolUse means the model is waiting on tool results.
	StopToolUse StopReason = "tool_use"
	// StopMaxTokens means the output limit was reached mid-answer.
	StopMaxTokens StopReason = "max_tokens"
	// StopSequence means a caller-supplied stop sequence was emitted.
	StopSequence StopReason = "stop_sequence"
	// StopRefusal means the model declined to answer.
	StopRefusal StopReason = "refusal"
	// StopError means the turn ended on a provider error.
	StopError StopReason = "error"
	// StopAborted means the caller's context ended the turn. It is separate
	// from StopError because it is not a failure to report, retry or
	// investigate — it is what the caller asked for.
	StopAborted StopReason = "aborted"
)

// Usage is the token accounting for one call.
//
// Input counts only the fresh, uncached prompt tokens; the cached portion is
// reported separately under CacheWrite and CacheRead. Protocols that report a
// single combined prompt figure (OpenAI's prompt_tokens) are split by their
// driver, so TotalInput is always the true prompt size and per-turn sums never
// count a re-read cache twice.
type Usage struct {
	Input      int `json:"input"`
	Output     int `json:"output"`
	CacheWrite int `json:"cache_write,omitempty"`
	CacheRead  int `json:"cache_read,omitempty"`

	// CacheWrite1h is the slice of CacheWrite written with a long lifetime,
	// not an addition to it. Anthropic bills those at twice the input rate
	// where a short write costs 1.25x, so a cost that ignores the split
	// understates a long-cache turn. Only Anthropic reports it.
	CacheWrite1h int `json:"cache_write_1h,omitempty"`

	// Reasoning is the thinking tokens the provider reported, when it reports
	// them at all. It is a subset of Output, not an addition to it: Output
	// already counts these. Zero means either none or not reported, which the
	// provider does not distinguish either.
	Reasoning int `json:"reasoning,omitempty"`
}

// TotalInput is the full prompt the model processed: fresh tokens plus the
// cached prefix. This is the figure that reflects context-window occupancy;
// Input alone undercounts it whenever prompt caching is active.
func (u Usage) TotalInput() int { return u.Input + u.CacheWrite + u.CacheRead }

// Total is every token billed for the call.
func (u Usage) Total() int { return u.TotalInput() + u.Output }

// Add accumulates another call's usage.
func (u *Usage) Add(other Usage) {
	u.Input += other.Input
	u.Output += other.Output
	u.CacheWrite += other.CacheWrite
	u.CacheWrite1h += other.CacheWrite1h
	u.CacheRead += other.CacheRead
	u.Reasoning += other.Reasoning
}

// Response is the aggregated result of one inference call.
type Response struct {
	Model string `json:"model,omitempty"`
	// ID is the provider's own identifier for this response, when it exposes
	// one. It is what a support ticket or a provider-side trace is keyed by.
	ID string `json:"id,omitempty"`
	// Err is the failure that ended the turn, when one did. A response with
	// Err set still carries whatever arrived before the failure — the text
	// already streamed, the tokens already billed — because those were real
	// and a caller that discards them loses both the work and the accounting.
	Err error `json:"-"`

	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	StopReason StopReason `json:"stop_reason,omitempty"`
	Usage      Usage      `json:"usage"`

	Thinking          string          `json:"thinking,omitempty"`
	ThinkingSignature string          `json:"thinking_signature,omitempty"`
	Reasoning         []ReasoningItem `json:"reasoning,omitempty"`
}

// Failed reports whether the turn ended on an error or an abort.
func (r *Response) Failed() bool {
	return r.StopReason == StopError || r.StopReason == StopAborted
}

// Message converts the response into the assistant turn to append to history.
// Appending this — rather than a hand-built message — is what carries the
// thinking signature and reasoning items forward, which reasoning models
// require on the following request.
func (r *Response) Message() Message {
	return Message{
		Role:              RoleAssistant,
		Content:           Text(r.Content),
		Thinking:          r.Thinking,
		ThinkingSignature: r.ThinkingSignature,
		Reasoning:         r.Reasoning,
		ToolCalls:         r.ToolCalls,
	}
}
