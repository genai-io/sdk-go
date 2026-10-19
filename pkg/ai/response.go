package ai

import (
	"cmp"
	"encoding/json"
	"fmt"
	"strings"
)

// StopReason says why the model stopped generating.
type StopReason string

const (
	// StopEndTurn is a complete answer: the model had nothing more to say.
	StopEndTurn StopReason = "end_turn"
	// StopToolUse means the turn is waiting on you to run the calls it made.
	StopToolUse StopReason = "tool_use"
	// StopMaxTokens is a truncated answer, cut off by the output cap. The text
	// is real but unfinished, so acting on it as a whole answer is a mistake.
	StopMaxTokens StopReason = "max_tokens"
	// StopSequence means generation hit one of Request.StopSequences.
	StopSequence StopReason = "stop_sequence"
	// StopRefusal is the model declining, which is an answer rather than a
	// failure: retrying the same prompt gets the same refusal.
	StopRefusal StopReason = "refusal"
	// StopError means the call failed; Response.Err says how.
	StopError StopReason = "error"
	// StopAborted means the caller's context ended the call mid-flight.
	StopAborted StopReason = "aborted"
)

// Usage is the token accounting for one call.
type Usage struct {
	Input      int `json:"input"`
	Output     int `json:"output"`
	CacheWrite int `json:"cache_write,omitempty"`
	CacheRead  int `json:"cache_read,omitempty"`

	// CacheWriteLong is the slice of CacheWrite written to a long-lifetime entry,
	// billed at twice the input rate rather than the cache-write one. It travels
	// separately because the two cannot be told apart afterwards.
	CacheWriteLong int `json:"cache_write_long,omitempty"`

	// Reasoning is the tokens spent thinking, where the provider reports them
	// apart from the answer. Cost prices Output, not this, so it is diagnostic.
	Reasoning int `json:"reasoning,omitempty"`
}

// TotalInput is the whole prompt: fresh tokens plus the cached prefix, whether
// that prefix was written or read. It is what a context window is compared
// against, and what a tiered rate card switches on.
func (u Usage) TotalInput() int { return u.Input + u.CacheWrite + u.CacheRead }

// Total is the whole call, prompt and generation together.
func (u Usage) Total() int { return u.TotalInput() + u.Output }

// Add accumulates another call's usage.
func (u *Usage) Add(other Usage) {
	u.Input += other.Input
	u.Output += other.Output
	u.CacheWrite += other.CacheWrite
	u.CacheWriteLong += other.CacheWriteLong
	u.CacheRead += other.CacheRead
	u.Reasoning += other.Reasoning
}

// SplitPromptTokens separates a combined prompt count into its fresh and
// cached halves.
func SplitPromptTokens(promptTokens, cachedTokens int) (fresh, cached int) {
	promptTokens = max(promptTokens, 0)
	cached = min(max(cachedTokens, 0), promptTokens)
	return promptTokens - cached, cached
}

// mergeUsage applies an update field by field, ignoring zeros. Protocols
// report usage in pieces — Anthropic sends input tokens at message_start and
// output tokens at message_delta — so a whole-struct replace would erase the
// half that arrived first.
func mergeUsage(dst *Usage, src Usage) {
	dst.Input = cmp.Or(src.Input, dst.Input)
	dst.Output = cmp.Or(src.Output, dst.Output)
	dst.CacheWrite = cmp.Or(src.CacheWrite, dst.CacheWrite)
	dst.CacheWriteLong = cmp.Or(src.CacheWriteLong, dst.CacheWriteLong)
	dst.CacheRead = cmp.Or(src.CacheRead, dst.CacheRead)
	dst.Reasoning = cmp.Or(src.Reasoning, dst.Reasoning)
}

// Response is the aggregated result of one inference call. Content is the
// canonical provider order; convenience accessors project individual kinds.
type Response struct {
	Model string `json:"model,omitempty"`
	ID    string `json:"id,omitempty"`
	Err   error  `json:"-"`

	Content    Content    `json:"content,omitempty"`
	StopReason StopReason `json:"stop_reason,omitempty"`
	Usage      Usage      `json:"usage"`
}

// Text returns answer text in block order.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	return r.Content.Text()
}

// Thinking returns human-readable reasoning text in block order.
func (r *Response) Thinking() string {
	if r == nil {
		return ""
	}
	return r.Content.Thinking()
}

// ToolCalls returns tool calls in block order.
func (r *Response) ToolCalls() []ToolCall {
	if r == nil {
		return nil
	}
	return r.Content.ToolCalls()
}

// Message converts the response into the assistant turn to append to history,
// preserving every block and its original order.
func (r *Response) Message() Message {
	if r == nil {
		return Message{Role: RoleAssistant}
	}
	return Message{Role: RoleAssistant, Content: r.Content.Clone()}
}

// Decoding an answer into a Go value.

// Unmarshal decodes the answer into v.
func (r *Response) Unmarshal(v any) error {
	if r == nil {
		return fmt.Errorf("ai: no response to decode")
	}
	text := r.Text()
	raw, ok := extractJSON(text)
	if !ok {
		return fmt.Errorf("ai: response contains no JSON value: %s", truncate(text, 200))
	}
	if err := json.Unmarshal([]byte(raw), v); err != nil {
		return fmt.Errorf("ai: decoding response: %w", err)
	}
	return nil
}

// extractJSON finds the JSON value in a model's answer.
func extractJSON(s string) (string, bool) {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return "", false
	}
	if json.Valid([]byte(trimmed)) {
		return trimmed, true
	}
	if fenced, ok := stripFence(trimmed); ok && json.Valid([]byte(fenced)) {
		return fenced, true
	}
	// The first object or array in the prose, if it decodes.
	if start := strings.IndexAny(trimmed, "{["); start >= 0 {
		var raw json.RawMessage
		if json.NewDecoder(strings.NewReader(trimmed[start:])).Decode(&raw) == nil {
			return string(raw), true
		}
	}
	return "", false
}

// stripFence unwraps a ```json … ``` block.
func stripFence(s string) (string, bool) {
	if !strings.HasPrefix(s, "```") {
		return "", false
	}
	rest := s[3:]
	if i := strings.IndexByte(rest, '\n'); i >= 0 {
		rest = rest[i+1:] // drop the language tag
	}
	end := strings.LastIndex(rest, "```")
	if end < 0 {
		return "", false
	}
	return strings.TrimSpace(rest[:end]), true
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
