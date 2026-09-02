package ai

import (
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

	// CacheWrite1h is the slice of CacheWrite written to a long-lifetime entry,
	// which is billed at twice the input rate rather than the cache-write one.
	// It travels separately because the totals cannot be told apart afterwards,
	// and pricing a long-cache turn at the short rate understates it.
	CacheWrite1h int `json:"cache_write_1h,omitempty"`

	// Reasoning is the tokens spent thinking, where the provider reports them
	// apart from the answer. Cost prices Output, not this, so it is diagnostic:
	// how much of a turn went on working the caller never saw.
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
	u.CacheWrite1h += other.CacheWrite1h
	u.CacheRead += other.CacheRead
	u.Reasoning += other.Reasoning
}

// mergeUsage applies an update field by field, ignoring zeros. Protocols
// report usage in pieces — Anthropic sends input tokens at message_start and
// output tokens at message_delta — so a whole-struct replace would erase the
// half that arrived first.
func mergeUsage(dst *Usage, src Usage) {
	if src.Input > 0 {
		dst.Input = src.Input
	}
	if src.Output > 0 {
		dst.Output = src.Output
	}
	if src.CacheWrite > 0 {
		dst.CacheWrite = src.CacheWrite
	}
	if src.CacheWrite1h > 0 {
		dst.CacheWrite1h = src.CacheWrite1h
	}
	if src.CacheRead > 0 {
		dst.CacheRead = src.CacheRead
	}
	if src.Reasoning > 0 {
		dst.Reasoning = src.Reasoning
	}
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

// Failed reports whether the turn ended on an error or an abort.
func (r *Response) Failed() bool {
	return r != nil && (r.StopReason == StopError || r.StopReason == StopAborted)
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

// ReasoningItems returns opaque reasoning state in block order.
func (r *Response) ReasoningItems() []ReasoningItem {
	if r == nil {
		return nil
	}
	return r.Content.ReasoningItems()
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
	raw, ok := ExtractJSON(text)
	if !ok {
		return fmt.Errorf("ai: response contains no JSON value: %s", truncate(text, 200))
	}
	if err := json.Unmarshal([]byte(raw), v); err != nil {
		return fmt.Errorf("ai: decoding response: %w", err)
	}
	return nil
}

// Parse decodes a completion into T, threading the call's error through so a
// typed answer is one statement:
//
//	person, err := ai.Parse[Person](client.Complete(ctx, messages))
func Parse[T any](resp *Response, err error) (T, error) {
	var out T
	if err != nil {
		return out, err
	}
	return out, resp.Unmarshal(&out)
}

// ExtractJSON finds the JSON value in a model's answer.
func ExtractJSON(s string) (string, bool) {
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
	if scanned, ok := scanBalanced(trimmed); ok {
		return scanned, true
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

// scanBalanced returns the first balanced object or array, ignoring braces
// that sit inside string literals.
func scanBalanced(s string) (string, bool) {
	start := strings.IndexAny(s, "{[")
	if start < 0 {
		return "", false
	}
	open := s[start]
	close := byte('}')
	if open == '[' {
		close = ']'
	}

	depth, inString, escaped := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case escaped:
			escaped = false
		case c == '\\' && inString:
			escaped = true
		case c == '"':
			inString = !inString
		case inString:
		case c == open:
			depth++
		case c == close:
			depth--
			if depth == 0 {
				candidate := s[start : i+1]
				return candidate, json.Valid([]byte(candidate))
			}
		}
	}
	return "", false
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
