package llm

import (
	"strings"
	"unicode/utf8"
)

// IsOverflow reports whether a turn failed because the prompt exceeded the
// model's context window.
//
// Most providers say so in an error, which Classify already files as
// KindContextExceeded. Two do not, and both are silent in a way that looks
// like a normal answer:
//
//   - Some endpoints accept an oversized prompt and answer anyway, truncating
//     internally. The only signal is that the prompt they billed for is larger
//     than the window they advertise.
//   - Others truncate the input to fill the window exactly, leaving no room to
//     generate, and return a length stop with zero output tokens.
//
// A caller that only checks the error keeps resending a prompt that will never
// fit. Pass the model so the structural cases can be seen; with a model whose
// window is unknown only the error case is detected, which is the honest
// limit.
func IsOverflow(resp *Response, model Model) bool {
	if resp == nil {
		return false
	}
	if resp.Err != nil && IsContextExceeded(resp.Err) {
		return true
	}
	window := model.ContextWindow
	if window <= 0 {
		return false
	}
	prompt := resp.Usage.TotalInput()
	if prompt == 0 {
		return false
	}
	// Accepted silently: billed for more prompt than the window holds.
	if prompt > window {
		return true
	}
	// Truncated to fit, then no room left to answer.
	return resp.StopReason == StopMaxTokens && resp.Usage.Output == 0 && prompt >= window
}

// sanitizeText replaces invalid UTF-8 with the replacement character.
//
// The case that matters is a lone UTF-16 surrogate: a conversation that passed
// through a JavaScript runtime, or a session file written by one, can carry
// half a surrogate pair encoded as three bytes that are not valid UTF-8. Go
// tolerates it in a string, some providers reject the request outright, and
// others return mojibake — so it is cleaned on the way out rather than left to
// fail differently on each endpoint.
//
// Text that is already valid is returned unchanged and unallocated.
func sanitizeText(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	return strings.ToValidUTF8(s, "�")
}

// PrepareMessages is what a driver runs over history before converting it.
//
// It enforces tool-call pairing and cleans invalid UTF-8 — the two corrections
// every protocol needs and none of them should each implement. Drivers call
// this rather than sanitizeToolMessages directly.
func PrepareMessages(msgs []Message) []Message {
	out := sanitizeToolMessages(msgs)
	for i := range out {
		out[i] = sanitizeMessage(out[i])
	}
	return out
}

// sanitizeMessage cleans every text-bearing field of one message, copying only
// when something actually changes.
func sanitizeMessage(m Message) Message {
	if content, changed := sanitizeContent(m.Content); changed {
		m.Content = content
	}
	if clean := sanitizeText(m.Thinking); clean != m.Thinking {
		m.Thinking = clean
	}
	for i, r := range m.ToolResults {
		if clean := sanitizeText(r.Content); clean != r.Content {
			m.ToolResults = append([]ToolResult(nil), m.ToolResults...)
			m.ToolResults[i].Content = clean
		}
	}
	for i, tc := range m.ToolCalls {
		if clean := sanitizeText(tc.Input); clean != tc.Input {
			m.ToolCalls = append([]ToolCall(nil), m.ToolCalls...)
			m.ToolCalls[i].Input = clean
		}
	}
	return m
}

func sanitizeContent(c Content) (Content, bool) {
	var out Content
	for i, part := range c {
		if part.Type != PartText {
			continue
		}
		clean := sanitizeText(part.Text)
		if clean == part.Text {
			continue
		}
		if out == nil {
			out = append(Content(nil), c...)
		}
		out[i].Text = clean
	}
	return out, out != nil
}
