package llm_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
)

func TestIsOverflowFromError(t *testing.T) {
	model := llm.Model{ID: "m", ContextWindow: 200_000}
	resp := &llm.Response{
		StopReason: llm.StopError,
		Err:        llm.Classify("test", 400, nil, "", "prompt is too long: 250000 tokens > 200000", nil),
	}
	if !llm.IsOverflow(resp, model) {
		t.Error("an explicit overflow error was not detected")
	}
}

// Some endpoints accept an oversized prompt and answer anyway. The only signal
// is that they billed for more prompt than the window holds.
func TestIsOverflowWhenAcceptedSilently(t *testing.T) {
	model := llm.Model{ID: "m", ContextWindow: 200_000}
	resp := &llm.Response{
		StopReason: llm.StopEndTurn,
		Content:    "a perfectly normal-looking answer",
		Usage:      llm.Usage{Input: 190_000, CacheRead: 20_000, Output: 100},
	}
	if !llm.IsOverflow(resp, model) {
		t.Error("a silently accepted overflow was not detected")
	}
	// A prompt inside the window is not an overflow, however full.
	fits := &llm.Response{StopReason: llm.StopEndTurn, Usage: llm.Usage{Input: 199_000, Output: 100}}
	if llm.IsOverflow(fits, model) {
		t.Error("a prompt inside the window was reported as overflow")
	}
}

// Others truncate the input to fill the window exactly, leaving no room to
// generate, and return a length stop with zero output.
func TestIsOverflowWhenTruncated(t *testing.T) {
	model := llm.Model{ID: "m", ContextWindow: 200_000}
	resp := &llm.Response{
		StopReason: llm.StopMaxTokens,
		Usage:      llm.Usage{Input: 200_000, Output: 0},
	}
	if !llm.IsOverflow(resp, model) {
		t.Error("a truncate-then-no-room overflow was not detected")
	}
	// A length stop that actually produced output is just a long answer.
	produced := &llm.Response{
		StopReason: llm.StopMaxTokens,
		Usage:      llm.Usage{Input: 200_000, Output: 4096},
	}
	if llm.IsOverflow(produced, model) {
		t.Error("a normal length stop was reported as overflow")
	}
}

// With no window there is nothing to compare against, so only the error case
// is detectable. Reporting that honestly beats guessing.
func TestIsOverflowWithUnknownWindow(t *testing.T) {
	model := llm.Model{ID: "m"}
	silent := &llm.Response{StopReason: llm.StopEndTurn, Usage: llm.Usage{Input: 5_000_000}}
	if llm.IsOverflow(silent, model) {
		t.Error("a structural check ran against an unknown window")
	}
	withErr := &llm.Response{Err: llm.Classify("test", 400, nil, "", "maximum context length", nil)}
	if !llm.IsOverflow(withErr, model) {
		t.Error("the error case should still be detected")
	}
}

// A lone UTF-16 surrogate — half a pair, encoded as three bytes that are not
// valid UTF-8 — reaches us from anything that passed through a JavaScript
// runtime. Some providers reject the request outright.
func TestSanitizeText(t *testing.T) {
	lone := string([]byte{0xED, 0xA0, 0x80}) // CESU-8 encoding of U+D800
	dirty := "before" + lone + "after"

	clean := llm.SanitizeText(dirty)
	if !strings.Contains(clean, "before") || !strings.Contains(clean, "after") {
		t.Errorf("surrounding text was lost: %q", clean)
	}
	if strings.Contains(clean, lone) {
		t.Error("the lone surrogate survived")
	}

	// Valid text is returned untouched.
	const ok = "héllo 世界 🌍"
	if got := llm.SanitizeText(ok); got != ok {
		t.Errorf("valid text was altered: %q", got)
	}
}

func TestPrepareMessagesCleansEveryTextField(t *testing.T) {
	lone := string([]byte{0xED, 0xA0, 0x80})
	msgs := []llm.Message{
		llm.User("hi" + lone),
		{
			Role:      llm.RoleAssistant,
			Content:   llm.Text("out" + lone),
			Thinking:  "think" + lone,
			ToolCalls: []llm.ToolCall{{ID: "1", Name: "ls", Input: `{"p":"` + lone + `"}`}},
		},
		llm.ToolResultsMessage(llm.ToolResult{ToolCallID: "1", Content: "res" + lone}),
	}

	got := llm.PrepareMessages(msgs)
	for _, m := range got {
		if !isValidUTF8(m.Content.String()) || !isValidUTF8(m.Thinking) {
			t.Errorf("message text still invalid: %+v", m)
		}
		for _, tc := range m.ToolCalls {
			if !isValidUTF8(tc.Input) {
				t.Errorf("tool input still invalid: %q", tc.Input)
			}
		}
		for _, r := range m.ToolResults {
			if !isValidUTF8(r.Content) {
				t.Errorf("tool result still invalid: %q", r.Content)
			}
		}
	}
	// The caller's slice is not mutated.
	if isValidUTF8(msgs[1].Thinking) {
		t.Error("the caller's messages were modified in place")
	}
}

func isValidUTF8(s string) bool { return llm.SanitizeText(s) == s }

// A long-lifetime cache write is billed at twice the input rate; a cost that
// ignores the split understates a long-cache turn.
func TestLongCacheWriteCostsDouble(t *testing.T) {
	p := llm.Pricing{Currency: llm.USD, Input: 5, Output: 25, CacheWrite: 6.25, CacheRead: 0.50}

	short := p.Cost(llm.Usage{CacheWrite: 1_000_000})
	if short.CacheWrite != 6.25 {
		t.Errorf("short write = %v, want the CacheWrite rate", short.CacheWrite)
	}

	long := p.Cost(llm.Usage{CacheWrite: 1_000_000, CacheWrite1h: 1_000_000})
	if long.CacheWrite != 10 {
		t.Errorf("long write = %v, want twice the input rate", long.CacheWrite)
	}

	// A mixed turn bills each slice at its own rate.
	mixed := p.Cost(llm.Usage{CacheWrite: 1_000_000, CacheWrite1h: 400_000})
	if want := 0.6*6.25 + 0.4*10; !approx(mixed.CacheWrite, want) {
		t.Errorf("mixed write = %v, want %v", mixed.CacheWrite, want)
	}
}

func approx(a, b float64) bool {
	d := a - b
	return d < 1e-9 && d > -1e-9
}

func TestErrorsUnwrapThroughResponse(t *testing.T) {
	want := errors.New("boom")
	resp := &llm.Response{StopReason: llm.StopError, Err: want}
	if !resp.Failed() {
		t.Error("a response with an error should report as failed")
	}
	if !errors.Is(resp.Err, want) {
		t.Error("the underlying error was lost")
	}
}
