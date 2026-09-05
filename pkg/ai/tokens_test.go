package ai

import (
	"strings"
	"testing"
)

// Measured against a real tokenizer, because an estimate is only worth what it
// is right about. Each count below is len(o200k_base.encode(text)) — the
// tokenizer the GPT-4o and GPT-5 families use.
//
// The rule is one-sided. Under-counting is how a conversation is judged to fit,
// is not compacted, and overflows the window on the call after that; going over
// costs a little context and nothing else. So every sample must estimate at or
// above the truth, and not wildly above it.
//
// The band sits deliberately on the high side of o200k, because o200k is the
// most token-efficient of the tokenizers this SDK talks to: Anthropic's needs
// noticeably more tokens for the same English, so an estimate landing exactly
// on o200k would land under Anthropic's.
func TestTheEstimateIsAboveARealTokenizerAndNotFarAbove(t *testing.T) {
	for _, tc := range []struct {
		name  string
		text  string
		exact int // len(o200k_base.encode(text))
	}{
		{
			name:  "english prose",
			text:  "The loop asks at a step boundary and again when a provider calls the prompt too long, and the person asks through the inbox. All three collapse to the same one message, announced the same way, so a consumer that draws one draws all of them.",
			exact: 51,
		},
		{
			name:  "a tool schema",
			text:  `{"name": "Read", "description": "Reads a file from the local filesystem. file_path must be an absolute path.", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "description": "The absolute path to the file to read"}, "offset": {"type": "integer", "description": "The line number to start reading from"}, "limit": {"type": "integer", "description": "The number of lines to read"}}, "required": ["file_path"]}}`,
			exact: 112,
		},
		{
			// The shape the ratio this replaced got most wrong: machine-written
			// JSON, almost no prose, which it read at 71% of its real size.
			name:  "a JSON tool result",
			text:  jsonToolResult,
			exact: 494,
		},
		{
			name:  "code, as a tool returns it",
			text:  "func (r *Recorder) write(ctx context.Context, turn int, e Entry) {\n\te.Turn = r.turnsBefore + turn\n\te.At = time.Now().UTC()\n\tif err := r.store.Append(ctx, r.id, e); err != nil {\n\t\tr.mu.Lock()\n\t\tif r.err == nil {\n\t\t\tr.err = err\n\t\t}\n\t\tr.mu.Unlock()\n\t}\n}",
			exact: 83,
		},
		{
			name:  "chinese",
			text:  "压缩是用它写出来的六行。这个包每个扩展点都是一个位置,以用途命名的扩展点会是异类。应用踩的不是缺功能,是踩错了缝。",
			exact: 49,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimateTokens(tc.text)
			if got < tc.exact {
				t.Errorf("EstimateTokens = %d, under the %d a real tokenizer counts; "+
					"an estimate below the truth is how a prompt overflows the window", got, tc.exact)
			}
			if over := float64(got)/float64(tc.exact) - 1; over > 0.45 {
				t.Errorf("EstimateTokens = %d against %d real tokens (+%.0f%%); "+
					"over-counting this far compacts a conversation that still fits", got, tc.exact, over*100)
			}
		})
	}
}

// A prompt is more than its messages. A dozen schemas can outweigh the
// conversation, and a request sized without them reports a window as half
// empty when it is nearly full.
func TestAPromptIsSizedWithItsToolsAndSystemPrompt(t *testing.T) {
	msgs := []Message{UserMessage("hello")}
	bare := (&Request{Messages: msgs}).EstimateTokens()

	withSystem := (&Request{System: strings.Repeat("you are a helpful assistant. ", 20), Messages: msgs}).EstimateTokens()
	if withSystem <= bare {
		t.Error("the system prompt was not counted")
	}

	withTools := (&Request{Messages: msgs, Tools: []Tool{{Schema: Schema{
		Name:        "read",
		Description: "Read a file from disk",
		Definition:  map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
	}}}}).EstimateTokens()
	if withTools <= bare {
		t.Error("the tool definitions were not counted")
	}
}

func TestSizingNothingCostsNothing(t *testing.T) {
	if got := EstimateTokens(""); got != 0 {
		t.Errorf("EstimateTokens(\"\") = %d, want 0", got)
	}
	var nilReq *Request
	if got := nilReq.EstimateTokens(); got != 0 {
		t.Errorf("(*Request)(nil).EstimateTokens() = %d, want 0", got)
	}
}

// An image a tool returned costs what an image costs anywhere else. Counting
// it as nothing is how a prompt measured as small arrives over the window.
func TestAnImageInsideAToolResultIsCounted(t *testing.T) {
	withText := (&Request{Messages: []Message{{
		Role:    RoleUser,
		Content: Content{ToolResultBlock(ToolResult{ToolCallID: "c1", Content: TextContent("ok")})},
	}}}).EstimateTokens()

	withImage := (&Request{Messages: []Message{{
		Role: RoleUser,
		Content: Content{ToolResultBlock(ToolResult{ToolCallID: "c1", Content: Content{
			ImageBlock(Image{Data: "not a decodable image", MediaType: "image/png"}),
		}})},
	}}}).EstimateTokens()

	if withImage <= withText {
		t.Errorf("a tool result carrying an image estimated at %d, no more than the %d for one carrying text",
			withImage, withText)
	}
}
