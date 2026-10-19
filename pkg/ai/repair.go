package ai

import (
	"slices"
	"strings"
)

// Repair returns a conversation every protocol will accept: tool calls and
// their results paired, results with no call dropped, invalid UTF-8 replaced,
// and any turn that would put nothing on the wire dropped with them — several
// OpenAI-compatible endpoints reject such a turn rather than ignoring it.
func Repair(msgs []Message) []Message {
	out := repairToolPairing(msgs)
	for i := range out {
		if content := sanitizeContent(out[i].Content); content != nil {
			out[i].Content = content
		}
	}
	return out
}

// repairToolPairing enforces tool-call/tool-result pairing.
func repairToolPairing(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))

	for i := 0; i < len(msgs); i++ {
		msg := msgs[i]

		// Tool results are only ever emitted alongside the assistant message
		// they answer, below. One reaching here answers no call, so only it goes.
		if msg.Content.Has(BlockToolResult) {
			msg.Content = filterBlocks(msg.Content, func(block Block) bool {
				return block.Type != BlockToolResult
			})
			if !isEmptyMessage(msg) {
				out = append(out, msg)
			}
			continue
		}

		if msg.Role != RoleAssistant || !msg.Content.Has(BlockToolCall) {
			if !isEmptyMessage(msg) {
				out = append(out, msg)
			}
			continue
		}

		// Collect the results that directly follow.
		// The IDs are read straight off the blocks: ToolCalls and ToolResults
		// would each build a fresh slice and deep-copy every signature, and
		// all that is wanted here is the identifiers.
		answered := make(map[string]bool)
		j := i + 1
		for j < len(msgs) && msgs[j].Content.Has(BlockToolResult) {
			for _, block := range msgs[j].Content {
				if block.Type == BlockToolResult && block.ToolResult != nil {
					answered[block.ToolResult.ToolCallID] = true
				}
			}
			j++
		}

		keptIDs := make(map[string]bool)
		for _, block := range msg.Content {
			if block.Type == BlockToolCall && block.ToolCall != nil && answered[block.ToolCall.ID] {
				keptIDs[block.ToolCall.ID] = true
			}
		}
		msg.Content = filterBlocks(msg.Content, func(block Block) bool {
			return block.Type != BlockToolCall || block.ToolCall != nil && keptIDs[block.ToolCall.ID]
		})
		if !isEmptyMessage(msg) {
			out = append(out, msg)
		}

		// Re-emit the following result messages, keeping only results whose
		// call survived.
		for k := i + 1; k < j; k++ {
			res := msgs[k]
			res.Content = filterBlocks(res.Content, func(block Block) bool {
				return block.Type != BlockToolResult || block.ToolResult != nil && keptIDs[block.ToolResult.ToolCallID]
			})
			if !isEmptyMessage(res) {
				out = append(out, res)
			}
		}
		i = j - 1
	}

	return out
}

// isEmptyMessage reports whether a message would send nothing at all. Several
// OpenAI-compatible endpoints reject such a message outright rather than
// ignoring it.
func isEmptyMessage(m Message) bool {
	for _, block := range m.Content {
		switch block.Type {
		case BlockText, BlockImage, BlockToolCall, BlockToolResult:
			if !block.empty() {
				return false
			}
		}
	}
	return true
}

// filterBlocks drops the blocks keep rejects, returning content untouched when
// it rejects none — which is every message in a conversation that needs no
// repair, i.e. almost all of them.
func filterBlocks(content Content, keep func(Block) bool) Content {
	for i, block := range content {
		if keep(block) {
			continue
		}
		out := make(Content, i, len(content)-1)
		copy(out, content[:i])
		for _, rest := range content[i+1:] {
			if keep(rest) {
				out = append(out, rest)
			}
		}
		return out
	}
	return content
}

// sanitizeContent returns c with its text made valid, or nil if nothing had to
// change.
func sanitizeContent(c Content) Content {
	var out Content
	for i, block := range c {
		clean, changed := sanitizeBlock(block)
		if !changed {
			continue
		}
		if out == nil {
			out = c.Clone()
		}
		out[i] = clean
	}
	return out
}

// sanitizeBlock returns the block with its text made valid, and whether
// anything had to change.
func sanitizeBlock(block Block) (Block, bool) {
	switch block.Type {
	case BlockText, BlockThinking:
		text := sanitizeText(block.Text)
		if text == block.Text {
			return block, false
		}
		// Block is a value and Text is a string: writing to the local copy
		// cannot reach the caller's block, so no clone is needed here.
		block.Text = text
		return block, true
	case BlockToolCall:
		if block.ToolCall == nil {
			return block, false
		}
		input := sanitizeText(block.ToolCall.Input)
		if input == block.ToolCall.Input {
			return block, false
		}
		clean := cloneBlock(block)
		clean.ToolCall.Input = input
		return clean, true
	case BlockToolResult:
		if block.ToolResult == nil {
			return block, false
		}
		content := sanitizeContent(block.ToolResult.Content)
		name := sanitizeText(block.ToolResult.ToolName)
		if content == nil && name == block.ToolResult.ToolName {
			return block, false
		}
		clean := cloneBlock(block)
		if content != nil {
			clean.ToolResult.Content = content
		}
		clean.ToolResult.ToolName = name
		return clean, true
	case BlockReasoning:
		if block.Reasoning == nil {
			return block, false
		}
		// EncryptedContent is provider state replayed byte for byte, so it is
		// not ours to rewrite; the summary is prose and travels as prose.
		summary := sanitizeText(block.Reasoning.Summary)
		if summary == block.Reasoning.Summary {
			return block, false
		}
		clean := cloneBlock(block)
		clean.Reasoning.Summary = summary
		return clean, true
	}
	return block, false
}

// sanitizeText replaces invalid UTF-8 with the replacement character.
func sanitizeText(s string) string { return strings.ToValidUTF8(s, "�") }

// replayable returns a block of another model's reasoning state as this model
// can take it back, and whether it can at all.
//
// Each protocol wants its own and tolerates no other's. Anthropic replays a
// thinking block only with the signature proving the text was not edited. A
// Chat Completions endpoint carries the text as reasoning_content, which only
// some accept and none accept a signature with — so the signature comes off
// and the reasoning still travels, because it is the thinking that is worth
// replaying and the signature is only Anthropic's proof of it. Gemini takes
// the text as a thought part on the same terms. The Responses protocol replays
// opaque items instead, so readable thinking there is display-only.
//
// A protocol this package does not know is left alone: nothing here knows
// better than the driver that declared it.
func (m Model) replayable(block Block) (Block, bool) {
	switch block.Type {
	case BlockReasoning:
		if m.API == "" {
			return block, true
		}
		if block.Reasoning == nil {
			return block, false
		}
		r := block.Reasoning
		switch r.API {
		case APIAnthropicMessages, APIAnthropicVertex:
			return block, m.API.anthropicFamily()
		case "":
			// Older Anthropic histories have only encrypted content, no item ID.
			if m.API.anthropicFamily() {
				return block, r.ID == "" && r.Summary == "" && r.EncryptedContent != ""
			}
			return block, m.API == APIOpenAIResponses
		case APIOpenAIResponses:
			return block, m.API == APIOpenAIResponses
		default:
			return block, false
		}
	case BlockThinking:
		switch {
		case m.API == "":
			return block, true
		case m.API.anthropicFamily():
			return block, block.Signature != ""
		case m.API == APIOpenAIChat:
			block.Signature = ""
			return block, CompatOf[OpenAIChatCompat](m).ReasoningContent
		case m.API.googleFamily():
			block.Signature = ""
			return block, true
		default:
			// APIOpenAIResponses, and any protocol that carries reasoning as
			// items rather than as text.
			return block, false
		}
	}
	return block, true
}

// strip drops the reasoning state this model cannot replay, which is what
// makes a conversation portable between models. It is the counterpart of
// validate: state the model produced is dropped or trimmed to what this one
// takes, and content the caller wrote is refused.
//
// A message is rebuilt only when it loses something, so a conversation that
// stays with one model is not copied on every call.
func (m Model) strip(msgs []Message) []Message {
	for i, msg := range msgs {
		if !m.rewrites(msg.Content) {
			continue
		}
		out := slices.Clone(msgs)
		for j := i; j < len(out); j++ {
			content := make(Content, 0, len(out[j].Content))
			for _, block := range out[j].Content {
				if kept, ok := m.replayable(block); ok {
					content = append(content, kept)
				}
			}
			out[j].Content = content
		}
		return out
	}
	return msgs
}

// rewrites reports whether this content would come back different, so a
// conversation staying with the model that made it is not copied per call.
func (m Model) rewrites(content Content) bool {
	for _, block := range content {
		kept, ok := m.replayable(block)
		if !ok || kept != block {
			return true
		}
	}
	return false
}
