package ai

import (
	"strings"
	"unicode/utf8"
)

// Repair returns a conversation every protocol will accept: tool calls and
// their results paired, results with no call dropped, invalid UTF-8 replaced,
// and any turn that would put nothing on the wire dropped with them — several
// OpenAI-compatible endpoints reject such a turn rather than ignoring it.
func Repair(msgs []Message) []Message {
	out := repairToolPairing(msgs)
	for i := range out {
		out[i] = repairEncoding(out[i])
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
		if msg.HasToolResults() {
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
		for j < len(msgs) && msgs[j].HasToolResults() {
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

// repairEncoding cleans every text-bearing field of one message, copying only
// when something actually changes.
func repairEncoding(m Message) Message {
	if content, changed := sanitizeContent(m.Content); changed {
		m.Content = content
	}
	return m
}

func sanitizeContent(c Content) (Content, bool) {
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
	return out, out != nil
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
		content, edited := sanitizeContent(block.ToolResult.Content)
		name := sanitizeText(block.ToolResult.ToolName)
		if !edited && name == block.ToolResult.ToolName {
			return block, false
		}
		clean := cloneBlock(block)
		if edited {
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
func sanitizeText(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	return strings.ToValidUTF8(s, "�")
}
