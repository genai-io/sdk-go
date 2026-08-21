package ai

import (
	"strings"
	"unicode/utf8"
)

// History repair: making a real conversation acceptable to every protocol.
//
// A conversation that a program actually accumulated is not always one a
// provider will take. Two things go wrong, both outside the caller's control,
// and every protocol here rejects both:
//
//   - A tool call with no answer. The user hits Ctrl-C while the model is
//     emitting a tool call, or a restored session was compacted between the
//     call and its result. The history now ends on an unanswered call, and the
//     next request comes back 400.
//   - Invalid UTF-8. A conversation that passed through a JavaScript runtime,
//     or a session file written by one, can carry half a UTF-16 surrogate pair.
//     Go keeps it in a string without complaint; some providers reject the
//     request and others answer in mojibake.
//
// Repairing both here, once, is what keeps four drivers from each carrying
// their own version of it — and what keeps the failure from being four
// different failures. Client.prepare runs RepairHistory before validation and
// before counting, so exact counting, estimated counting, middleware and
// generation all observe the same conversation that goes on the wire.
//
// This is repair, not policy. It removes only what the protocol itself would
// reject. Deciding that a conversation is too long, summarizing it, dropping
// the oldest turns — those are the application's calls to make, with knowledge
// this package does not have, and none of them happen here.

// RepairHistory returns a conversation every protocol will accept: tool calls
// and their results paired, invalid UTF-8 replaced.
//
// It never writes to msgs or to anything msgs points at — the caller's history
// is theirs, and a session that silently lost a turn to an SDK is a bug nobody
// can find.
func RepairHistory(msgs []Message) []Message {
	out := repairToolPairing(msgs)
	for i := range out {
		out[i] = repairEncoding(out[i])
	}
	return out
}

// repairToolPairing enforces tool-call/tool-result pairing.
//
// The rule is strict adjacency — for an assistant message, only the run of
// tool-result messages directly after it counts. A call with no matching
// result is stripped, a result matching no kept call is dropped, and a message
// left carrying nothing at all is removed.
func repairToolPairing(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))

	for i := 0; i < len(msgs); i++ {
		msg := msgs[i]

		// Tool results are only ever emitted alongside the assistant message
		// they answer, below. One reached here on its own, so it is orphaned.
		if msg.HasToolResults() {
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
			if res.Content.Has(BlockToolResult) {
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
//
// An assistant message is empty unless it carries text or tool calls:
// reasoning content alone does not satisfy Chat Completions validation, which
// DeepSeek rejects with "content or tool_calls must be set".
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
//
// It decides before it allocates. Cloning first and comparing afterwards costs
// a block copy for every block in the conversation, on every request, to
// discover that a conversation which is already valid — nearly all of them —
// needed nothing.
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
		content := sanitizeText(block.ToolResult.Content)
		if content == block.ToolResult.Content {
			return block, false
		}
		clean := cloneBlock(block)
		clean.ToolResult.Content = content
		return clean, true
	}
	return block, false
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
