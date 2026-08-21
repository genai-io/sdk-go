package llm

// sanitizeToolMessages enforces tool-call/tool-result pairing.
//
// Every protocol here rejects a conversation where an assistant turn's tool
// calls are not answered by the turn that follows, or where a result answers
// no call. Interrupted runs leave exactly that behind: a mid-stream cancel
// drops the results for calls already in history, and a restored session can
// carry results whose calls were compacted away.
//
// The rule is strict adjacency — for an assistant message, only the run of
// tool-result messages directly after it counts. A call with no matching
// result is stripped, a result matching no kept call is dropped, and a message
// left carrying nothing at all is removed.
//
// Drivers call this on the way in, so a caller never has to think about it.
func sanitizeToolMessages(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))

	for i := 0; i < len(msgs); i++ {
		msg := msgs[i]

		// Tool results are only ever emitted alongside the assistant message
		// they answer, below. One reached here on its own, so it is orphaned.
		if msg.IsToolResult() {
			continue
		}

		if msg.Role != RoleAssistant || len(msg.ToolCalls) == 0 {
			if !isEmptyMessage(msg) {
				out = append(out, msg)
			}
			continue
		}

		// Collect the results that directly follow.
		answered := make(map[string]bool)
		j := i + 1
		for j < len(msgs) && msgs[j].IsToolResult() {
			for _, r := range msgs[j].ToolResults {
				answered[r.ToolCallID] = true
			}
			j++
		}

		kept := make([]ToolCall, 0, len(msg.ToolCalls))
		keptIDs := make(map[string]bool, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			if answered[tc.ID] {
				kept = append(kept, tc)
				keptIDs[tc.ID] = true
			}
		}
		msg.ToolCalls = kept
		if !isEmptyMessage(msg) {
			out = append(out, msg)
		}

		// Re-emit the following result messages, keeping only results whose
		// call survived.
		for k := i + 1; k < j; k++ {
			results := make([]ToolResult, 0, len(msgs[k].ToolResults))
			for _, r := range msgs[k].ToolResults {
				if keptIDs[r.ToolCallID] {
					results = append(results, r)
				}
			}
			if len(results) > 0 {
				res := msgs[k]
				res.ToolResults = results
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
	if len(m.ToolCalls) > 0 || len(m.ToolResults) > 0 {
		return false
	}
	return m.Content.IsEmpty()
}

// dropEmptyMessages removes messages that would send no content.
func dropEmptyMessages(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		if !isEmptyMessage(m) {
			out = append(out, m)
		}
	}
	return out
}
