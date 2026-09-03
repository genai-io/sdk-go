package ai

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func userText(text string) Message {
	return Message{Role: RoleUser, Content: Content{TextBlock(text)}}
}

func assistantCalls(ids ...string) Message {
	msg := Message{Role: RoleAssistant}
	for _, id := range ids {
		msg.Content = append(msg.Content, ToolCallBlock(ToolCall{ID: id, Name: "search"}))
	}
	return msg
}

func results(ids ...string) Message {
	out := make([]ToolResult, len(ids))
	for i, id := range ids {
		out[i] = ToolResult{ToolCallID: id, Content: TextContent("done")}
	}
	return ToolResultsMessage(out...)
}

// shape renders a conversation as the thing a protocol actually sees, so a
// failing case reads as what went on the wire rather than as a struct dump.
func shape(msgs []Message) string {
	var sb strings.Builder
	for i, msg := range msgs {
		if i > 0 {
			sb.WriteString(" | ")
		}
		sb.WriteString(string(msg.Role))
		for _, block := range msg.Content {
			sb.WriteString(" " + string(block.Type))
			switch {
			case block.Type == BlockToolCall && block.ToolCall != nil:
				sb.WriteString("(" + block.ToolCall.ID + ")")
			case block.Type == BlockToolResult && block.ToolResult != nil:
				sb.WriteString("(" + block.ToolResult.ToolCallID + ")")
			case block.Text != "":
				sb.WriteString("(" + block.Text + ")")
			}
		}
	}
	return sb.String()
}

// Every protocol rejects a call with no result and a result with no call, and
// most reject a turn that would send nothing. Nothing else may be dropped.
func TestRepairKeepsEverythingItDoesNotHaveToDrop(t *testing.T) {
	for name, tc := range map[string]struct {
		in   []Message
		want string
	}{
		"a paired call and result is left alone": {
			in:   []Message{userText("search"), assistantCalls("c1"), results("c1")},
			want: "user text(search) | assistant tool_call(c1) | user tool_result(c1)",
		},
		"a call nobody answered is dropped": {
			in:   []Message{assistantCalls("c1", "c2"), results("c1")},
			want: "assistant tool_call(c1) | user tool_result(c1)",
		},
		"results spread over several turns all count": {
			in:   []Message{assistantCalls("c1", "c2"), results("c1"), results("c2")},
			want: "assistant tool_call(c1) tool_call(c2) | user tool_result(c1) | user tool_result(c2)",
		},
		"an orphaned result on its own goes": {
			in:   []Message{userText("hello"), results("ghost")},
			want: "user text(hello)",
		},
		"an orphaned result beside text keeps the text": {
			in: []Message{{Role: RoleUser, Content: Content{
				ToolResultBlock(ToolResult{ToolCallID: "ghost", Content: TextContent("done")}),
				TextBlock("and while you are at it"),
			}}},
			want: "user text(and while you are at it)",
		},
		"a result for the wrong call keeps the turn's text": {
			in: []Message{assistantCalls("c1"), {Role: RoleUser, Content: Content{
				ToolResultBlock(ToolResult{ToolCallID: "ghost", Content: TextContent("done")}),
				TextBlock("never mind"),
			}}},
			want: "user text(never mind)",
		},
		"an assistant turn of nothing but thinking is dropped": {
			in: []Message{userText("hi"),
				{Role: RoleAssistant, Content: Content{ThinkingBlock("hmm", "sig")}}},
			want: "user text(hi)",
		},
		"thinking alongside an answer is kept": {
			in: []Message{{Role: RoleAssistant, Content: Content{
				ThinkingBlock("hmm", "sig"), TextBlock("yes"),
			}}},
			want: "assistant thinking(hmm) text(yes)",
		},
		"an empty turn is dropped": {
			in:   []Message{userText("hi"), {Role: RoleAssistant}, userText("still there?")},
			want: "user text(hi) | user text(still there?)",
		},
		"a whitespace-only turn is dropped": {
			in:   []Message{userText("hi"), {Role: RoleAssistant, Content: Content{TextBlock("  \n")}}},
			want: "user text(hi)",
		},
		"two answers to one call are both the caller's to keep": {
			// Repair pairs; it does not choose between two answers to the same
			// call, because either choice would be silent.
			in:   []Message{assistantCalls("c1"), results("c1", "c1")},
			want: "assistant tool_call(c1) | user tool_result(c1) tool_result(c1)",
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := shape(Repair(tc.in)); got != tc.want {
				t.Errorf("Repair produced\n\t%s\nwant\n\t%s", got, tc.want)
			}
		})
	}
}

// Repair copies only what it changes, so a conversation needing none must come
// back as the caller's own slice rather than a duplicate of it.
func TestRepairDoesNotTouchAConversationThatNeedsNothing(t *testing.T) {
	in := []Message{userText("hi"), {Role: RoleAssistant, Content: Content{TextBlock("hello")}}}
	out := Repair(in)
	if len(out) != len(in) {
		t.Fatalf("Repair returned %d messages, want %d", len(out), len(in))
	}
	for i := range out {
		if &out[i].Content[0] != &in[i].Content[0] {
			t.Errorf("message %d was copied; nothing about it needed repair", i)
		}
	}
}

// Invalid UTF-8 is rejected outright by several endpoints, and it can arrive in
// any string a turn carries — not only in the prose.
func TestRepairMakesEveryTextFieldValid(t *testing.T) {
	bad := "oops" + string([]byte{0xff})
	in := []Message{
		{Role: RoleAssistant, Content: Content{
			TextBlock(bad),
			ThinkingBlock(bad, "sig"),
			ToolCallBlock(ToolCall{ID: "c1", Name: "search", Input: `{"q":"` + bad + `"}`}),
			ReasoningBlock(ReasoningItem{ID: "r1", EncryptedContent: "opaque", Summary: bad}),
		}},
		{Role: RoleUser, Content: Content{
			ToolResultBlock(ToolResult{ToolCallID: "c1", ToolName: bad, Content: TextContent(bad)}),
		}},
	}

	out := Repair(in)
	assistant, user := out[0].Content, out[1].Content
	for field, got := range map[string]string{
		"text":              assistant[0].Text,
		"thinking":          assistant[1].Text,
		"tool call input":   assistant[2].ToolCall.Input,
		"reasoning summary": assistant[3].Reasoning.Summary,
		"tool result name":  user[0].ToolResult.ToolName,
		"tool result body":  user[0].ToolResult.Text(),
	} {
		if !utf8.ValidString(got) || !strings.ContainsRune(got, utf8.RuneError) {
			t.Errorf("%s = %q, want the invalid byte replaced", field, got)
		}
	}

	// Opaque provider state is replayed byte for byte and is not ours to
	// rewrite, so it must come back untouched.
	if got := assistant[3].Reasoning.EncryptedContent; got != "opaque" {
		t.Errorf("encrypted reasoning = %q, want it carried through unchanged", got)
	}
	// The caller's own messages must not have been edited under them.
	if in[0].Content[0].Text != bad {
		t.Error("Repair rewrote the caller's message in place")
	}
}

// Repair rewrites the history on the way to every call, so it is the one place
// a name could quietly go missing. An application that keyed its store on
// these would find the conversation it got back pointing at nothing.
func TestRepairKeepsTheNamesItWasGiven(t *testing.T) {
	in := []Message{
		{ID: "a", Role: RoleUser, Content: Content{TextBlock("hi\x00there")}}, // needs sanitising
		{ID: "b", Role: RoleAssistant, Content: Content{ToolCallBlock(ToolCall{ID: "t1", Name: "search"})}},
		{ID: "c", Role: RoleUser, Content: Content{ToolResultBlock(ToolResult{
			ToolCallID: "t1", Content: TextContent("done")})}},
	}

	out := Repair(in)
	if len(out) != len(in) {
		t.Fatalf("Repair returned %d messages, want %d", len(out), len(in))
	}
	for i, want := range []string{"a", "b", "c"} {
		if out[i].ID != want {
			t.Errorf("message %d came back named %q, want %q", i, out[i].ID, want)
		}
	}
}

// A name is never sent, so it cannot be part of what a prompt costs. Counting
// it would make a conversation look larger than the one the provider is
// pricing, and compact a caller early for the whole of a long session.
func TestANameCostsNoTokens(t *testing.T) {
	plain := &Request{Messages: []Message{userText("how large is this?")}}
	named := &Request{Messages: []Message{{
		ID:   "a-name-long-enough-to-show-up-in-any-estimate-at-all",
		Role: RoleUser, Content: Content{TextBlock("how large is this?")},
	}}}

	if a, b := EstimateTokens(plain), EstimateTokens(named); a != b {
		t.Errorf("the named prompt is estimated at %d and the same prompt at %d", b, a)
	}
}
