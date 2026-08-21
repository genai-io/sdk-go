package llm_test

import (
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
)

func assistantCalling(ids ...string) llm.Message {
	calls := make([]llm.ToolCall, len(ids))
	for i, id := range ids {
		calls[i] = llm.ToolCall{ID: id, Name: "tool", Input: "{}"}
	}
	return llm.Message{Role: llm.RoleAssistant, Content: llm.Text("working"), ToolCalls: calls}
}

func resultsFor(ids ...string) llm.Message {
	results := make([]llm.ToolResult, len(ids))
	for i, id := range ids {
		results[i] = llm.ToolResult{ToolCallID: id, Content: "ok"}
	}
	return llm.ToolResultsMessage(results...)
}

func TestSanitizeKeepsPairedCalls(t *testing.T) {
	msgs := []llm.Message{
		llm.User("hi"),
		assistantCalling("a", "b"),
		resultsFor("a", "b"),
		llm.Assistant("done"),
	}
	got := llm.SanitizeToolMessages(msgs)
	if len(got) != 4 {
		t.Fatalf("got %d messages, want 4: %+v", len(got), got)
	}
	if len(got[1].ToolCalls) != 2 || len(got[2].ToolResults) != 2 {
		t.Errorf("paired calls were altered: %+v", got)
	}
}

func TestSanitizeStripsUnansweredCall(t *testing.T) {
	// A mid-stream cancel leaves "b" with no result.
	msgs := []llm.Message{
		assistantCalling("a", "b"),
		resultsFor("a"),
	}
	got := llm.SanitizeToolMessages(msgs)
	if len(got) != 2 {
		t.Fatalf("got %d messages, want 2: %+v", len(got), got)
	}
	if len(got[0].ToolCalls) != 1 || got[0].ToolCalls[0].ID != "a" {
		t.Errorf("unanswered call survived: %+v", got[0].ToolCalls)
	}
}

func TestSanitizeDropsOrphanResults(t *testing.T) {
	// A restored session whose assistant turn was compacted away.
	msgs := []llm.Message{
		llm.User("hi"),
		resultsFor("gone"),
		llm.Assistant("done"),
	}
	got := llm.SanitizeToolMessages(msgs)
	if len(got) != 2 {
		t.Fatalf("got %d messages, want 2: %+v", len(got), got)
	}
	for _, m := range got {
		if m.IsToolResult() {
			t.Errorf("orphan result survived: %+v", m)
		}
	}
}

func TestSanitizeDropsAssistantLeftEmpty(t *testing.T) {
	// The assistant said nothing but "here are some tools", and none was
	// answered — the message would send no content at all.
	msgs := []llm.Message{
		llm.User("hi"),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "a", Name: "tool"}}},
	}
	got := llm.SanitizeToolMessages(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d messages, want 1: %+v", len(got), got)
	}
}

func TestSanitizeIgnoresNonAdjacentResults(t *testing.T) {
	// A result separated from its call by a user turn is not valid pairing on
	// any of the supported protocols.
	msgs := []llm.Message{
		assistantCalling("a"),
		llm.User("actually, never mind"),
		resultsFor("a"),
	}
	got := llm.SanitizeToolMessages(msgs)
	for _, m := range got {
		if len(m.ToolCalls) > 0 {
			t.Errorf("unpaired call survived: %+v", m)
		}
		if m.IsToolResult() {
			t.Errorf("non-adjacent result survived: %+v", m)
		}
	}
}

func TestDropEmptyMessages(t *testing.T) {
	msgs := []llm.Message{
		llm.User("hi"),
		llm.User("   "),
		{Role: llm.RoleAssistant, Thinking: "reasoning only"},
		llm.Assistant("real"),
	}
	got := llm.DropEmptyMessages(msgs)
	if len(got) != 2 {
		t.Fatalf("got %d messages, want 2: %+v", len(got), got)
	}
}

func TestContentHelpers(t *testing.T) {
	img := llm.Image{MediaType: "image/png", Data: "AAAA"}
	c := llm.Content{llm.TextPart("look at "), llm.ImagePart(img), llm.TextPart(" closely")}

	if got := c.String(); got != "look at  closely" {
		t.Errorf("String() = %q", got)
	}
	if !c.HasImages() || len(c.Images()) != 1 {
		t.Errorf("images not reported: %+v", c)
	}
	if c.IsEmpty() {
		t.Error("content with an image reported empty")
	}
	if !(llm.Content{llm.TextPart("  ")}).IsEmpty() {
		t.Error("whitespace-only content should be empty")
	}
	if llm.Text("") != nil {
		t.Error("Text(\"\") should yield nil content")
	}
}
