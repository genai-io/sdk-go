package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/llmtest"
)

func TestClientAggregatesDeltas(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{
		{Thinking: "let me "},
		{Thinking: "think"},
		{ThinkingSignature: "sig-a"},
		{ThinkingSignature: "sig-b"},
		{Text: "Hello, "},
		{Text: "world"},
		{Usage: &llm.Usage{Input: 12}},
		{Usage: &llm.Usage{Output: 3, CacheRead: 100, Reasoning: 2}},
		{StopReason: llm.StopEndTurn},
	}}}}

	resp, err := llmtest.Client(drv).Complete(context.Background(), &llm.Prompt{}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "Hello, world" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello, world")
	}
	if resp.Thinking != "let me think" {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
	// Signature fragments append; usage fields merge across deltas rather than
	// replacing wholesale.
	if resp.ThinkingSignature != "sig-asig-b" {
		t.Errorf("ThinkingSignature = %q", resp.ThinkingSignature)
	}
	if want := (llm.Usage{Input: 12, Output: 3, CacheRead: 100, Reasoning: 2}); resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
	if resp.Usage.TotalInput() != 112 {
		t.Errorf("TotalInput = %d, want 112", resp.Usage.TotalInput())
	}
}

func TestClientStreamEventOrder(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{
		{Thinking: "hmm"},
		{Text: "a"},
		{ToolCall: &llm.ToolCall{ID: "1", Name: "ls", Input: "{}"}},
	}}}}

	var types []llm.EventType
	var done *llm.Response
	for event, err := range llmtest.Client(drv).Stream(context.Background(), &llm.Prompt{}, nil) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		types = append(types, event.Type)
		if event.Type == llm.EventDone {
			done = event.Response
		}
	}

	want := []llm.EventType{
		llm.EventThinkingStart, llm.EventThinkingDelta, llm.EventThinkingEnd,
		llm.EventTextStart, llm.EventTextDelta, llm.EventTextEnd,
		llm.EventToolCall, llm.EventDone,
	}
	if len(types) != len(want) {
		t.Fatalf("events = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("events = %v, want %v", types, want)
		}
	}
	// No driver reported a stop reason, but pending tool calls say what
	// happened.
	if done.StopReason != llm.StopToolUse {
		t.Errorf("StopReason = %q, want %q", done.StopReason, llm.StopToolUse)
	}
	if len(done.ToolCalls) != 1 || done.ToolCalls[0].Name != "ls" {
		t.Errorf("ToolCalls = %+v", done.ToolCalls)
	}
}

func TestClientInfersEndTurn(t *testing.T) {
	resp, err := llmtest.Client(&llmtest.Driver{Turns: []llmtest.Turn{{
		Deltas: []llm.Delta{{Text: "done"}},
	}}}).Complete(context.Background(), &llm.Prompt{}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != llm.StopEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llm.StopEndTurn)
	}
}

func TestClientPropagatesError(t *testing.T) {
	want := errors.New("boom")
	_, err := llmtest.Client(llmtest.Fail(want)).Complete(context.Background(), &llm.Prompt{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
}

func TestNilOptionsMeansDefaults(t *testing.T) {
	drv := llmtest.Text("ok")
	client := llm.New(drv, llmtest.Model)
	if _, err := client.Complete(context.Background(), &llm.Prompt{}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := drv.Last().Options.MaxTokens; got != llmtest.Model.MaxOutput {
		t.Errorf("MaxTokens = %d, want the model's %d", got, llmtest.Model.MaxOutput)
	}
}

func TestClientDefaultsDoNotMutateCallerOptions(t *testing.T) {
	drv := llmtest.Text("ok")
	client := llm.New(drv, llmtest.Model, llm.WithMaxTokens(999), llm.WithTemperature(0.5))

	opts := &llm.Options{}
	if _, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("hi")}}, opts); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if opts.MaxTokens != 0 || opts.Temperature != 0 {
		t.Errorf("the caller's Options were mutated: %+v", opts)
	}
	sent := drv.Last().Options
	if sent.MaxTokens != 999 || sent.Temperature != 0.5 {
		t.Errorf("defaults not applied: MaxTokens=%d Temperature=%v", sent.MaxTokens, sent.Temperature)
	}
}

func TestPerRequestOptionsBeatClientDefaults(t *testing.T) {
	drv := llmtest.Text("ok")
	client := llm.New(drv, llmtest.Model, llm.WithMaxTokens(999))

	if _, err := client.Complete(context.Background(), &llm.Prompt{}, &llm.Options{MaxTokens: 42}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := drv.Last().Options.MaxTokens; got != 42 {
		t.Errorf("MaxTokens = %d, want the per-request 42", got)
	}
}

// The model's sampling parameters sit underneath the caller's, so a caller can
// override one without restating the rest.
func TestSamplingParamsLayer(t *testing.T) {
	model := llmtest.Model
	model.SamplingParams = map[string]any{"top_p": 0.9, "top_k": 40}

	drv := llmtest.Text("ok")
	client := llm.New(drv, model)
	if _, err := client.Complete(context.Background(), &llm.Prompt{},
		&llm.Options{SamplingParams: map[string]any{"top_p": 0.5}}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	sent := drv.Last().Options.SamplingParams
	if sent["top_p"] != 0.5 {
		t.Errorf("top_p = %v, want the caller's 0.5", sent["top_p"])
	}
	if sent["top_k"] != 40 {
		t.Errorf("top_k = %v, want the model's 40", sent["top_k"])
	}
	if model.SamplingParams["top_p"] != 0.9 {
		t.Error("the model's own map was mutated")
	}
}

func TestSplitPromptTokens(t *testing.T) {
	tests := []struct {
		prompt, cached     int
		wantFresh, wantHit int
	}{
		{100, 40, 60, 40},
		{100, 0, 100, 0},
		{100, 100, 0, 100},
		// Malformed wire data must not push fresh below zero.
		{100, 500, 0, 100},
		{100, -5, 100, 0},
		{-1, 10, 0, 0},
	}
	for _, tc := range tests {
		fresh, cached := llm.SplitPromptTokens(tc.prompt, tc.cached)
		if fresh != tc.wantFresh || cached != tc.wantHit {
			t.Errorf("SplitPromptTokens(%d, %d) = (%d, %d), want (%d, %d)",
				tc.prompt, tc.cached, fresh, cached, tc.wantFresh, tc.wantHit)
		}
	}
}

func TestResponseMessageCarriesReasoningForward(t *testing.T) {
	resp := &llm.Response{
		Content:           "answer",
		Thinking:          "reasoning",
		ThinkingSignature: "sig",
		Reasoning:         []llm.ReasoningItem{{ID: "r1", EncryptedContent: "enc"}},
		ToolCalls:         []llm.ToolCall{{ID: "1", Name: "ls"}},
	}
	msg := resp.Message()
	if msg.Role != llm.RoleAssistant || msg.Text() != "answer" {
		t.Fatalf("message = %+v", msg)
	}
	if msg.ThinkingSignature != "sig" || len(msg.Reasoning) != 1 || len(msg.ToolCalls) != 1 {
		t.Errorf("reasoning state was dropped: %+v", msg)
	}
}

func TestOpenWithoutDriverExplainsItself(t *testing.T) {
	_, err := llm.Open(llm.Config{Model: llm.Model{ID: "x", API: "no-such-protocol"}})
	var unreg *llm.UnregisteredAPIError
	if !errors.As(err, &unreg) {
		t.Fatalf("err = %v, want *UnregisteredAPIError", err)
	}
}

// A turn that streamed text and burned tokens before failing must hand both
// back. Returning only an error throws away the partial answer and the
// accounting for the spend that already happened.
func TestFailedTurnCarriesPartialContentAndUsage(t *testing.T) {
	want := errors.New("connection reset")
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{
		Deltas: []llm.Delta{
			{Text: "here is the first ha"},
			{Usage: &llm.Usage{Input: 4_000, Output: 12}},
		},
		Err: want,
	}}}

	resp, err := llmtest.Client(drv).Complete(context.Background(), &llm.Prompt{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	if resp == nil {
		t.Fatal("no response returned alongside the error")
	}
	if resp.Content != "here is the first ha" {
		t.Errorf("Content = %q, want the text that arrived before the failure", resp.Content)
	}
	if resp.Usage.Input != 4_000 || resp.Usage.Output != 12 {
		t.Errorf("Usage = %+v, want the tokens already billed", resp.Usage)
	}
	if resp.StopReason != llm.StopError {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, llm.StopError)
	}
	if !errors.Is(resp.Err, want) {
		t.Error("Response.Err does not carry the failure")
	}
	if !resp.Failed() {
		t.Error("Failed() should be true")
	}
}

// A cancelled turn is not a failure to investigate — it is what the caller
// asked for, so it reports separately from an error.
func TestCancelledTurnReportsAborted(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{{Text: "never"}}}}}
	resp, err := llmtest.Client(drv).Complete(ctx, &llm.Prompt{}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if resp == nil || resp.StopReason != llm.StopAborted {
		t.Errorf("resp = %+v, want StopAborted", resp)
	}
}

// The streaming caller sees the same thing: one yield carries the error and
// the partial response together.
func TestStreamErrorEventCarriesPartial(t *testing.T) {
	want := errors.New("boom")
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{
		Deltas: []llm.Delta{{Text: "partial"}},
		Err:    want,
	}}}

	var sawErr error
	var partial *llm.Response
	for event, err := range llmtest.Client(drv).Stream(context.Background(), &llm.Prompt{}, nil) {
		if err != nil {
			sawErr = err
			partial = event.Response
			break
		}
	}
	if !errors.Is(sawErr, want) {
		t.Fatalf("err = %v", sawErr)
	}
	if partial == nil || partial.Content != "partial" {
		t.Errorf("partial = %+v, want the streamed text", partial)
	}
}

func TestResponseIDIsRecorded(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{
		{ID: "msg_01abc", Model: "served-model"},
		{Text: "hi"},
		{StopReason: llm.StopEndTurn},
	}}}}
	resp, err := llmtest.Client(drv).Complete(context.Background(), &llm.Prompt{}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.ID != "msg_01abc" {
		t.Errorf("ID = %q", resp.ID)
	}
	// A gateway routing to a concrete model reports which one it served.
	if resp.Model != "served-model" {
		t.Errorf("Model = %q, want the model actually served", resp.Model)
	}
}

// A consumer has to be able to tell "the model started a second paragraph"
// from "it kept typing" — the difference between opening a new bubble and
// appending to the last one.
func TestContentBlocksAreBounded(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{
		{Thinking: "let me "},
		{Thinking: "think"},
		{Text: "first "},
		{Text: "block"},
		{EndBlock: true},
		{Text: "second block"},
		{StopReason: llm.StopEndTurn},
	}}}}

	type record struct {
		kind  llm.EventType
		index int
		text  string
	}
	var got []record
	for event, err := range llmtest.Client(drv).Stream(context.Background(), &llm.Prompt{}, nil) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if event.Type != llm.EventDone {
			got = append(got, record{event.Type, event.Index, event.Text})
		}
	}

	want := []record{
		{llm.EventThinkingStart, 0, ""},
		{llm.EventThinkingDelta, 0, "let me "},
		{llm.EventThinkingDelta, 0, "think"},
		{llm.EventThinkingEnd, 0, "let me think"},
		{llm.EventTextStart, 1, ""},
		{llm.EventTextDelta, 1, "first "},
		{llm.EventTextDelta, 1, "block"},
		{llm.EventTextEnd, 1, "first block"},
		// The explicit boundary is what separates two blocks of the same kind.
		{llm.EventTextStart, 2, ""},
		{llm.EventTextDelta, 2, "second block"},
		{llm.EventTextEnd, 2, "second block"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d events, want %d:\n%+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// A tool call closes whatever text was being written, so a consumer never has
// an unterminated block.
func TestToolCallClosesTheOpenBlock(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{
		{Text: "let me look"},
		{ToolCall: &llm.ToolCall{ID: "1", Name: "ls", Input: "{}"}},
		{StopReason: llm.StopToolUse},
	}}}}

	var seen []llm.EventType
	for event, err := range llmtest.Client(drv).Stream(context.Background(), &llm.Prompt{}, nil) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		seen = append(seen, event.Type)
	}
	want := []llm.EventType{
		llm.EventTextStart, llm.EventTextDelta, llm.EventTextEnd,
		llm.EventToolCall, llm.EventDone,
	}
	for i := range want {
		if i >= len(seen) || seen[i] != want[i] {
			t.Fatalf("events = %v, want %v", seen, want)
		}
	}
}

// A turn that dies mid-block still closes it, so a consumer's render state is
// never left half-open.
func TestFailureClosesTheOpenBlock(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{
		Deltas: []llm.Delta{{Text: "half"}},
		Err:    errors.New("boom"),
	}}}

	var closed bool
	for event, err := range llmtest.Client(drv).Stream(context.Background(), &llm.Prompt{}, nil) {
		if event.Type == llm.EventTextEnd {
			closed = true
		}
		if err != nil {
			break
		}
	}
	if !closed {
		t.Error("a block was left open when the turn failed")
	}
}
