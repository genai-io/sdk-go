package ai

import (
	"strings"
	"testing"
)

func thinkingTurn(text, signature string) []Message {
	return []Message{
		UserMessage("hi"),
		{Role: RoleAssistant, Content: Content{ThinkingBlock(text, signature), TextBlock("yes")}},
		UserMessage("go on"),
	}
}

// Anthropic on Vertex is the same body over a different door: a rule naming
// only APIAnthropicMessages breaks thinking replay for every Vertex caller.
func TestVertexIsHeldToTheAnthropicRules(t *testing.T) {
	for name, tc := range map[string]struct {
		api      API
		messages []Message
		opts     []Option
		wantErr  string
	}{
		"a signed thinking block replays on Messages": {
			api: APIAnthropicMessages, messages: thinkingTurn("hmm", "sig"),
		},
		"a signed thinking block replays on Vertex": {
			api: APIAnthropicVertex, messages: thinkingTurn("hmm", "sig"),
		},
		"an unsigned thinking block is refused on Messages": {
			api: APIAnthropicMessages, messages: thinkingTurn("hmm", ""),
			wantErr: "requires its signature",
		},
		"an unsigned thinking block is refused on Vertex": {
			// Without this the driver drops the block on the floor and the
			// model loses its own reasoning with nothing said.
			api: APIAnthropicVertex, messages: thinkingTurn("hmm", ""),
			wantErr: "requires its signature",
		},
		"a signature belongs to no other protocol": {
			api: APIOpenAIChat, messages: thinkingTurn("hmm", "sig"),
			wantErr: "signed thinking blocks belong to the Anthropic Messages protocol",
		},
		"sampling extensions have nowhere to go on Messages": {
			api:      APIAnthropicMessages,
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithSamplingParams(map[string]any{"top_k": 40})},
			wantErr:  "does not support OpenAI sampling parameter extensions",
		},
		"sampling extensions have nowhere to go on Vertex either": {
			// They pass validation and are then never sent, which reads as the
			// parameter having had no effect.
			api:      APIAnthropicVertex,
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithSamplingParams(map[string]any{"top_k": 40})},
			wantErr:  "does not support OpenAI sampling parameter extensions",
		},
		"sampling extensions are fine on the protocol that defines them": {
			api:      APIOpenAIChat,
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithSamplingParams(map[string]any{"top_k": 40})},
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := Model{ID: "m", API: tc.api}.Validate(tc.messages, tc.opts...)
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("Validate: %v, want it accepted", err)
			case tc.wantErr == "":
			case err == nil:
				t.Fatalf("Validate accepted the request; want %q", tc.wantErr)
			case !strings.Contains(err.Error(), tc.wantErr):
				t.Errorf("err = %q\nwant it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// Validation is what makes a request cost nothing when it cannot work, so the
// checks have to fire before the network rather than after the bill.
func TestValidateCatchesWhatTheModelCannotDo(t *testing.T) {
	image := Message{Role: RoleUser, Content: Content{ImageBlock(Image{MediaType: "image/png", Data: "x"})}}
	tool := Tool{Schema: Schema{Name: "search", Description: "look things up"}}

	for name, tc := range map[string]struct {
		model    Model
		messages []Message
		opts     []Option
		wantErr  string
	}{
		"a retired model says what to use instead": {
			model:    Model{ID: "old", Stage: StageRetired, Replacement: "new"},
			messages: []Message{UserMessage("hi")},
			wantErr:  "use new",
		},
		"an image to a text-only model": {
			model:    Model{ID: "m"},
			messages: []Message{image},
			wantErr:  "does not accept image input",
		},
		"an image to a model that takes them": {
			model:    Model{ID: "m", Input: []Modality{ModalityText, ModalityImage}},
			messages: []Message{image},
		},
		"tools to a model without them": {
			model:    Model{ID: "m", Unsupported: Unsupported{Tools: true}},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithTools(tool)},
			wantErr:  "does not support tools",
		},
		"a system prompt to a model with no system role": {
			model:    Model{ID: "m", Unsupported: Unsupported{System: true}},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithSystem("be brief")},
			wantErr:  "has no system role",
		},
		"stop sequences on the Responses protocol": {
			model:    Model{ID: "m", API: APIOpenAIResponses},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithStopSequences("END")},
			wantErr:  "does not support stop sequences",
		},
		"a rung the model's ladder does not declare": {
			model:    Model{ID: "m", Reasoning: []ReasoningLevel{{Effort: EffortLow, Value: "low"}}},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithEffort(Effort("turbo"))},
			wantErr:  "does not offer reasoning effort",
		},
		"a portable rung the model does not list still passes": {
			// ResolveLevel snaps it; refusing here would make the named rungs
			// unportable, which is the whole point of having them.
			model:    Model{ID: "m", Reasoning: []ReasoningLevel{{Effort: EffortLow, Value: "low"}}},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithEffort(EffortHigh)},
		},
		"forcing a tool that is not in the prompt": {
			model:    Model{ID: "m"},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithTools(tool), WithForceTool("other")},
			wantErr:  "is not present in the prompt",
		},
		"two tools with one name": {
			model:    Model{ID: "m"},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithTools(tool, tool)},
			wantErr:  `tool name "search" is duplicated`,
		},
		"a negative token cap": {
			model:    Model{ID: "m"},
			messages: []Message{UserMessage("hi")},
			opts:     []Option{WithMaxTokens(-1)},
			wantErr:  "max tokens cannot be negative",
		},
		"a thinking block in a user turn": {
			model:    Model{ID: "m"},
			messages: []Message{{Role: RoleUser, Content: Content{ThinkingBlock("hmm", "")}}},
			wantErr:  "thinking block belongs to an assistant message",
		},
		"a block claiming to be two things at once": {
			model: Model{ID: "m"},
			messages: []Message{{Role: RoleUser, Content: Content{
				{Type: BlockText, Text: "hi", Image: &Image{MediaType: "image/png", Data: "x"}},
			}}},
			wantErr: "contains a payload for another block type",
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := tc.model.Validate(tc.messages, tc.opts...)
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("Validate: %v, want it accepted", err)
			case tc.wantErr == "":
			case err == nil:
				t.Fatalf("Validate accepted the request; want %q", tc.wantErr)
			case !strings.Contains(err.Error(), tc.wantErr):
				t.Errorf("err = %q\nwant it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// Structure is checked before history repair, because repair would remove the
// malformed block and hide the mistake instead of reporting it.
func TestAMalformedBlockIsReportedRatherThanRepairedAway(t *testing.T) {
	c := drive(script{deltas: []Delta{{Block: TextBlock("never")}}})
	orphan := []Message{{Role: RoleUser, Content: Content{{Type: BlockToolResult}}}}

	_, err := c.Complete(t.Context(), orphan)
	if err == nil {
		t.Fatal("a tool-result block with no payload was accepted")
	}
	if !strings.Contains(err.Error(), "message 0 block 0") {
		t.Errorf("err = %q, want it to say which block", err)
	}
}
