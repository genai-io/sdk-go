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
		// Reasoning state is not a validation subject any more: what a model
		// cannot replay is dropped, not refused. TestVertexStripsWhatMessages
		// StripsToo holds the parity these cases used to.
		"an unsigned thinking block is not an error on Messages": {
			api: APIAnthropicMessages, messages: thinkingTurn("hmm", ""),
		},
		"an unsigned thinking block is not an error on Vertex": {
			api: APIAnthropicVertex, messages: thinkingTurn("hmm", ""),
		},
		"a signature on another protocol is not an error either": {
			api: APIOpenAIChat, messages: thinkingTurn("hmm", "sig"),
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

// What a tool returned is content of its own, and two kinds of it mean
// something: what the model reads, and what it looks at. Anything else in
// there is a mistake worth naming rather than a block a driver silently drops.
func TestAToolResultHoldsTextAndImagesAndNothingElse(t *testing.T) {
	answer := func(inner Block) []Message {
		return []Message{{Role: RoleUser, Content: Content{
			ToolResultBlock(ToolResult{ToolCallID: "c1", Content: Content{inner}}),
		}}}
	}

	c := drive(script{deltas: []Delta{{Block: TextBlock("never")}}})
	for _, ok := range []Block{
		TextBlock("what it says"),
		ImageBlock(Image{MediaType: "image/png", Data: "AAAA"}),
	} {
		if _, err := c.Complete(t.Context(), answer(ok)); err != nil {
			t.Errorf("a tool result holding a %s block was refused: %v", ok.Type, err)
		}
	}

	for _, bad := range []Block{
		ToolCallBlock(ToolCall{ID: "c2", Name: "again"}),
		ThinkingBlock("out loud", "sig"),
		ToolResultBlock(ToolResult{ToolCallID: "c1", Content: TextContent("nested")}),
	} {
		_, err := c.Complete(t.Context(), answer(bad))
		if err == nil {
			t.Errorf("a tool result holding a %s block was accepted", bad.Type)
			continue
		}
		if !strings.Contains(err.Error(), "tool result") {
			t.Errorf("err = %q, want it to say where the block was", err)
		}
	}
}

// Vertex carries the Anthropic Messages protocol, so it must keep and drop
// exactly what Messages does. A rule written for one and not the other is how
// thinking replay silently stopped working on Vertex once before.
func TestVertexStripsWhatMessagesStripsToo(t *testing.T) {
	for _, tc := range []struct {
		name string
		msgs []Message
		kept bool
	}{
		{"a signed block replays", thinkingTurn("hmm", "sig"), true},
		{"an unsigned one cannot be proved and goes", thinkingTurn("hmm", ""), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := Model{ID: "m", API: APIAnthropicMessages}.strip(tc.msgs)
			vertex := Model{ID: "m", API: APIAnthropicVertex}.strip(tc.msgs)

			if got := hasThinking(messages); got != tc.kept {
				t.Errorf("Messages kept thinking = %v, want %v", got, tc.kept)
			}
			if got := hasThinking(vertex); got != tc.kept {
				t.Errorf("Vertex kept thinking = %v, want %v", got, tc.kept)
			}
		})
	}
}

func hasThinking(msgs []Message) bool {
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Type == BlockThinking {
				return true
			}
		}
	}
	return false
}

// A conversation that loses nothing is not copied: strip runs on every call,
// and the common case is a conversation staying with the model that made it.
func TestStrippingNothingKeepsTheSameSlice(t *testing.T) {
	msgs := thinkingTurn("hmm", "sig")
	out := Model{ID: "m", API: APIAnthropicMessages}.strip(msgs)
	if &out[0] != &msgs[0] {
		t.Error("a conversation with nothing to drop was copied anyway")
	}
}

// What the caller wrote is still refused, which is the other half of the line:
// a picture a tool returned is content the model would answer about wrongly if
// it never arrived, not state it can do without.
func TestCallerContentIsStillRefused(t *testing.T) {
	msgs := []Message{{Role: RoleUser, Content: Content{ToolResultBlock(ToolResult{
		ToolCallID: "c1",
		Content:    Content{TextBlock("here"), ImageBlock(Image{MediaType: "image/png", Data: "x"})},
	})}}}

	err := Model{ID: "m", API: APIOpenAIChat}.Validate(msgs)
	if err == nil || !strings.Contains(err.Error(), "only text in a tool result") {
		t.Errorf("err = %v, want the image refused", err)
	}
}

// Every protocol's answer, because this decides silently: a block kept where
// it does not belong is a rejected request, one dropped where it does is a
// model that starts its reasoning over on every turn, and a signature left on
// is a request no endpoint but Anthropic will take.
func TestWhichProtocolReplaysWhichReasoningState(t *testing.T) {
	signed := ThinkingBlock("hmm", "sig")
	plain := ThinkingBlock("hmm", "")
	opaque := ReasoningBlock(ReasoningItem{ID: "r1", EncryptedContent: "x"})

	chatWithReasoning := Model{ID: "m", API: APIOpenAIChat,
		Compat: OpenAIChatCompat{ReasoningContent: true}}

	for _, tc := range []struct {
		name    string
		model   Model
		block   Block
		want    bool
		wantSig string
	}{
		{"Anthropic replays what it signed", Model{API: APIAnthropicMessages}, signed, true, "sig"},
		{"Anthropic cannot prove unsigned thinking", Model{API: APIAnthropicMessages}, plain, false, ""},
		{"Vertex answers as Messages does", Model{API: APIAnthropicVertex}, signed, true, "sig"},
		{"Chat takes it only where the endpoint says so", chatWithReasoning, plain, true, ""},
		{"Chat without reasoning_content takes none", Model{API: APIOpenAIChat}, plain, false, ""},
		// The signature is Anthropic's proof, not the reasoning. It comes off
		// and the thinking still travels, because that is the part worth
		// replaying and this endpoint can read it.
		{"Chat keeps foreign thinking without its signature", chatWithReasoning, signed, true, ""},
		{"Gemini takes the text as a thought", Model{API: APIGoogleGenAI}, plain, true, ""},
		{"Gemini too drops only the signature", Model{API: APIGoogleGenAI}, signed, true, ""},
		{"Responses replays items, not readable thinking", Model{API: APIOpenAIResponses}, plain, false, ""},
		{"opaque items belong to Responses", Model{API: APIOpenAIResponses}, opaque, true, ""},
		{"and nowhere else", Model{API: APIAnthropicMessages}, opaque, false, ""},
		{"a protocol this package does not know is left alone", Model{API: "custom"}, signed, false, ""},
		{"a model that states no protocol keeps everything", Model{}, opaque, true, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tc.model.replayable(tc.block)
			if ok != tc.want {
				t.Fatalf("replayable = %v, want %v", ok, tc.want)
			}
			if ok && got.Signature != tc.wantSig {
				t.Errorf("signature = %q, want %q", got.Signature, tc.wantSig)
			}
			if ok && got.Text != tc.block.Text {
				t.Errorf("text = %q, want it unchanged (%q)", got.Text, tc.block.Text)
			}
		})
	}
}

// A conversation that came from Anthropic and is now being sent to a Chat
// endpoint keeps the reasoning and loses only the proof, which is the case
// that made this rewrite rather than a filter.
func TestAForeignSignatureComesOffAndTheThinkingStays(t *testing.T) {
	history := []Message{
		UserMessage("think"),
		{Role: RoleAssistant, Content: Content{
			ThinkingBlock("step one", "sig-from-anthropic"),
			TextBlock("the answer"),
		}},
	}
	model := Model{ID: "m", API: APIOpenAIChat, Compat: OpenAIChatCompat{ReasoningContent: true}}

	out := model.strip(history)
	blocks := out[1].Content
	if len(blocks) != 2 || blocks[0].Type != BlockThinking {
		t.Fatalf("content came out as %d blocks, want the thinking kept", len(blocks))
	}
	if blocks[0].Text != "step one" || blocks[0].Signature != "" {
		t.Errorf("thinking = %q with signature %q, want the text kept and the proof gone",
			blocks[0].Text, blocks[0].Signature)
	}
	// And the caller's history is not edited under it.
	if history[1].Content[0].Signature != "sig-from-anthropic" {
		t.Error("strip wrote through to the caller's messages")
	}
}

// The signature rides with the thinking rather than being its text, so Thinking
// cannot return it and an application replaying a turn would otherwise have to
// walk the blocks itself.
func TestThinkingSignatureIsReadableWithoutWalkingTheBlocks(t *testing.T) {
	c := Content{TextBlock("before"), ThinkingBlock("weighing it", "sig-1"), TextBlock("after")}
	if got := c.ThinkingSignature(); got != "sig-1" {
		t.Errorf("ThinkingSignature = %q, want sig-1", got)
	}
	// Thinking a provider signs nothing still reads as thinking.
	unsigned := Content{ThinkingBlock("weighing it", "")}
	if got := unsigned.ThinkingSignature(); got != "" {
		t.Errorf("ThinkingSignature = %q, want empty for unsigned thinking", got)
	}
	plain := Content{TextBlock("no thinking here")}
	if got := plain.ThinkingSignature(); got != "" {
		t.Errorf("ThinkingSignature = %q, want empty", got)
	}
}
