package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// Role identifies who produced a message. There is no system role: the system
// prompt is a field on Request, which is where every supported protocol puts it.
type Role string

const (
	// RoleUser is you: what you asked, and the results of any tools the model
	// asked you to run. Tool results are a user turn on every protocol here,
	// because they are something handed *to* the model.
	RoleUser Role = "user"
	// RoleAssistant is the model: its answer, its thinking, and the tool calls
	// it wants you to make.
	RoleAssistant Role = "assistant"
)

// BlockType identifies the semantic kind of one content block.
type BlockType string

const (
	// BlockText is ordinary prose, in Block.Text. Either role.
	BlockText BlockType = "text"

	// BlockImage is an inline picture, in Block.Image. User turns only.
	BlockImage BlockType = "image"

	// BlockThinking is reasoning you are allowed to read, in Block.Text, with
	// its provider signature in Block.Signature. Assistant turns only.
	BlockThinking BlockType = "thinking"

	// BlockToolCall is the model asking you to run something, in
	// Block.ToolCall. Assistant turns only.
	BlockToolCall BlockType = "tool_call"

	// BlockToolResult is your answer to one call, in Block.ToolResult. User
	// turns only, and ToolResultsMessage is how you build that turn.
	BlockToolResult BlockType = "tool_result"

	// BlockReasoning is reasoning state you cannot read, in Block.Reasoning.
	// Assistant turns only: the OpenAI Responses protocol carries it, as does
	// Anthropic's redacted thinking.
	BlockReasoning BlockType = "reasoning"
)

// Block is the primitive carried by messages, responses and stream events.
//
//	Type              carried in         produced by
//	BlockText         Text               either
//	BlockImage        Image              you
//	BlockThinking     Text, Signature    the model
//	BlockToolCall     ToolCall           the model
//	BlockToolResult   ToolResult         you
//	BlockReasoning    Reasoning          the model
type Block struct {
	// Type says which field below carries this block's payload.
	Type BlockType `json:"type"`

	// Text carries BlockText and BlockThinking. Signature is the provider
	// token that comes with a thinking block and must be replayed with it.
	Text      string `json:"text,omitempty"`
	Signature string `json:"signature,omitempty"`

	Image      *Image         `json:"image,omitempty"`
	ToolCall   *ToolCall      `json:"tool_call,omitempty"`
	ToolResult *ToolResult    `json:"tool_result,omitempty"`
	Reasoning  *ReasoningItem `json:"reasoning,omitempty"`
}

// Content is one turn's blocks, in the order they were produced.
type Content []Block

// TextBlock returns an answer or user-text block.
func TextBlock(text string) Block { return Block{Type: BlockText, Text: text} }

// ImageBlock returns a user-turn block carrying one inline picture.
func ImageBlock(image Image) Block { return Block{Type: BlockImage, Image: &image} }

// ThinkingBlock returns human-readable reasoning and its optional opaque
// provider signature.
func ThinkingBlock(text, signature string) Block {
	return Block{Type: BlockThinking, Text: text, Signature: signature}
}

// ToolCallBlock returns a block requesting one tool invocation.
func ToolCallBlock(call ToolCall) Block { return Block{Type: BlockToolCall, ToolCall: &call} }

// ToolResultBlock returns a block answering one tool invocation.
func ToolResultBlock(result ToolResult) Block {
	return Block{Type: BlockToolResult, ToolResult: &result}
}

// ReasoningBlock returns an opaque provider reasoning item that must be
// replayed on a later request.
func ReasoningBlock(item ReasoningItem) Block {
	return Block{Type: BlockReasoning, Reasoning: &item}
}

// TextContent returns content holding one text block. Empty text yields nil.
func TextContent(text string) Content {
	if text == "" {
		return nil
	}
	return Content{TextBlock(text)}
}

// Text concatenates answer/user text blocks, ignoring all other kinds.
func (c Content) Text() string { return c.textOf(BlockText) }

// Thinking concatenates human-readable thinking blocks.
func (c Content) Thinking() string { return c.textOf(BlockThinking) }

// ThinkingSignature is the provider token that proves this turn's thinking, or
// empty. It rides with a thinking block rather than being its text, which is
// why Thinking cannot return it: an application replaying a turn has to send
// it back attached to the block it belongs to, or the provider rejects the
// whole turn.
func (c Content) ThinkingSignature() string {
	for _, block := range c {
		if block.Type == BlockThinking && block.Signature != "" {
			return block.Signature
		}
	}
	return ""
}

func (c Content) textOf(kind BlockType) string {
	var out strings.Builder
	for _, block := range c {
		if block.Type == kind {
			out.WriteString(block.Text)
		}
	}
	return out.String()
}

// ToolCalls returns tool-call blocks in order.
func (c Content) ToolCalls() []ToolCall {
	var out []ToolCall
	for _, block := range c {
		if block.Type == BlockToolCall && block.ToolCall != nil {
			call := *block.ToolCall
			call.Signature = slices.Clone(call.Signature)
			out = append(out, call)
		}
	}
	return out
}

// ToolResults returns tool-result blocks in order.
func (c Content) ToolResults() []ToolResult {
	var out []ToolResult
	for _, block := range c {
		if block.Type == BlockToolResult && block.ToolResult != nil {
			out = append(out, *block.ToolResult)
		}
	}
	return out
}

// ReasoningItems returns opaque reasoning blocks in order.
func (c Content) ReasoningItems() []ReasoningItem {
	var out []ReasoningItem
	for _, block := range c {
		if block.Type == BlockReasoning && block.Reasoning != nil {
			out = append(out, *block.Reasoning)
		}
	}
	return out
}

// Has reports whether content contains a populated block of kind.
func (c Content) Has(kind BlockType) bool {
	for _, block := range c {
		if block.Type == kind && !block.empty() {
			return true
		}
	}
	return false
}

// HasImages reports whether any block is an image.
func (c Content) HasImages() bool { return c.Has(BlockImage) }

func (b Block) empty() bool {
	switch b.Type {
	case BlockText, BlockThinking:
		return strings.TrimSpace(b.Text) == "" && b.Signature == ""
	case BlockImage:
		return b.Image == nil
	case BlockToolCall:
		return b.ToolCall == nil
	case BlockToolResult:
		return b.ToolResult == nil
	case BlockReasoning:
		return b.Reasoning == nil
	default:
		return true
	}
}

// Clone returns a content snapshot whose mutable payloads do not alias c.
func (c Content) Clone() Content {
	out := slices.Clone(c)
	for i := range out {
		out[i] = cloneBlock(out[i])
	}
	return out
}

func cloneBlock(block Block) Block {
	if block.Image != nil {
		v := *block.Image
		block.Image = &v
	}
	if block.ToolCall != nil {
		v := *block.ToolCall
		v.Signature = slices.Clone(v.Signature)
		block.ToolCall = &v
	}
	if block.ToolResult != nil {
		v := *block.ToolResult
		v.Content = v.Content.Clone()
		block.ToolResult = &v
	}
	if block.Reasoning != nil {
		v := *block.Reasoning
		block.Reasoning = &v
	}
	return block
}

// Image is a picture sent inline with a message.
type Image struct {
	// MediaType is the IANA type, e.g. "image/png" or "image/jpeg".
	MediaType string `json:"media_type"`
	// Data is the image, base64-encoded, with no "data:" URI prefix — each
	// driver adds whatever framing its own protocol wants.
	Data string `json:"data"`
	// FileName is optional and shown to the model where a protocol carries it.
	FileName string `json:"file_name,omitempty"`
}

// ToolCall is the model asking you to run one of the tools you offered.
type ToolCall struct {
	// ID is the provider's identifier for this call. Echo it back on the
	// ToolResult so the model knows which answer belongs to which call.
	ID string `json:"id"`
	// Name is the tool being called, matching a Tool in the prompt.
	Name string `json:"name"`
	// Input is the arguments as raw JSON, exactly as the model produced them
	// rather than decoded and re-encoded, so replaying the call is byte-exact.
	// Check it with the tool's Schema.Validate, then decode with UnmarshalArgs.
	Input string `json:"input"`

	// Signature is an opaque provider token that must be echoed back with the
	// call on the following turn. Gemini uses it for thought signatures.
	Signature []byte `json:"signature,omitempty"`
}

// UnmarshalArgs decodes the call's arguments into a Go value. A field the
// model did not send keeps whatever the target already holds, and a field the
// schema does not have is an error rather than a silent drop — a model that
// invents an argument should be told, not obeyed halfway.
func (c ToolCall) UnmarshalArgs(into any) error {
	trimmed := bytes.TrimSpace([]byte(c.Input))
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		return fmt.Errorf("arguments for %s: %w", c.Name, err)
	}
	return nil
}

// ToolResult is what running one tool produced.
type ToolResult struct {
	// ToolCallID is the ID of the ToolCall this answers. It is what pairs the
	// two; without it the next request is rejected.
	ToolCallID string `json:"tool_call_id"`
	// ToolName is optional context for protocols that carry it.
	ToolName string `json:"tool_name,omitempty"`
	// Content is what the tool returned: text, and images where the protocol
	// carries them — a screenshot, a rendered chart. TextContent says the
	// text-only answer most tools give; Model.Validate refuses an image on the
	// two protocols that would drop it.
	Content Content `json:"content"`
	// IsError marks a tool that failed. Say so rather than failing the turn: a
	// model shown an error can correct its arguments and try again.
	IsError bool `json:"is_error,omitempty"`
}

// Text is the readable part of what the tool returned, which is what a log, a
// transcript and a session record want.
func (r ToolResult) Text() string { return r.Content.Text() }

// ReasoningItem is reasoning state you cannot read and must carry forward.
type ReasoningItem struct {
	// ID is the provider's handle for this piece of reasoning.
	ID string `json:"id"`
	// EncryptedContent is the model's own working. Replay it untouched and a
	// reasoning model resumes; omit it and it starts over every turn.
	EncryptedContent string `json:"encrypted_content,omitempty"`
	// Summary is a readable précis where the provider offers one.
	Summary string `json:"summary,omitempty"`
}

// Message is one turn of the conversation: who spoke, and what they produced.
type Message struct {
	// ID names this turn, for an application that has to point at one later:
	// the row a person is editing, the entry a store already holds, the turn a
	// fork branches at. It is never sent — no protocol has a field for it — so
	// nothing about it reaches the model and nothing about it costs a token.
	//
	// Empty is the default, because this package makes one call and has no
	// conversation to name anything within. agent.WithMessageIDs is what fills
	// it in, and a session stores and restores whatever it finds here.
	ID string `json:"id,omitempty"`

	Role    Role    `json:"role"`
	Content Content `json:"content,omitempty"`
}

// UserMessage returns a user turn: text, then any images.
func UserMessage(text string, images ...Image) Message {
	content := TextContent(text)
	for _, image := range images {
		content = append(content, ImageBlock(image))
	}
	return Message{Role: RoleUser, Content: content}
}

// AssistantMessage returns an assistant turn carrying text.
func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Content: TextContent(text)}
}

// ToolResultsMessage returns the user turn that answers a set of tool calls.
func ToolResultsMessage(results ...ToolResult) Message {
	content := make(Content, 0, len(results))
	for _, result := range results {
		content = append(content, ToolResultBlock(result))
	}
	return Message{Role: RoleUser, Content: content}
}

// Text returns the message's answer/user text.
func (m Message) Text() string { return m.Content.Text() }

// ToolCalls returns the message's calls in block order.
func (m Message) ToolCalls() []ToolCall { return m.Content.ToolCalls() }

// ToolResults returns the message's results in block order.
func (m Message) ToolResults() []ToolResult { return m.Content.ToolResults() }

// HasToolResults reports whether this turn carries any tool result.
func (m Message) HasToolResults() bool { return m.Content.Has(BlockToolResult) }
