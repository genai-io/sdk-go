package ai

import (
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
//
// Content is deliberately a tagged sequence instead of a Message with parallel
// text, image, thinking and tool slices. The sequence preserves the order the
// model produced and must later receive back.
type BlockType string

const (
	// BlockText is ordinary prose, in Block.Text. Either role.
	//
	// It is what you wrote and what the model answered; Response.Text joins
	// every one of them in order.
	BlockText BlockType = "text"

	// BlockImage is an inline picture, in Block.Image. User turns only.
	//
	// Send one to a model whose Input lists ModalityImage; sending one to a
	// text-only model is refused before the request leaves, rather than being
	// dropped so the model answers about something it never saw.
	BlockImage BlockType = "image"

	// BlockThinking is reasoning you are allowed to read, in Block.Text, with
	// its provider signature in Block.Signature. Assistant turns only.
	//
	// Show it or hide it as you like — but replay it unchanged in the history
	// of the next turn. Anthropic rejects a thinking block whose signature is
	// missing, because the signature is what proves the text was not edited.
	BlockThinking BlockType = "thinking"

	// BlockToolCall is the model asking you to run something, in
	// Block.ToolCall. Assistant turns only.
	//
	// Every call must be answered by a BlockToolResult in the turn that
	// follows, or the next request is rejected — see RepairHistory.
	BlockToolCall BlockType = "tool_call"

	// BlockToolResult is your answer to one call, in Block.ToolResult. User
	// turns only, and ToolResultsMessage is how you build that turn.
	BlockToolResult BlockType = "tool_result"

	// BlockReasoning is reasoning state you cannot read, in Block.Reasoning.
	// Assistant turns only, and only on the OpenAI Responses protocol.
	//
	// It is the model's own working, encrypted. Carry it forward untouched and
	// a reasoning model resumes where it left off; drop it and the model
	// re-reasons from scratch on every turn.
	BlockReasoning BlockType = "reasoning"
)

// Block is the primitive carried by messages, responses and stream events.
//
// It is a tagged union: Type says which one field below is meaningful, and the
// rest must be zero. A block carrying a payload that does not match its Type is
// refused before the request leaves — quietly sending the wrong one would mean
// the model answered about something you did not send.
//
//	Type              carried in         produced by
//	BlockText         Text               either
//	BlockImage        Image              you
//	BlockThinking     Text, Signature    the model
//	BlockToolCall     ToolCall           the model
//	BlockToolResult   ToolResult         you
//	BlockReasoning    Reasoning          the model
//
// Use the constructors — TextBlock, ImageBlock and the rest — rather than
// filling this in by hand.
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
//
// The order is the point. A model that thought, then called a tool, then
// explained itself produced three blocks in that sequence, and the next
// request has to carry them back the same way. Parallel fields — a Text here,
// a ToolCalls slice there — would lose it, and no protocol accepts a
// conversation whose order was reconstructed by guesswork.
type Content []Block

// TextBlock returns an answer or user-text block.
func TextBlock(text string) Block { return Block{Type: BlockText, Text: text} }

// ImageBlock returns an inline-image block.
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

func (c Content) textOf(kind BlockType) string {
	var out strings.Builder
	for _, block := range c {
		if block.Type == kind {
			out.WriteString(block.Text)
		}
	}
	return out.String()
}

// Images returns image blocks in order.
func (c Content) Images() []Image {
	var out []Image
	for _, block := range c {
		if block.Type == BlockImage && block.Image != nil {
			out = append(out, *block.Image)
		}
	}
	return out
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

// IsEmpty reports whether the content would send nothing.
func (c Content) IsEmpty() bool {
	for _, block := range c {
		if !block.empty() {
			return false
		}
	}
	return true
}

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
	// Check it with Tool.ValidateArgs, then decode with UnmarshalArgs.
	Input string `json:"input"`

	// Signature is an opaque provider token that must be echoed back with the
	// call on the following turn. Gemini uses it for thought signatures.
	Signature []byte `json:"signature,omitempty"`
}

// ToolResult is what running one tool produced.
type ToolResult struct {
	// ToolCallID is the ID of the ToolCall this answers. It is what pairs the
	// two; without it the next request is rejected.
	ToolCallID string `json:"tool_call_id"`
	// ToolName is optional context for protocols that carry it.
	ToolName string `json:"tool_name,omitempty"`
	// Content is what the tool returned, as text.
	Content string `json:"content"`
	// IsError marks a tool that failed. Say so rather than failing the turn: a
	// model shown an error can correct its arguments and try again.
	IsError bool `json:"is_error,omitempty"`
}

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
//
// Build one with UserMessage, AssistantMessage or ToolResultsMessage rather
// than by hand, and append Response.Message to your history to carry a model's
// turn — including its thinking and reasoning state — into the next request.
type Message struct {
	Role    Role    `json:"role"`
	Content Content `json:"content,omitempty"`
}

// UserMessage returns a user turn: text, then any images.
//
// For content in an order this does not produce — text, image, more text —
// write the message out: Message{Role: RoleUser, Content: Content{…}}.
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
