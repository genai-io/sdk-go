package llm

import "strings"

// Role identifies who produced a message. There is no system role: the system
// prompt is a field on Request, which is where every supported protocol puts
// it.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// PartType distinguishes the kinds of content a message can carry.
type PartType string

const (
	PartText  PartType = "text"
	PartImage PartType = "image"
)

// Part is one segment of a message's content.
type Part struct {
	Type  PartType `json:"type"`
	Text  string   `json:"text,omitempty"`
	Image *Image   `json:"image,omitempty"`
}

// Content is an ordered sequence of parts. Order is meaningful — it is how a
// caller places an image between two runs of text, which every protocol here
// supports and which a separate Images field could not express.
type Content []Part

// TextPart returns a text part.
func TextPart(text string) Part { return Part{Type: PartText, Text: text} }

// ImagePart returns an image part.
func ImagePart(img Image) Part { return Part{Type: PartImage, Image: &img} }

// Text returns Content holding a single text part. Empty text yields nil
// Content so an empty message stays empty rather than carrying a blank part.
func Text(text string) Content {
	if text == "" {
		return nil
	}
	return Content{TextPart(text)}
}

// String concatenates the text parts, ignoring images.
func (c Content) String() string {
	switch len(c) {
	case 0:
		return ""
	case 1:
		if c[0].Type == PartText {
			return c[0].Text
		}
	}
	var sb strings.Builder
	for _, p := range c {
		if p.Type == PartText {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// Images returns the image parts in order.
func (c Content) Images() []Image {
	var out []Image
	for _, p := range c {
		if p.Type == PartImage && p.Image != nil {
			out = append(out, *p.Image)
		}
	}
	return out
}

// HasImages reports whether any part is an image.
func (c Content) HasImages() bool {
	for _, p := range c {
		if p.Type == PartImage && p.Image != nil {
			return true
		}
	}
	return false
}

// IsEmpty reports whether the content would send nothing: no images and no
// non-blank text.
func (c Content) IsEmpty() bool {
	for _, p := range c {
		switch p.Type {
		case PartImage:
			if p.Image != nil {
				return false
			}
		case PartText:
			if strings.TrimSpace(p.Text) != "" {
				return false
			}
		}
	}
	return true
}

// Image is an inline image attachment. Data is base64-encoded, without a data:
// URI prefix — drivers add whatever framing their protocol wants.
type Image struct {
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	FileName  string `json:"file_name,omitempty"`
}

// ToolCall is a model's request to run a tool. Input is the raw JSON arguments
// string as the model produced it, not a decoded map: it is echoed back
// verbatim on the next turn, and re-encoding a decoded map would not round-trip
// byte for byte.
type ToolCall struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input string `json:"input"`

	// Signature is an opaque provider token that must be echoed back with the
	// call on the following turn. Google Gemini uses it for thought
	// signatures; other protocols leave it empty.
	Signature []byte `json:"signature,omitempty"`
}

// ToolResult is the outcome of running one tool call.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name,omitempty"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

// ReasoningItem is an opaque reasoning block a model emits and expects back on
// the next request. OpenAI's stateless (store=false) backend requires a
// reasoning model's function call to be preceded by its reasoning item, and
// EncryptedContent is what lets the model restore that state without the
// server holding it.
type ReasoningItem struct {
	ID               string `json:"id"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
	Summary          string `json:"summary,omitempty"`
}

// Message is one turn of the conversation.
//
// A turn carrying tool results is a RoleUser message with ToolResults set —
// that is the shape every protocol here expects, tool results ride inside a
// user turn. Distinguish it from a typed user turn by len(ToolResults) > 0,
// never by Role.
type Message struct {
	Role    Role    `json:"role"`
	Content Content `json:"content,omitempty"`

	// Thinking is the human-readable reasoning text. ThinkingSignature is
	// Anthropic's opaque token that lets one thinking block be replayed
	// verbatim; Reasoning carries OpenAI's encrypted reasoning items. A
	// message uses at most one of the latter two.
	Thinking          string          `json:"thinking,omitempty"`
	ThinkingSignature string          `json:"thinking_signature,omitempty"`
	Reasoning         []ReasoningItem `json:"reasoning,omitempty"`

	ToolCalls   []ToolCall   `json:"tool_calls,omitempty"`
	ToolResults []ToolResult `json:"tool_results,omitempty"`
}

// User returns a user message with the given text and, optionally, images
// appended after it. Use UserContent when the images belong between runs of
// text rather than at the end.
func User(text string, images ...Image) Message {
	content := Text(text)
	for _, img := range images {
		content = append(content, ImagePart(img))
	}
	return Message{Role: RoleUser, Content: content}
}

// UserContent returns a user message with explicitly ordered content.
func UserContent(c Content) Message {
	return Message{Role: RoleUser, Content: c}
}

// Assistant returns an assistant message with the given text.
func Assistant(text string) Message {
	return Message{Role: RoleAssistant, Content: Text(text)}
}

// ToolResultsMessage returns the user turn that answers a set of tool calls.
func ToolResultsMessage(results ...ToolResult) Message {
	return Message{Role: RoleUser, ToolResults: results}
}

// Text returns the message's text content.
func (m Message) Text() string { return m.Content.String() }

// IsToolResult reports whether this turn carries tool results.
func (m Message) IsToolResult() bool { return len(m.ToolResults) > 0 }
