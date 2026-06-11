// Package llm provides a public SDK for calling LLMs through a unified
// Provider abstraction. It defines the canonical types that providers implement
// and consumers depend on — no dependency on any specific backend.
package llm

import "context"

// Role identifies who produced a message in the conversation.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool_result"
)

// Message is the unit exchanged between consumer and LLM.
type Message struct {
	Role              Role         `json:"role"`
	Content           string       `json:"content,omitempty"`
	Images            []Image      `json:"images,omitempty"`
	Thinking          string       `json:"thinking,omitempty"`
	ThinkingSignature string       `json:"thinking_signature,omitempty"`
	ToolCalls         []ToolCall   `json:"tool_calls,omitempty"`
	ToolResult        *ToolResult  `json:"tool_result,omitempty"`
	Meta              map[string]any `json:"meta,omitempty"`
}

// Image represents an image attachment.
type Image struct {
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	FileName  string `json:"file_name"`
	Size      int    `json:"size"`
}

// ToolCall represents a tool call from the model.
type ToolCall struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input string `json:"input"`
}

// ToolResult is the outcome of a tool execution.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name,omitempty"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

// ToolSchema is a typed tool definition sent to the LLM.
type ToolSchema struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"input_schema,omitempty"`
}

// UserMessage creates a user message with optional images.
func UserMessage(text string, images []Image) Message {
	return Message{
		Role:    RoleUser,
		Content: text,
		Images:  images,
	}
}

// AssistantMessage creates an assistant message.
func AssistantMessage(text, thinking string, calls []ToolCall) Message {
	return Message{
		Role:      RoleAssistant,
		Content:   text,
		Thinking:  thinking,
		ToolCalls: calls,
	}
}

// StopReason describes why the LLM stopped generating.
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopMaxTokens StopReason = "max_tokens"
	StopToolUse   StopReason = "tool_use"
	StopMaxSteps  StopReason = "max_steps"
	StopCancelled StopReason = "cancelled"
)

// InferRequest is sent to the LLM for inference.
type InferRequest struct {
	System   string       // assembled system prompt
	Messages []Message    // conversation history
	Tools    []ToolSchema // available tools
}

// InferResponse is the final aggregated response from one LLM call.
type InferResponse struct {
	Content           string     // text output
	Thinking          string     // chain-of-thought
	ThinkingSignature string     // signature for replaying thinking blocks
	ToolCalls         []ToolCall // tool execution requests
	StopReason        StopReason
	TokensIn          int
	TokensOut         int
	CacheCreateTokens int
	CacheReadTokens   int
}

// Chunk is one piece of a streaming LLM response.
type Chunk struct {
	Text     string // incremental text
	Thinking string // incremental thinking
	Done     bool   // true on final chunk

	Response *InferResponse // non-nil only when Done=true
	Err      error          // non-nil on stream error
}

// LLM is the inference abstraction — call a language model.
//
// Infer sends a request and returns a channel of streaming chunks.
// The channel is closed when the response is complete or on error.
// The final chunk has Done=true and carries the aggregated InferResponse.
type LLM interface {
	Infer(ctx context.Context, req InferRequest) (<-chan Chunk, error)
	InputLimit() int
}

// CompletionOptions contains options for a completion request.
type CompletionOptions struct {
	Model          string
	Messages       []Message
	MaxTokens      int
	Temperature    float64
	Tools          []ToolSchema
	SystemPrompt   string
	ThinkingEffort string
}

// ModelInfo represents information about an available model.
type ModelInfo struct {
	ID               string `json:"id"`
	Name             string `json:"name"`
	DisplayName      string `json:"displayName,omitempty"`
	InputTokenLimit  int    `json:"inputTokenLimit,omitempty"`
	OutputTokenLimit int    `json:"outputTokenLimit,omitempty"`
}

// Provider is the interface all LLM backends implement.
type Provider interface {
	Stream(ctx context.Context, opts CompletionOptions) <-chan StreamChunk
	ListModels(ctx context.Context) ([]ModelInfo, error)
	Name() string
}

// ChunkType represents the type of a stream chunk from a provider.
type ChunkType string

const (
	ChunkTypeText      ChunkType = "text"
	ChunkTypeThinking  ChunkType = "thinking"
	ChunkTypeToolStart ChunkType = "tool_start"
	ChunkTypeToolInput ChunkType = "tool_input"
	ChunkTypeDone      ChunkType = "done"
	ChunkTypeError     ChunkType = "error"
)

// StreamChunk is a chunk in a streaming response from a provider.
type StreamChunk struct {
	Type     ChunkType
	Text     string            // for text/thinking chunks
	ToolID   string            // for tool_start chunks
	ToolName string            // for tool_start chunks
	Response *CompletionResponse // for done chunks
	Error    error             // for error chunks
}

// CompletionResponse is the aggregated response from a provider.
type CompletionResponse struct {
	Content           string     `json:"content,omitempty"`
	Thinking          string     `json:"thinking,omitempty"`
	ThinkingSignature string     `json:"thinking_signature,omitempty"`
	ToolCalls         []ToolCall `json:"tool_calls,omitempty"`
	StopReason        string     `json:"stop_reason"`
	Usage             Usage      `json:"usage"`
}

// Usage contains token usage information.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}
