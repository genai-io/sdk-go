package ai

import (
	"fmt"
	"math"
	"strings"
)

// Request validation: everything caught before the network.

// Validate checks a call against what the model can actually do, before it
// reaches the network. It takes the same arguments as Complete, so a caller
// can ask whether a request is sendable without sending it.
func (m Model) Validate(messages []Message, opts ...Option) error {
	return m.validate(newRequest(m, messages, opts))
}

func (m Model) validate(req *Request) error {
	if !m.Stage.Available() {
		msg := fmt.Sprintf("model %s is retired and no longer serves requests", m)
		if m.Replacement != "" {
			msg += "; use " + m.Replacement
		}
		return &Error{Kind: KindUnsupported, Message: msg}
	}
	if err := m.checkCompat(); err != nil {
		return err
	}
	if req == nil {
		req = &Request{}
	}
	if err := validateSettings(req); err != nil {
		return err
	}
	if err := m.validateStructure(req); err != nil {
		return err
	}
	return m.validateCapabilities(req)
}

// validateCapabilities checks the request against what this model declares it
// cannot do.
func (m Model) validateCapabilities(req *Request) error {
	if !m.Accepts(ModalityImage) {
		for _, msg := range req.Messages {
			if msg.Content.HasImages() {
				return m.unsupported("does not accept image input")
			}
		}
	}
	// A rung is checked against the model, not against a fixed list. The named
	// constants are the portable vocabulary and always pass — ResolveLevel
	// snaps one the model lacks onto a rung it has. A name outside them is
	// allowed too, but only if this model's own ladder declares it: that is
	// what lets a live listing or a hand-built Model carry a rung this package
	// has never heard of, while still catching a typo instead of silently
	// falling back to the default.
	if want := req.Effort; want != EffortDefault && !want.Valid() && !m.Offers(want) {
		if rungs := m.Efforts(); len(rungs) > 0 {
			return invalidRequest("model %s does not offer reasoning effort %q; it offers %v", m, want, rungs)
		}
		return invalidRequest("unknown reasoning effort %q, and model %s declares no reasoning ladder", want, m)
	}
	if m.Unsupported.Tools && len(req.Tools) > 0 {
		return m.unsupported("does not support tools, but %d were provided", len(req.Tools))
	}
	// These two are keyed off the protocol rather than off Unsupported because
	// they are facts about the wire format, not about any one endpoint: the
	// Anthropic and Gemini bodies have nowhere to put an OpenAI sampling
	// extension, and Responses has no stop-sequence parameter. A per-model
	// flag would have to be set on every row and would silently stop firing
	// the first time someone forgot.
	if len(req.SamplingParams) > 0 && (m.API == APIAnthropicMessages || m.API == APIGoogleGenAI) {
		return m.unsupported("does not support OpenAI sampling parameter extensions")
	}
	if len(req.StopSequences) > 0 && m.API == APIOpenAIResponses {
		return m.unsupported("does not support stop sequences on the Responses API")
	}
	if m.Unsupported.ToolChoice && req.ToolChoice != ToolChoiceAuto {
		return m.unsupported("does not support constraining which tool is called")
	}
	if m.Unsupported.System && req.System != "" {
		return m.unsupported("has no system role; put the instructions in the first user message instead")
	}
	if m.Unsupported.Multiturn && len(req.Messages) > 1 {
		return m.unsupported("accepts a single message, but %d were provided", len(req.Messages))
	}
	if req.Schema != nil {
		switch {
		case m.Unsupported.Schema:
			return m.unsupported("cannot constrain output to a schema; " +
				"ask for the shape in the prompt and decode with Response.Unmarshal, which tolerates a fenced or prefaced answer")
		case m.Unsupported.SchemaWithTools && len(req.Tools) > 0:
			return m.unsupported("cannot constrain output and offer tools in the same request")
		}
	}
	return nil
}

// validateStructure checks the tagged-union and protocol invariants before any
// history repair can remove a malformed block. validate calls it again
// afterward so the final request is the one capabilities see.
func (m Model) validateStructure(req *Request) error {
	for messageIndex, message := range req.Messages {
		if message.Role != RoleUser && message.Role != RoleAssistant {
			return invalidBlock(messageIndex, -1, "unknown role %q", message.Role)
		}
		for blockIndex, block := range message.Content {
			if err := validateBlock(message.Role, block); err != nil {
				return invalidBlock(messageIndex, blockIndex, "%v", err)
			}
			if err := m.validateProtocolBlock(block); err != nil {
				return invalidBlock(messageIndex, blockIndex, "%v", err)
			}
		}
	}
	return nil
}

func validateSettings(req *Request) error {
	switch {
	case req.MaxTokens < 0:
		return invalidRequest("max tokens cannot be negative")
	case req.Temperature != nil && (math.IsNaN(*req.Temperature) || math.IsInf(*req.Temperature, 0)):
		return invalidRequest("temperature must be finite")
	case req.Temperature != nil && *req.Temperature < 0:
		return invalidRequest("temperature cannot be negative")
	}
	switch req.CacheRetention {
	case CacheDefault, CacheNone, CacheShort, CacheLong:
	default:
		return invalidRequest("unknown cache retention %q", req.CacheRetention)
	}
	for i, stop := range req.StopSequences {
		if stop == "" {
			return invalidRequest("stop sequence %d is empty", i)
		}
	}

	toolNames := make(map[string]bool, len(req.Tools))
	for i, tool := range req.Tools {
		if strings.TrimSpace(tool.Name) == "" {
			return invalidRequest("tool %d has no name", i)
		}
		if tool.Parameters != nil && tool.ParameterSchema() == nil {
			return invalidRequest("tool %q parameters must be a JSON Schema object", tool.Name)
		}
		if toolNames[tool.Name] {
			return invalidRequest("tool name %q is duplicated", tool.Name)
		}
		toolNames[tool.Name] = true
	}
	if name, forced := req.ToolChoice.Tool(); forced {
		if name == "" {
			return invalidRequest("tool choice names no tool")
		}
		if !toolNames[name] {
			return invalidRequest("forced tool %q is not present in the prompt", name)
		}
	}
	if req.ToolChoice == ToolChoiceRequired && len(req.Tools) == 0 {
		return invalidRequest("required tool choice needs at least one tool")
	}
	if req.Schema != nil && req.Schema.DefinitionMap() == nil {
		return invalidRequest("response schema has no object definition")
	}
	return nil
}

func (m Model) validateProtocolBlock(block Block) error {
	if block.Type == BlockReasoning && m.API != "" && m.API != APIOpenAIResponses {
		return fmt.Errorf("opaque reasoning blocks belong to the OpenAI Responses protocol")
	}
	if block.Type != BlockThinking {
		return nil
	}
	if block.Signature != "" && m.API != "" && m.API != APIAnthropicMessages {
		return fmt.Errorf("signed thinking blocks belong to the Anthropic Messages protocol")
	}
	switch m.API {
	case APIAnthropicMessages:
		if block.Text != "" && block.Signature == "" {
			return fmt.Errorf("Anthropic thinking replay requires its signature")
		}
	case APIOpenAIChat:
		if !CompatOf[OpenAIChatCompat](m).ReasoningContent {
			return fmt.Errorf("this Chat Completions endpoint cannot replay thinking blocks")
		}
	}
	return nil
}

func validateBlock(role Role, block Block) error {
	noPointers := block.Image == nil && block.ToolCall == nil &&
		block.ToolResult == nil && block.Reasoning == nil
	switch block.Type {
	case BlockText:
		if !noPointers || block.Signature != "" {
			return fmt.Errorf("text block contains a payload for another block type")
		}
	case BlockImage:
		if role != RoleUser {
			return fmt.Errorf("image block belongs to a user message")
		}
		if block.Image == nil || block.Text != "" || block.Signature != "" ||
			block.ToolCall != nil || block.ToolResult != nil || block.Reasoning != nil {
			return fmt.Errorf("image block must contain only an image payload")
		}
	case BlockThinking:
		if role != RoleAssistant {
			return fmt.Errorf("thinking block belongs to an assistant message")
		}
		if !noPointers {
			return fmt.Errorf("thinking block contains a payload for another block type")
		}
	case BlockToolCall:
		if role != RoleAssistant {
			return fmt.Errorf("tool-call block belongs to an assistant message")
		}
		if block.ToolCall == nil || block.Text != "" || block.Signature != "" ||
			block.Image != nil || block.ToolResult != nil || block.Reasoning != nil {
			return fmt.Errorf("tool-call block must contain only a tool-call payload")
		}
	case BlockToolResult:
		if role != RoleUser {
			return fmt.Errorf("tool-result block belongs to a user message")
		}
		if block.ToolResult == nil || block.Text != "" || block.Signature != "" ||
			block.Image != nil || block.ToolCall != nil || block.Reasoning != nil {
			return fmt.Errorf("tool-result block must contain only a tool-result payload")
		}
	case BlockReasoning:
		if role != RoleAssistant {
			return fmt.Errorf("reasoning block belongs to an assistant message")
		}
		if block.Reasoning == nil || block.Text != "" || block.Signature != "" ||
			block.Image != nil || block.ToolCall != nil || block.ToolResult != nil {
			return fmt.Errorf("reasoning block must contain only a reasoning payload")
		}
	default:
		return fmt.Errorf("unknown block type %q", block.Type)
	}
	return nil
}

func invalidRequest(format string, args ...any) error {
	return &Error{Kind: KindInvalidRequest, Message: "ai: " + fmt.Sprintf(format, args...)}
}

func invalidBlock(messageIndex, blockIndex int, format string, args ...any) error {
	where := fmt.Sprintf("message %d", messageIndex)
	if blockIndex >= 0 {
		where += fmt.Sprintf(" block %d", blockIndex)
	}
	return &Error{
		Kind:    KindInvalidRequest,
		Message: fmt.Sprintf("ai: %s: %s", where, fmt.Sprintf(format, args...)),
	}
}

func (m Model) unsupported(format string, args ...any) error {
	return &Error{
		Kind:    KindUnsupported,
		Message: fmt.Sprintf("model %s %s", m, fmt.Sprintf(format, args...)),
	}
}
