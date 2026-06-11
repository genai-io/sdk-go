package san

import "github.com/genai-io/sdk-go/pkg/llm"

type agentConfig struct {
	id       string
	llm      llm.LLM
	system   string
	tools    *ToolSet
	maxSteps int
	inboxBuf int
	outboxBuf int
}

func defaultConfig() agentConfig {
	return agentConfig{
		id:        "agent",
		maxSteps:  0, // unlimited
		inboxBuf:  16,
		outboxBuf: 64,
		tools:     NewToolSet(),
	}
}

func (c *agentConfig) validate() error {
	if c.llm == nil {
		return errMissingField("llm")
	}
	if c.system == "" {
		return errMissingField("system")
	}
	return nil
}

// AgentOption is a functional option for configuring an Agent.
type AgentOption func(*agentConfig)

// WithLLM sets the LLM backend (required).
func WithLLM(l llm.LLM) AgentOption {
	return func(c *agentConfig) { c.llm = l }
}

// WithSystem sets the system prompt (required).
func WithSystem(s string) AgentOption {
	return func(c *agentConfig) { c.system = s }
}

// WithTools sets the tool set (optional, defaults to empty).
func WithTools(tools *ToolSet) AgentOption {
	return func(c *agentConfig) { c.tools = tools }
}

// WithID sets the agent identifier (optional).
func WithID(id string) AgentOption {
	return func(c *agentConfig) { c.id = id }
}

// WithMaxSteps sets the max LLM inference steps per turn (0 = unlimited).
func WithMaxSteps(n int) AgentOption {
	return func(c *agentConfig) { c.maxSteps = n }
}

type fieldError struct{ field string }

func (e fieldError) Error() string { return "required field missing: " + e.field }

func errMissingField(field string) error { return fieldError{field: field} }
