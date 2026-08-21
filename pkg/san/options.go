package san

import "github.com/genai-io/sdk-go/pkg/llm"

type agentConfig struct {
	id        string
	model     Model
	system    string
	tools     *ToolSet
	maxSteps  int
	options   *llm.Options
	inboxBuf  int
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
	if c.model == nil {
		return errMissingField("model")
	}
	if c.system == "" {
		return errMissingField("system")
	}
	return nil
}

// AgentOption is a functional option for configuring an Agent.
type AgentOption func(*agentConfig)

// WithModel sets the inference backend (required). An *llm.Client satisfies
// Model.
func WithModel(m Model) AgentOption {
	return func(c *agentConfig) { c.model = m }
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

// WithOptions sets the inference options used for every turn — reasoning
// effort, output cap, temperature. Nil leaves the model's own defaults alone.
func WithOptions(o *llm.Options) AgentOption {
	return func(c *agentConfig) { c.options = o }
}

type fieldError struct{ field string }

func (e fieldError) Error() string { return "required field missing: " + e.field }

func errMissingField(field string) error { return fieldError{field: field} }
