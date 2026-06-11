package san

import (
	"context"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Tool represents a single capability an agent can execute.
type Tool interface {
	Name() string
	Description() string
	Schema() llm.ToolSchema
	Execute(ctx context.Context, input map[string]any) (string, error)
}

// ToolSet is a mutable collection of tools.
type ToolSet struct {
	tools map[string]Tool
}

// NewToolSet creates an empty tool set.
func NewToolSet() *ToolSet {
	return &ToolSet{tools: make(map[string]Tool)}
}

// Add registers or replaces a tool.
func (ts *ToolSet) Add(t Tool) {
	ts.tools[t.Name()] = t
}

// Remove unregisters a tool by name. No-op if absent.
func (ts *ToolSet) Remove(name string) {
	delete(ts.tools, name)
}

// Get returns a tool by name, or nil if not found.
func (ts *ToolSet) Get(name string) Tool {
	return ts.tools[name]
}

// Schemas returns all tool schemas for sending to the LLM.
func (ts *ToolSet) Schemas() []llm.ToolSchema {
	schemas := make([]llm.ToolSchema, 0, len(ts.tools))
	for _, t := range ts.tools {
		schemas = append(schemas, t.Schema())
	}
	return schemas
}
