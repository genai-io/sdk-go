package llm

import "github.com/genai-io/sdk-go/pkg/llm/jsonschema"

// SchemaOf builds a Schema from a Go type, ready for Options.Schema.
func SchemaOf[T any](name, description string) *Schema {
	return &Schema{
		Name:        name,
		Description: description,
		Definition:  jsonschema.Of[T](),
		Strict:      true,
	}
}

// ToolFor builds a tool whose parameters are derived from a Go type, so the
// schema the model sees and the struct the arguments decode into cannot drift
// apart.
func ToolFor[T any](name, description string) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  jsonschema.Of[T](),
	}
}
