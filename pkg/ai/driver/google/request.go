package google

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Building a request: the neutral ai.Request translated into
// what this protocol wants on the wire.

// request assembles the body of a generate call.
func (d *Driver) request(req *ai.Request) (*generateRequest, error) {
	if err := ai.RejectProtocolOptions(req, Name); err != nil {
		return nil, err
	}
	contents, err := d.convertContents(req)
	if err != nil {
		return nil, err
	}
	body := &generateRequest{
		Contents:         contents,
		GenerationConfig: d.generationConfig(req),
		ToolConfig:       toolChoiceConfig(req),
		Tools:            declarations(req.Tools),
	}
	if req.System != "" {
		body.SystemInstruction = &content{Parts: []*part{{Text: req.System}}}
	}
	return body, nil
}

func (d *Driver) generationConfig(req *ai.Request) *generationConfig {
	cfg := &generationConfig{}
	if req.MaxTokens > 0 {
		cfg.MaxOutputTokens = int32(req.MaxTokens)
	}
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if len(req.StopSequences) > 0 {
		cfg.StopSequences = req.StopSequences
	}
	// Gemini 3 replaced the token budget with a level; 2.5 still takes a
	// budget. The rung carries whichever this model's endpoint wants.
	if level, ok := d.model.ResolveLevel(req.Effort); ok {
		switch {
		case d.compat.ThinkingLevel && level.Value != "":
			cfg.ThinkingConfig = &thinkingConfig{IncludeThoughts: true, ThinkingLevel: level.Value}
		case !d.compat.ThinkingLevel && level.Budget > 0:
			budget := int32(level.Budget)
			cfg.ThinkingConfig = &thinkingConfig{IncludeThoughts: true, ThinkingBudget: &budget}
		}
	}
	// Gemini takes the schema as raw JSON Schema alongside a JSON mime type;
	// the mime type alone would only promise valid JSON, not the shape.
	if definition := req.Schema.DefinitionMap(); definition != nil {
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJSONSchema = definition
	}
	return cfg
}

func declarations(tools []ai.Tool) []*tool {
	if len(tools) == 0 {
		return nil
	}
	decls := make([]*functionDeclaration, 0, len(tools))
	for _, t := range tools {
		decl := &functionDeclaration{Name: t.Schema.Name, Description: t.Schema.Description}
		if schema := t.Schema.DefinitionMap(); schema != nil {
			decl.ParametersJSONSchema = schema
		}
		decls = append(decls, decl)
	}
	return []*tool{{FunctionDeclarations: decls}}
}

// toolChoiceConfig maps the neutral constraint onto Gemini's function-calling
// mode. A nil result leaves the field off, which is the API's own default.
func toolChoiceConfig(req *ai.Request) *toolConfig {
	var mode string
	switch name, forced := req.ToolChoice.Tool(); {
	case forced:
		// Forcing one tool is ANY mode narrowed to a single allowed name.
		return &toolConfig{FunctionCallingConfig: &functionCallingConfig{
			Mode:                 modeAny,
			AllowedFunctionNames: []string{name},
		}}
	case req.ToolChoice == ai.ToolChoiceNone:
		mode = modeNone
	case req.ToolChoice == ai.ToolChoiceRequired:
		mode = modeAny
	default:
		return nil
	}
	return &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: mode}}
}

func (d *Driver) convertContents(req *ai.Request) ([]*content, error) {
	out := make([]*content, 0, len(req.Messages))

	for _, msg := range req.Messages {
		role := "user"
		if msg.Role == ai.RoleAssistant {
			role = "model"
		}

		parts, err := contentParts(msg.Content)
		if err != nil {
			return nil, err
		}

		if len(parts) == 0 {
			continue
		}
		out = append(out, &content{Role: role, Parts: parts})
	}
	return out, nil
}

// contentParts converts message content. Gemini takes image bytes rather than
// base64 text, so a corrupt attachment is reported here instead of failing as
// an opaque server-side rejection.
func contentParts(c ai.Content) ([]*part, error) {
	parts := make([]*part, 0, len(c))
	for _, block := range c {
		switch block.Type {
		case ai.BlockText:
			if block.Text != "" {
				parts = append(parts, &part{Text: block.Text})
			}
		case ai.BlockThinking:
			if block.Text != "" {
				parts = append(parts, &part{Text: block.Text, Thought: true})
			}
		case ai.BlockImage:
			if block.Image == nil {
				continue
			}
			decoded, err := base64.StdEncoding.DecodeString(block.Image.Data)
			if err != nil {
				return nil, &ai.Error{
					Driver:  Name,
					Kind:    ai.KindInvalidRequest,
					Message: fmt.Sprintf("image %q is not valid base64", block.Image.FileName),
					Err:     err,
				}
			}
			parts = append(parts, &part{InlineData: &blob{MIMEType: block.Image.MediaType, Data: decoded}})
		case ai.BlockToolCall:
			if block.ToolCall == nil {
				continue
			}
			tc := block.ToolCall
			var args map[string]any
			if tc.Input != "" {
				_ = json.Unmarshal([]byte(tc.Input), &args)
			}
			parts = append(parts, &part{
				FunctionCall:     &functionCall{ID: tc.ID, Name: tc.Name, Args: args},
				ThoughtSignature: tc.Signature,
			})
		case ai.BlockToolResult:
			if block.ToolResult == nil {
				continue
			}
			r := block.ToolResult
			// Gemini requires a JSON object. Wrap plain text so it still
			// round-trips instead of disappearing.
			var response map[string]any
			if err := json.Unmarshal([]byte(r.Content), &response); err != nil {
				response = map[string]any{"result": r.Content}
			}
			parts = append(parts, &part{FunctionResponse: &functionResponse{
				ID: r.ToolCallID, Name: r.ToolName, Response: response,
			}})
		}
	}
	return parts, nil
}
