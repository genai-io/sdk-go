package chat

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// Building a request: the neutral ai.Request translated into
// what this protocol wants on the wire.

func (d *Driver) buildParams(req *ai.Request, level ai.ReasoningLevel) sdk.ChatCompletionNewParams {
	params := sdk.ChatCompletionNewParams{
		Model:    d.model.ID,
		Messages: d.convertMessages(req),
	}
	if !d.compat.NoUsageInStream {
		params.StreamOptions = sdk.ChatCompletionStreamOptionsParam{IncludeUsage: sdk.Bool(true)}
	}
	if req.MaxTokens > 0 {
		// Servers that predate max_completion_tokens only understand the older
		// name and ignore the new one, which silently uncaps the response.
		if d.compat.MaxTokensField == "max_tokens" {
			params.MaxTokens = sdk.Int(int64(req.MaxTokens))
		} else {
			params.MaxCompletionTokens = sdk.Int(int64(req.MaxTokens))
		}
	}
	if req.Temperature != nil {
		params.Temperature = sdk.Float(*req.Temperature)
	}
	if len(req.StopSequences) > 0 {
		params.Stop = sdk.ChatCompletionNewParamsStopUnion{OfStringArray: req.StopSequences}
	}
	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
		if choice := toolChoice(req); choice != nil {
			params.ToolChoice = *choice
		}
	}
	if schema := req.Schema; schema != nil {
		if def := schema.DefinitionMap(); def != nil {
			params.ResponseFormat = sdk.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        schema.WireName(),
						Description: sdk.String(schema.Description),
						Schema:      def,
						Strict:      sdk.Bool(schema.Strict),
					},
				},
			}
		}
	}
	d.applyReasoning(&params, level)
	// Caller and model sampling parameters land last so they can reach fields
	// this driver does not model, and override the ones it does.
	if len(req.SamplingParams) > 0 {
		params.SetExtraFields(req.SamplingParams)
	}
	return params
}

// thinkingOn reports whether a rung asks for any reasoning at all.
func thinkingOn(level ai.ReasoningLevel) bool {
	return level.Value != "" || level.Budget > 0
}

// toolChoice maps the neutral constraint onto Chat Completions. A nil result
// leaves the field off, which is the API's own default.
func toolChoice(req *ai.Request) *sdk.ChatCompletionToolChoiceOptionUnionParam {
	if req.ForceTool != "" {
		return &sdk.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &sdk.ChatCompletionNamedToolChoiceParam{
				Function: sdk.ChatCompletionNamedToolChoiceFunctionParam{Name: req.ForceTool},
			},
		}
	}
	switch req.ToolChoice {
	case ai.ToolChoiceNone:
		return &sdk.ChatCompletionToolChoiceOptionUnionParam{OfAuto: sdk.Opt("none")}
	case ai.ToolChoiceRequired:
		return &sdk.ChatCompletionToolChoiceOptionUnionParam{OfAuto: sdk.Opt("required")}
	default:
		return nil
	}
}

// applyReasoning writes the endpoint's reasoning switch into the request.
//
// The rung supplies the value; this only decides which field it goes in. That
// split is why DeepSeek works: "on" is a reasoning_effort string and "off" is
// a thinking object — two different fields, which no single value could carry.
func (d *Driver) applyReasoning(params *sdk.ChatCompletionNewParams, level ai.ReasoningLevel) {
	if d.compat.Thinking == ai.ThinkingNone || level.Effort == ai.EffortDefault {
		return
	}
	on := thinkingOn(level)

	switch d.compat.Thinking {
	case ai.ThinkingEffort:
		if on {
			params.SetExtraFields(map[string]any{"reasoning_effort": level.Value})
		}

	case ai.ThinkingEffortOrDisable:
		if on {
			params.SetExtraFields(map[string]any{"reasoning_effort": level.Value})
			return
		}
		// Reasoning is on by default here, so leaving the field out keeps it
		// on: switching it off has to be explicit.
		params.SetExtraFields(map[string]any{
			"thinking": map[string]any{"type": "disabled"},
		})

	case ai.ThinkingType:
		if on {
			params.SetExtraFields(map[string]any{
				"thinking": map[string]any{"type": level.Value},
			})
		}

	case ai.ThinkingEnableFlag:
		if !on {
			return
		}
		extra := map[string]any{"enable_thinking": true}
		if level.Budget > 0 {
			extra["thinking_budget"] = level.Budget
		}
		params.SetExtraFields(extra)

	case ai.ThinkingReasoningObject:
		if on {
			params.SetExtraFields(map[string]any{
				"reasoning": map[string]any{"effort": level.Value},
			})
		}
	}
}

func (d *Driver) convertMessages(req *ai.Request) []sdk.ChatCompletionMessageParamUnion {
	out := make([]sdk.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)

	if req.System != "" {
		out = append(out, sdk.SystemMessage(req.System))
	}
	for _, msg := range req.Messages {
		if msg.HasToolResults() {
			// Chat Completions carries one tool result per message, so a turn
			// answering several calls expands into several messages.
			for _, r := range msg.ToolResults() {
				out = append(out, sdk.ToolMessage(r.Content, r.ToolCallID))
			}
			continue
		}
		switch msg.Role {
		case ai.RoleUser:
			out = append(out, d.userMessage(msg))
		case ai.RoleAssistant:
			out = append(out, d.assistantMessage(msg))
		}
	}
	return out
}

func (d *Driver) userMessage(msg ai.Message) sdk.ChatCompletionMessageParamUnion {
	// A text-only turn takes the simple string form, which every
	// implementation accepts; the array form is only needed for images.
	//
	// An image bound for a text-only endpoint is not dropped here. Model.
	// Validate refuses that request before it is sent, and a caller reaching
	// the driver directly is better served by the endpoint's own rejection
	// than by an answer about a picture the model never saw.
	if !msg.Content.HasImages() {
		return sdk.UserMessage(msg.Content.Text())
	}
	parts := make([]sdk.ChatCompletionContentPartUnionParam, 0, len(msg.Content))
	for _, block := range msg.Content {
		switch block.Type {
		case ai.BlockText:
			if block.Text == "" {
				continue
			}
			parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
				OfText: &sdk.ChatCompletionContentPartTextParam{Text: block.Text},
			})
		case ai.BlockImage:
			if block.Image == nil {
				continue
			}
			parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
				OfImageURL: &sdk.ChatCompletionContentPartImageParam{
					ImageURL: sdk.ChatCompletionContentPartImageImageURLParam{
						URL: dataURI(block.Image.MediaType, block.Image.Data),
					},
				},
			})
		}
	}
	return sdk.ChatCompletionMessageParamUnion{
		OfUser: &sdk.ChatCompletionUserMessageParam{
			Content: sdk.ChatCompletionUserMessageParamContentUnion{OfArrayOfContentParts: parts},
		},
	}
}

func (d *Driver) assistantMessage(msg ai.Message) sdk.ChatCompletionMessageParamUnion {
	var param sdk.ChatCompletionAssistantMessageParam
	if text := msg.Content.Text(); text != "" {
		param.Content.OfString = sdk.Opt(text)
	}
	if calls := msg.ToolCalls(); len(calls) > 0 {
		param.ToolCalls = convertToolCalls(calls)
	}
	if d.compat.ReasoningContent {
		// The field has to be present even when empty; these endpoints reject
		// a thinking-enabled turn whose assistant history omits it.
		param.SetExtraFields(map[string]any{"reasoning_content": msg.Content.Thinking()})
	}
	return sdk.ChatCompletionMessageParamUnion{OfAssistant: &param}
}

func convertToolCalls(calls []ai.ToolCall) []sdk.ChatCompletionMessageToolCallUnionParam {
	out := make([]sdk.ChatCompletionMessageToolCallUnionParam, len(calls))
	for i, tc := range calls {
		out[i] = sdk.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &sdk.ChatCompletionMessageFunctionToolCallParam{
				ID: tc.ID,
				Function: sdk.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      tc.Name,
					Arguments: tc.Input,
				},
			},
		}
	}
	return out
}

func convertTools(tools []ai.Tool) []sdk.ChatCompletionToolUnionParam {
	out := make([]sdk.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		var params sdk.FunctionParameters
		if props := t.ParameterSchema(); props != nil {
			params = props
		}
		out = append(out, sdk.ChatCompletionToolUnionParam{
			OfFunction: &sdk.ChatCompletionFunctionToolParam{
				Function: sdk.FunctionDefinitionParam{
					Name:        t.Name,
					Description: sdk.String(t.Description),
					Parameters:  params,
				},
			},
		})
	}
	return out
}

func dataURI(mediaType, data string) string {
	return "data:" + mediaType + ";base64," + data
}
