package responses

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	sdk "github.com/openai/openai-go/v3"
	wire "github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

// Building a request: the neutral ai.Request translated into
// what this protocol wants on the wire.

func (d *Driver) buildParams(req *ai.Request) (wire.ResponseNewParams, error) {
	native, err := ai.ProtocolOptionsAs[Options](req)
	if err != nil {
		return wire.ResponseNewParams{}, err
	}
	params := wire.ResponseNewParams{
		Model: d.model.ID,
		Input: wire.ResponseNewParamsInputUnion{OfInputItemList: d.convertInput(req)},
	}
	if req.System != "" {
		params.Instructions = sdk.Opt(req.System)
	}
	if req.Temperature != nil {
		params.Temperature = sdk.Opt(*req.Temperature)
	}
	// The stateless backend rejects an explicit output cap along with
	// store=false, so the cap is only sent on the regular API.
	if req.MaxTokens > 0 && !d.compat.Stateless {
		params.MaxOutputTokens = sdk.Opt(int64(req.MaxTokens))
	}
	if d.compat.Stateless {
		params.Store = sdk.Bool(false)
		params.Include = []wire.ResponseIncludable{
			wire.ResponseIncludableReasoningEncryptedContent,
		}
	}
	// The rung's value is the literal reasoning.effort wants — "none" for off,
	// through to "xhigh" and "max" where the model offers them. An empty value
	// leaves the parameter off entirely.
	if level, ok := d.model.ResolveLevel(req.Effort); ok && level.Value != "" {
		params.Reasoning = shared.ReasoningParam{
			Effort:  shared.ReasoningEffort(level.Value),
			Summary: shared.ReasoningSummaryAuto,
		}
	}
	if choice := toolChoice(req); choice != nil {
		params.ToolChoice = *choice
	}
	if schema := req.Schema; schema != nil {
		if def := schema.DefinitionMap(); def != nil {
			params.Text.Format = wire.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &wire.ResponseFormatTextJSONSchemaConfigParam{
					Name:        schema.WireName(),
					Description: sdk.String(schema.Description),
					Schema:      def,
					Strict:      sdk.Bool(schema.Strict),
				},
			}
		}
	}
	// A long retention is the caller asking to pay more per write for a cache
	// that survives between turns; an endpoint that rejects the field falls
	// back to its default rather than failing.
	if req.CacheRetention == ai.CacheLong && !d.compat.NoLongCacheRetention {
		params.PromptCacheRetention = wire.ResponseNewParamsPromptCacheRetention24h
	}
	params.Include = append(params.Include, includables(native.Include)...)
	if native.PromptCacheKey != "" {
		params.PromptCacheKey = sdk.Opt(native.PromptCacheKey)
	}
	if len(req.Tools) > 0 {
		tools := make([]wire.ToolUnionParam, len(req.Tools))
		for i, t := range req.Tools {
			var schema map[string]any
			if props := t.Schema.DefinitionMap(); props != nil {
				schema = props
			}
			tools[i] = wire.ToolUnionParam{
				OfFunction: &wire.FunctionToolParam{
					Name:        t.Schema.Name,
					Description: sdk.Opt(t.Schema.Description),
					Parameters:  schema,
				},
			}
		}
		params.Tools = tools
	}
	// Extension fields land last so they can deliberately override normalized
	// fields, matching the Chat Completions driver's escape-hatch semantics.
	if len(req.SamplingParams) > 0 {
		params.SetExtraFields(req.SamplingParams)
	}
	return params, nil
}

// Options carries the Responses-only settings the normalized Options do not
// model. Pass it with ai.WithProtocolOptions; the zero value changes nothing.
type Options struct {
	// Include asks for extra response fields, e.g.
	// "reasoning.encrypted_content". The stateless backend sets that one
	// itself; this is for anything else.
	Include []string

	// PromptCacheKey routes requests that share a prefix to the same cache.
	PromptCacheKey string
}

// ProtocolOptions marks this as one driver's request settings.
func (Options) ProtocolOptions() {}

// toolChoice maps the neutral constraint onto the Responses union. A nil
// result leaves the field off, which is the API's own default.
func toolChoice(req *ai.Request) *wire.ResponseNewParamsToolChoiceUnion {
	switch name, forced := req.ToolChoice.Tool(); {
	case forced:
		return &wire.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &wire.ToolChoiceFunctionParam{Name: name},
		}
	case req.ToolChoice == ai.ToolChoiceNone:
		return &wire.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: sdk.Opt(wire.ToolChoiceOptionsNone)}
	case req.ToolChoice == ai.ToolChoiceRequired:
		return &wire.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: sdk.Opt(wire.ToolChoiceOptionsRequired)}
	}
	return nil
}

func includables(names []string) []wire.ResponseIncludable {
	out := make([]wire.ResponseIncludable, len(names))
	for i, n := range names {
		out[i] = wire.ResponseIncludable(n)
	}
	return out
}

func (d *Driver) convertInput(req *ai.Request) wire.ResponseInputParam {
	items := make(wire.ResponseInputParam, 0, len(req.Messages)+1)

	for _, msg := range req.Messages {
		role := wire.EasyInputMessageRoleUser
		if msg.Role == ai.RoleAssistant {
			role = wire.EasyInputMessageRoleAssistant
		}

		// Text and image blocks share one wire message while semantic items are
		// separate. Flush each contiguous run so no block crosses or moves past
		// a reasoning/tool item.
		var message ai.Content
		flushMessage := func() {
			if len(message) == 0 {
				return
			}
			items = append(items, wire.ResponseInputItemUnionParam{
				OfMessage: d.messageParam(role, message),
			})
			message = nil
		}
		for _, block := range msg.Content {
			switch block.Type {
			case ai.BlockText, ai.BlockImage:
				message = append(message, block)
			case ai.BlockReasoning:
				flushMessage()
				if block.Reasoning != nil && block.Reasoning.EncryptedContent != "" {
					items = append(items, wire.ResponseInputItemUnionParam{OfReasoning: reasoningParam(*block.Reasoning)})
				}
			case ai.BlockToolCall:
				flushMessage()
				if block.ToolCall != nil {
					tc := block.ToolCall
					items = append(items, wire.ResponseInputItemUnionParam{
						OfFunctionCall: &wire.ResponseFunctionToolCallParam{
							CallID: tc.ID, Name: tc.Name, Arguments: tc.Input,
						},
					})
				}
			case ai.BlockToolResult:
				flushMessage()
				if block.ToolResult != nil {
					r := block.ToolResult
					items = append(items, wire.ResponseInputItemUnionParam{
						OfFunctionCallOutput: &wire.ResponseInputItemFunctionCallOutputParam{
							CallID: r.ToolCallID,
							Output: wire.ResponseInputItemFunctionCallOutputOutputUnionParam{
								OfString: sdk.Opt(r.Content),
							},
						},
					})
				}
			}
		}
		flushMessage()
	}
	return items
}

func (d *Driver) messageParam(role wire.EasyInputMessageRole, content ai.Content) *wire.EasyInputMessageParam {
	param := &wire.EasyInputMessageParam{Role: role}
	if !content.HasImages() {
		param.Content = wire.EasyInputMessageContentUnionParam{OfString: sdk.Opt(content.Text())}
		return param
	}
	list := make(wire.ResponseInputMessageContentListParam, 0, len(content))
	for _, block := range content {
		switch block.Type {
		case ai.BlockText:
			if block.Text != "" {
				list = append(list, wire.ResponseInputContentParamOfInputText(block.Text))
			}
		case ai.BlockImage:
			if block.Image != nil {
				list = append(list, imagePart(block.Image.MediaType, block.Image.Data))
			}
		}
	}
	param.Content = wire.EasyInputMessageContentUnionParam{OfInputItemContentList: list}
	return param
}

func imagePart(mediaType, data string) wire.ResponseInputContentUnionParam {
	part := wire.ResponseInputContentParamOfInputImage(wire.ResponseInputImageDetailAuto)
	if part.OfInputImage != nil {
		part.OfInputImage.ImageURL = sdk.String("data:" + mediaType + ";base64," + data)
	}
	return part
}

// reasoningParam echoes a stored reasoning item back to the stateless backend.
func reasoningParam(r ai.ReasoningItem) *wire.ResponseReasoningItemParam {
	p := &wire.ResponseReasoningItemParam{
		ID:      r.ID,
		Summary: []wire.ResponseReasoningItemSummaryParam{},
	}
	if r.EncryptedContent != "" {
		p.EncryptedContent = sdk.Opt(r.EncryptedContent)
	}
	if r.Summary != "" {
		p.Summary = []wire.ResponseReasoningItemSummaryParam{{Text: r.Summary}}
	}
	return p
}
