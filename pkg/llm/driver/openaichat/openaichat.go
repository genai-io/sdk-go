// Package openaichat implements the OpenAI Chat Completions protocol.
//
// Chat Completions is the industry's interchange format, so this one driver
// serves most of the catalog: DeepSeek, Moonshot, Alibaba DashScope, Z.ai,
// SenseNova, Agnes-AI, GitHub Copilot and a local Ollama all speak it. What
// separates them is a base URL and a reasoning dialect, both of which arrive
// as catalog data rather than code. Import it for its side effect:
//
//	import _ "github.com/genai-io/sdk-go/pkg/llm/driver/openaichat"
package openaichat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"slices"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Name is the driver's identifier.
const Name = string(llm.APIOpenAIChat)

func init() { llm.RegisterAPI(llm.APIOpenAIChat, New) }

// Driver talks to one Chat Completions endpoint.
type Driver struct {
	client sdk.Client
	model  llm.Model
	compat llm.OpenAIChatCompat
}

// New builds a driver from a Config. Registered as the factory for
// llm.APIOpenAIChat.
func New(cfg llm.Config) (llm.Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}

	opts := []option.RequestOption{option.WithMaxRetries(0)}
	if url := cfg.Endpoint(); url != "" {
		opts = append(opts, option.WithBaseURL(url))
	}
	// Keyless endpoints exist — a local Ollama ignores the header entirely —
	// but the SDK still wants a value, so send a placeholder rather than an
	// empty credential that reads as a configuration mistake.
	key := cfg.APIKey
	if key == "" {
		key = "unused"
	}
	opts = append(opts, option.WithAPIKey(key))
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	// Model headers first so a Config header of the same name wins.
	for k, v := range cfg.Model.Headers {
		opts = append(opts, option.WithHeader(k, v))
	}
	for k, v := range cfg.Headers {
		opts = append(opts, option.WithHeader(k, v))
	}

	return &Driver{
		client: sdk.NewClient(opts...),
		model:  cfg.Model,
		compat: llm.CompatOf[llm.OpenAIChatCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Generate runs one Chat Completions call.
func (d *Driver) Generate(ctx context.Context, p *llm.Prompt, opts llm.Options) iter.Seq2[llm.Delta, error] {
	return func(yield func(llm.Delta, error) bool) {
		level, _ := d.model.ResolveLevel(opts.Effort)
		params := d.buildParams(p, opts, level)

		stream := d.client.Chat.Completions.NewStreaming(ctx, params)
		defer stream.Close()

		// Tool calls arrive as indexed argument fragments spread across
		// chunks, so they can only be emitted once the stream ends.
		calls := make(map[int]*llm.ToolCall)
		reasoning := d.compat.Thinking != llm.ThinkingNone && thinkingOn(level)

		for stream.Next() {
			chunk := stream.Current()

			for _, choice := range chunk.Choices {
				var out llm.Delta
				if reasoning {
					out.Thinking = reasoningContent(choice.Delta.RawJSON())
				}
				out.Text = choice.Delta.Content
				if choice.FinishReason != "" && !d.compat.NoFinishReason {
					out.StopReason = mapFinishReason(choice.FinishReason)
				}
				if chunk.Model != "" {
					out.Model = chunk.Model
				}
				if chunk.ID != "" {
					out.ID = chunk.ID
				}

				for _, tc := range choice.Delta.ToolCalls {
					idx := int(tc.Index)
					if _, ok := calls[idx]; !ok {
						calls[idx] = &llm.ToolCall{ID: tc.ID, Name: tc.Function.Name}
					}
					// An ID or name can arrive on a later fragment than the
					// first one for that index.
					if tc.ID != "" {
						calls[idx].ID = tc.ID
					}
					if tc.Function.Name != "" {
						calls[idx].Name = tc.Function.Name
					}
					calls[idx].Input += tc.Function.Arguments
				}

				if out.Text != "" || out.Thinking != "" || out.StopReason != "" {
					if !yield(out, nil) {
						return
					}
				}
			}

			// prompt_tokens is the whole prompt; the cached slice sits under
			// prompt_tokens_details. Split it so Usage.Input stays "fresh
			// tokens only" across every protocol.
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				fresh, cached := llm.SplitPromptTokens(
					int(chunk.Usage.PromptTokens),
					int(chunk.Usage.PromptTokensDetails.CachedTokens),
				)
				if !yield(llm.Delta{Usage: &llm.Usage{
					Input:     fresh,
					Output:    int(chunk.Usage.CompletionTokens),
					CacheRead: cached,
				}}, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(llm.Delta{}, d.wrapStream(err))
			return
		}

		for _, idx := range slices.Sorted(maps.Keys(calls)) {
			if !yield(llm.Delta{ToolCall: calls[idx]}, nil) {
				return
			}
		}
	}
}

// Models lists what the endpoint serves. Most OpenAI-compatible endpoints
// return the bare shape — id, object, owned_by — with no limits; where one
// includes context_length it is read out of the raw JSON, since the typed SDK
// struct has no field for a non-standard extension.
func (d *Driver) Models(ctx context.Context) ([]llm.Model, error) {
	page, err := d.client.Models.List(ctx)
	if err != nil {
		return nil, d.wrap(err)
	}
	out := make([]llm.Model, 0, len(page.Data))
	for _, m := range page.Data {
		model := llm.Model{ID: m.ID, Name: m.ID, API: llm.APIOpenAIChat, Vendor: d.model.Vendor}
		if raw := m.RawJSON(); raw != "" {
			var extra struct {
				ContextLength int `json:"context_length"`
			}
			if json.Unmarshal([]byte(raw), &extra) == nil && extra.ContextLength > 0 {
				model.ContextWindow = extra.ContextLength
			}
		}
		out = append(out, model)
	}
	return out, nil
}

func (d *Driver) buildParams(p *llm.Prompt, opts llm.Options, level llm.ReasoningLevel) sdk.ChatCompletionNewParams {
	params := sdk.ChatCompletionNewParams{
		Model:    d.model.ID,
		Messages: d.convertMessages(p),
	}
	if !d.compat.NoUsageInStream {
		params.StreamOptions = sdk.ChatCompletionStreamOptionsParam{IncludeUsage: sdk.Bool(true)}
	}
	if opts.MaxTokens > 0 {
		// Servers that predate max_completion_tokens only understand the older
		// name and ignore the new one, which silently uncaps the response.
		if d.compat.MaxTokensField == "max_tokens" {
			params.MaxTokens = sdk.Int(int64(opts.MaxTokens))
		} else {
			params.MaxCompletionTokens = sdk.Int(int64(opts.MaxTokens))
		}
	}
	if opts.Temperature > 0 {
		params.Temperature = sdk.Float(opts.Temperature)
	}
	if len(opts.StopSequences) > 0 {
		params.Stop = sdk.ChatCompletionNewParamsStopUnion{OfStringArray: opts.StopSequences}
	}
	if len(p.Tools) > 0 {
		params.Tools = convertTools(p.Tools)
		if choice := toolChoice(opts); choice != nil {
			params.ToolChoice = *choice
		}
	}
	if schema := opts.Schema; schema != nil {
		if def := schemaDefinition(schema); def != nil {
			params.ResponseFormat = sdk.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        schemaName(schema),
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
	if len(opts.SamplingParams) > 0 {
		params.SetExtraFields(opts.SamplingParams)
	}
	return params
}

// thinkingOn reports whether a rung asks for any reasoning at all.
func thinkingOn(level llm.ReasoningLevel) bool {
	return level.Value != "" || level.Budget > 0
}

// toolChoice maps the neutral constraint onto Chat Completions. A nil result
// leaves the field off, which is the API's own default.
func toolChoice(opts llm.Options) *sdk.ChatCompletionToolChoiceOptionUnionParam {
	if opts.ForceTool != "" {
		return &sdk.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &sdk.ChatCompletionNamedToolChoiceParam{
				Function: sdk.ChatCompletionNamedToolChoiceFunctionParam{Name: opts.ForceTool},
			},
		}
	}
	switch opts.ToolChoice {
	case llm.ToolChoiceNone:
		return &sdk.ChatCompletionToolChoiceOptionUnionParam{OfAuto: sdk.Opt("none")}
	case llm.ToolChoiceRequired:
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
func (d *Driver) applyReasoning(params *sdk.ChatCompletionNewParams, level llm.ReasoningLevel) {
	if d.compat.Thinking == llm.ThinkingNone || level.Effort == llm.EffortDefault {
		return
	}
	on := thinkingOn(level)

	switch d.compat.Thinking {
	case llm.ThinkingEffort:
		if on {
			params.SetExtraFields(map[string]any{"reasoning_effort": level.Value})
		}

	case llm.ThinkingEffortOrDisable:
		if on {
			params.SetExtraFields(map[string]any{"reasoning_effort": level.Value})
			return
		}
		// Reasoning is on by default here, so leaving the field out keeps it
		// on: switching it off has to be explicit.
		params.SetExtraFields(map[string]any{
			"thinking": map[string]any{"type": "disabled"},
		})

	case llm.ThinkingType:
		if on {
			params.SetExtraFields(map[string]any{
				"thinking": map[string]any{"type": level.Value},
			})
		}

	case llm.ThinkingEnableFlag:
		if !on {
			return
		}
		extra := map[string]any{"enable_thinking": true}
		if level.Budget > 0 {
			extra["thinking_budget"] = level.Budget
		}
		params.SetExtraFields(extra)

	case llm.ThinkingReasoningObject:
		if on {
			params.SetExtraFields(map[string]any{
				"reasoning": map[string]any{"effort": level.Value},
			})
		}
	}
}

func (d *Driver) convertMessages(p *llm.Prompt) []sdk.ChatCompletionMessageParamUnion {
	msgs := llm.PrepareMessages(p.Messages)
	out := make([]sdk.ChatCompletionMessageParamUnion, 0, len(msgs)+1)

	if p.System != "" {
		out = append(out, sdk.SystemMessage(p.System))
	}
	for _, msg := range msgs {
		if msg.IsToolResult() {
			// Chat Completions carries one tool result per message, so a turn
			// answering several calls expands into several messages.
			for _, r := range msg.ToolResults {
				out = append(out, sdk.ToolMessage(r.Content, r.ToolCallID))
			}
			continue
		}
		switch msg.Role {
		case llm.RoleUser:
			out = append(out, d.userMessage(msg))
		case llm.RoleAssistant:
			out = append(out, d.assistantMessage(msg))
		}
	}
	return out
}

func (d *Driver) userMessage(msg llm.Message) sdk.ChatCompletionMessageParamUnion {
	// A text-only turn takes the simple string form, which every
	// implementation accepts; the array form is only needed for images.
	//
	// An image bound for a text-only endpoint is not dropped here. Model.
	// Validate refuses that request before it is sent, and a caller reaching
	// the driver directly is better served by the endpoint's own rejection
	// than by an answer about a picture the model never saw.
	if !msg.Content.HasImages() {
		return sdk.UserMessage(msg.Content.String())
	}
	parts := make([]sdk.ChatCompletionContentPartUnionParam, 0, len(msg.Content))
	for _, p := range msg.Content {
		switch p.Type {
		case llm.PartText:
			if p.Text == "" {
				continue
			}
			parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
				OfText: &sdk.ChatCompletionContentPartTextParam{Text: p.Text},
			})
		case llm.PartImage:
			if p.Image == nil {
				continue
			}
			parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
				OfImageURL: &sdk.ChatCompletionContentPartImageParam{
					ImageURL: sdk.ChatCompletionContentPartImageImageURLParam{
						URL: dataURI(p.Image.MediaType, p.Image.Data),
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

func (d *Driver) assistantMessage(msg llm.Message) sdk.ChatCompletionMessageParamUnion {
	var param sdk.ChatCompletionAssistantMessageParam
	if text := msg.Content.String(); text != "" {
		param.Content.OfString = sdk.Opt(text)
	}
	if len(msg.ToolCalls) > 0 {
		param.ToolCalls = convertToolCalls(msg.ToolCalls)
	}
	if d.compat.ReasoningContent {
		// The field has to be present even when empty; these endpoints reject
		// a thinking-enabled turn whose assistant history omits it.
		param.SetExtraFields(map[string]any{"reasoning_content": msg.Thinking})
	}
	return sdk.ChatCompletionMessageParamUnion{OfAssistant: &param}
}

func convertToolCalls(calls []llm.ToolCall) []sdk.ChatCompletionMessageToolCallUnionParam {
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

func convertTools(tools []llm.Tool) []sdk.ChatCompletionToolUnionParam {
	out := make([]sdk.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		var params sdk.FunctionParameters
		if props, ok := t.Parameters.(map[string]any); ok {
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

// schemaDefinition renders a schema as the map the parameter type takes.
func schemaDefinition(s *llm.Schema) map[string]any {
	raw, err := json.Marshal(s.Definition)
	if err != nil {
		return nil
	}
	var out map[string]any
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}

// schemaName supplies the identifier these protocols require.
func schemaName(s *llm.Schema) string {
	if s.Name == "" {
		return "response"
	}
	return s.Name
}

func dataURI(mediaType, data string) string {
	return "data:" + mediaType + ";base64," + data
}

// reasoningContent reads the reasoning_content extension out of a raw stream
// delta. It is not part of the standard schema, so the typed SDK struct has no
// field for it, but Moonshot, DeepSeek, Alibaba and Z.ai all stream reasoning
// there.
func reasoningContent(rawJSON string) string {
	if rawJSON == "" {
		return ""
	}
	var delta struct {
		ReasoningContent string `json:"reasoning_content"`
	}
	if err := json.Unmarshal([]byte(rawJSON), &delta); err != nil {
		return ""
	}
	return delta.ReasoningContent
}

func mapFinishReason(reason string) llm.StopReason {
	switch reason {
	case "stop":
		return llm.StopEndTurn
	case "tool_calls", "function_call":
		return llm.StopToolUse
	case "length":
		return llm.StopMaxTokens
	case "content_filter":
		return llm.StopRefusal
	case "":
		return ""
	default:
		return llm.StopReason(reason)
	}
}

func (d *Driver) wrap(err error) error {
	status, code, msg, resp := errorDetails(err)
	return llm.Classify(Name, status, resp, code, msg, err)
}

func (d *Driver) wrapStream(err error) error {
	status, code, msg, resp := errorDetails(err)
	return llm.StreamError(Name, status, resp, code, msg, err)
}

func errorDetails(err error) (status int, code, message string, resp *http.Response) {
	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		return 0, "", "", nil
	}
	message = strings.TrimSpace(apiErr.Message)
	if message == "" {
		message = strings.TrimSpace(apiErr.RawJSON())
	}
	return apiErr.StatusCode, apiErr.Code, message, apiErr.Response
}

var _ llm.Driver = (*Driver)(nil)
