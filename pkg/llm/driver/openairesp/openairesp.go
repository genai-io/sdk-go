// Package openairesp implements the OpenAI Responses protocol.
//
// Responses is OpenAI's current API and the only one that round-trips a
// reasoning model's internal state, so it is what the openai vendor uses. The
// older Chat Completions protocol lives in package openaichat and is what the
// rest of the industry implements. Import this one for its side effect:
//
//	import _ "github.com/genai-io/sdk-go/pkg/llm/driver/openairesp"
package openairesp

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
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Name is the driver's identifier.
const Name = string(llm.APIOpenAIResponses)

func init() { llm.RegisterAPI(llm.APIOpenAIResponses, New) }

// Driver talks to one Responses endpoint.
type Driver struct {
	client sdk.Client
	model  llm.Model
	compat llm.OpenAIResponsesCompat
}

// New builds a driver from a Config. Registered as the factory for
// llm.APIOpenAIResponses.
func New(cfg llm.Config) (llm.Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}
	opts := []option.RequestOption{option.WithMaxRetries(0)}
	if url := cfg.Endpoint(); url != "" {
		opts = append(opts, option.WithBaseURL(url))
	}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
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
		compat: llm.CompatOf[llm.OpenAIResponsesCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Generate runs one Responses call.
func (d *Driver) Generate(ctx context.Context, p *llm.Prompt, opts llm.Options) iter.Seq2[llm.Delta, error] {
	return func(yield func(llm.Delta, error) bool) {
		stream := d.client.Responses.NewStreaming(ctx, d.buildParams(p, opts))
		defer stream.Close()

		// Responses identifies a function call by output-item ID while its
		// arguments stream, so calls are collected and emitted at the end.
		calls := make(map[string]*llm.ToolCall)
		order := make(map[string]int)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				if !yield(llm.Delta{Text: event.AsResponseOutputTextDelta().Delta}, nil) {
					return
				}

			case "response.reasoning_summary_part.added":
				// The summary streams as discrete parts, each a self-contained
				// "**headline**\n\nbody" with no separator between them.
				// Without a break, two adjacent bold headlines collide into
				// "…truncation****Updating…".
				if event.AsResponseReasoningSummaryPartAdded().SummaryIndex > 0 {
					if !yield(llm.Delta{Thinking: "\n\n"}, nil) {
						return
					}
				}

			case "response.reasoning_summary_text.delta":
				if !yield(llm.Delta{Thinking: event.AsResponseReasoningSummaryTextDelta().Delta}, nil) {
					return
				}

			case "response.reasoning_text.delta":
				if !yield(llm.Delta{Thinking: event.AsResponseReasoningTextDelta().Delta}, nil) {
					return
				}

			case "response.output_item.added":
				item := event.AsResponseOutputItemAdded()
				if item.Item.Type != "function_call" {
					continue
				}
				fn := item.Item.AsFunctionCall()
				calls[fn.ID] = &llm.ToolCall{ID: fn.CallID, Name: fn.Name}
				order[fn.ID] = len(order)

			case "response.function_call_arguments.delta":
				delta := event.AsResponseFunctionCallArgumentsDelta()
				if call, ok := calls[delta.ItemID]; ok {
					call.Input += delta.Delta
				}

			case "response.completed":
				resp := event.AsResponseCompleted().Response
				out := llm.Delta{Model: resp.Model, ID: resp.ID}

				if d.compat.Stateless {
					// Only the stateless backend returns encrypted content,
					// and only that backend needs it echoed back.
					out.Reasoning = extractReasoning(resp.Output)
				}

				// input_tokens is the whole prompt; the cached slice sits
				// under input_tokens_details.
				fresh, cached := llm.SplitPromptTokens(
					int(resp.Usage.InputTokens),
					int(resp.Usage.InputTokensDetails.CachedTokens),
				)
				out.Usage = &llm.Usage{
					Input:     fresh,
					Output:    int(resp.Usage.OutputTokens),
					CacheRead: cached,
				}

				switch resp.Status {
				case responses.ResponseStatusCompleted:
					if len(calls) > 0 {
						out.StopReason = llm.StopToolUse
					} else {
						out.StopReason = llm.StopEndTurn
					}
				case responses.ResponseStatusIncomplete:
					out.StopReason = llm.StopMaxTokens
				case responses.ResponseStatusFailed:
					yield(llm.Delta{}, d.responseError(string(resp.Error.Code), resp.Error.Message))
					return
				default:
					out.StopReason = llm.StopReason(resp.Status)
				}

				if !yield(out, nil) {
					return
				}

			case "error":
				e := event.AsError()
				yield(llm.Delta{}, d.responseError(e.Code, e.Message))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(llm.Delta{}, d.wrapStream(err))
			return
		}

		for _, id := range slices.SortedFunc(maps.Keys(calls), func(a, b string) int { return order[a] - order[b] }) {
			if !yield(llm.Delta{ToolCall: calls[id]}, nil) {
				return
			}
		}
	}
}

// Models lists the models the endpoint serves, filtered to the ones that can
// answer a Responses request — the listing also carries image, audio,
// embedding and moderation models, which would only be noise in a model
// picker.
func (d *Driver) Models(ctx context.Context) ([]llm.Model, error) {
	page, err := d.client.Models.List(ctx)
	if err != nil {
		return nil, d.wrap(err)
	}
	out := make([]llm.Model, 0, len(page.Data))
	for _, m := range page.Data {
		if !isTextModel(m.ID) {
			continue
		}
		out = append(out, llm.Model{ID: m.ID, Name: m.ID, API: llm.APIOpenAIResponses, Vendor: d.model.Vendor})
	}
	slices.SortFunc(out, func(a, b llm.Model) int { return strings.Compare(a.ID, b.ID) })
	return out, nil
}

// nonTextPrefixes and nonTextFragments name the model families that cannot
// serve a text completion.
var (
	nonTextPrefixes = []string{
		"dall-e", "tts-", "whisper-", "text-embedding", "omni-moderation",
		"davinci", "babbage", "sora", "gpt-image",
	}
	nonTextFragments = []string{"-tts", "-transcribe", "-realtime", "computer-use"}
)

func isTextModel(id string) bool {
	for _, p := range nonTextPrefixes {
		if strings.HasPrefix(id, p) {
			return false
		}
	}
	for _, f := range nonTextFragments {
		if strings.Contains(id, f) {
			return false
		}
	}
	return !strings.HasSuffix(id, "-instruct")
}

func (d *Driver) buildParams(p *llm.Prompt, opts llm.Options) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: d.model.ID,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: d.convertInput(p)},
	}
	if p.System != "" {
		params.Instructions = sdk.Opt(p.System)
	}
	if opts.Temperature > 0 {
		params.Temperature = sdk.Opt(opts.Temperature)
	}
	// The stateless backend rejects an explicit output cap along with
	// store=false, so the cap is only sent on the regular API.
	if opts.MaxTokens > 0 && !d.compat.Stateless {
		params.MaxOutputTokens = sdk.Opt(int64(opts.MaxTokens))
	}
	if d.compat.Stateless {
		params.Store = sdk.Bool(false)
		params.Include = []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		}
	}
	// The rung's value is the literal reasoning.effort wants — "none" for off,
	// through to "xhigh" and "max" where the model offers them. An empty value
	// leaves the parameter off entirely.
	if level, ok := d.model.ResolveLevel(opts.Effort); ok && level.Value != "" {
		params.Reasoning = shared.ReasoningParam{
			Effort:  shared.ReasoningEffort(level.Value),
			Summary: shared.ReasoningSummaryAuto,
		}
	}
	if choice := toolChoice(opts); choice != nil {
		params.ToolChoice = *choice
	}
	if schema := opts.Schema; schema != nil {
		if def := schemaDefinition(schema); def != nil {
			params.Text.Format = responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:        schemaName(schema),
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
	if opts.CacheRetention == llm.CacheLong && !d.compat.NoLongCacheRetention {
		params.PromptCacheRetention = responses.ResponseNewParamsPromptCacheRetention24h
	}
	native := llm.NativeOf[Native](opts)
	params.Include = append(params.Include, includables(native.Include)...)
	if native.PromptCacheKey != "" {
		params.PromptCacheKey = sdk.Opt(native.PromptCacheKey)
	}
	if len(p.Tools) > 0 {
		tools := make([]responses.ToolUnionParam, len(p.Tools))
		for i, t := range p.Tools {
			var schema map[string]any
			if props, ok := t.Parameters.(map[string]any); ok {
				schema = props
			}
			tools[i] = responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        t.Name,
					Description: sdk.Opt(t.Description),
					Parameters:  schema,
				},
			}
		}
		params.Tools = tools
	}
	return params
}

// toolChoice maps the neutral constraint onto the Responses union. A nil
// result leaves the field off, which is the API's own default.
// Native carries the Responses-only settings the normalized Options do not
// model. Pass it as llm.Options.Native; the zero value changes nothing.
type Native struct {
	// Include asks for extra response fields, e.g.
	// "reasoning.encrypted_content". The stateless backend sets that one
	// itself; this is for anything else.
	Include []string

	// PromptCacheKey routes requests that share a prefix to the same cache.
	PromptCacheKey string
}

func toolChoice(opts llm.Options) *responses.ResponseNewParamsToolChoiceUnion {
	if opts.ForceTool != "" {
		return &responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: opts.ForceTool},
		}
	}
	switch opts.ToolChoice {
	case llm.ToolChoiceNone:
		return &responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: sdk.Opt(responses.ToolChoiceOptionsNone)}
	case llm.ToolChoiceRequired:
		return &responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: sdk.Opt(responses.ToolChoiceOptionsRequired)}
	default:
		return nil
	}
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

func includables(names []string) []responses.ResponseIncludable {
	out := make([]responses.ResponseIncludable, len(names))
	for i, n := range names {
		out[i] = responses.ResponseIncludable(n)
	}
	return out
}

func (d *Driver) convertInput(p *llm.Prompt) responses.ResponseInputParam {
	msgs := llm.PrepareMessages(p.Messages)
	items := make(responses.ResponseInputParam, 0, len(msgs)+1)

	for _, msg := range msgs {
		if msg.IsToolResult() {
			for _, r := range msg.ToolResults {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
						CallID: r.ToolCallID,
						Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
							OfString: sdk.Opt(r.Content),
						},
					},
				})
			}
			continue
		}

		switch msg.Role {
		case llm.RoleUser:
			items = append(items, responses.ResponseInputItemUnionParam{
				OfMessage: d.messageParam(responses.EasyInputMessageRoleUser, msg),
			})

		case llm.RoleAssistant:
			// Reasoning items come first: the stateless backend requires a
			// reasoning model's function call to be preceded by the reasoning
			// item it came from.
			for _, r := range msg.Reasoning {
				if r.EncryptedContent == "" {
					continue
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfReasoning: reasoningParam(r)})
			}
			if !msg.Content.IsEmpty() {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfMessage: d.messageParam(responses.EasyInputMessageRoleAssistant, msg),
				})
			}
			for _, tc := range msg.ToolCalls {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfFunctionCall: &responses.ResponseFunctionToolCallParam{
						CallID:    tc.ID,
						Name:      tc.Name,
						Arguments: tc.Input,
					},
				})
			}
		}
	}
	return items
}

func (d *Driver) messageParam(role responses.EasyInputMessageRole, msg llm.Message) *responses.EasyInputMessageParam {
	param := &responses.EasyInputMessageParam{Role: role}
	if !msg.Content.HasImages() {
		param.Content = responses.EasyInputMessageContentUnionParam{OfString: sdk.Opt(msg.Content.String())}
		return param
	}
	list := make(responses.ResponseInputMessageContentListParam, 0, len(msg.Content))
	for _, p := range msg.Content {
		switch p.Type {
		case llm.PartText:
			if p.Text != "" {
				list = append(list, responses.ResponseInputContentParamOfInputText(p.Text))
			}
		case llm.PartImage:
			if p.Image != nil {
				list = append(list, imagePart(p.Image.MediaType, p.Image.Data))
			}
		}
	}
	param.Content = responses.EasyInputMessageContentUnionParam{OfInputItemContentList: list}
	return param
}

func imagePart(mediaType, data string) responses.ResponseInputContentUnionParam {
	part := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailAuto)
	if part.OfInputImage != nil {
		part.OfInputImage.ImageURL = sdk.String("data:" + mediaType + ";base64," + data)
	}
	return part
}

// reasoningParam echoes a stored reasoning item back to the stateless backend.
func reasoningParam(r llm.ReasoningItem) *responses.ResponseReasoningItemParam {
	p := &responses.ResponseReasoningItemParam{
		ID:      r.ID,
		Summary: []responses.ResponseReasoningItemSummaryParam{},
	}
	if r.EncryptedContent != "" {
		p.EncryptedContent = sdk.Opt(r.EncryptedContent)
	}
	if r.Summary != "" {
		p.Summary = []responses.ResponseReasoningItemSummaryParam{{Text: r.Summary}}
	}
	return p
}

// extractReasoning pulls the reasoning items worth replaying out of a
// completed response. An item with no encrypted content is skipped: without it
// the model cannot restore the reasoning, so echoing it back adds tokens and
// no state.
func extractReasoning(output []responses.ResponseOutputItemUnion) []llm.ReasoningItem {
	var items []llm.ReasoningItem
	for _, item := range output {
		if item.Type != "reasoning" {
			continue
		}
		r := item.AsReasoning()
		if r.EncryptedContent == "" {
			continue
		}
		var summary strings.Builder
		for i, s := range r.Summary {
			if i > 0 {
				summary.WriteString("\n\n") // keep parts separated, as in the stream
			}
			summary.WriteString(s.Text)
		}
		items = append(items, llm.ReasoningItem{
			ID:               r.ID,
			EncryptedContent: r.EncryptedContent,
			Summary:          summary.String(),
		})
	}
	return items
}

// responseError converts an in-band API failure. These arrive inside a 200
// response, so there is no status to classify from — the error code is the
// only signal for whether another attempt could work.
func (d *Driver) responseError(code, message string) error {
	if message == "" {
		message = "responses API failed"
	}
	err := &llm.Error{Driver: Name, Code: code, Message: message}
	if kind, ok := llm.ClassifyMessage(message); ok {
		err.Kind = kind
		return err
	}
	switch code {
	case string(responses.ResponseErrorCodeServerError),
		string(responses.ResponseErrorCodeRateLimitExceeded),
		string(responses.ResponseErrorCodeVectorStoreTimeout):
		err.Kind = llm.KindOverloaded
	default:
		err.Kind = llm.KindInvalidRequest
	}
	return err
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
