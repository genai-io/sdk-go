// Package anthropic implements the Anthropic Messages protocol.
//
// It backs every vendor whose endpoint speaks that protocol — Anthropic
// itself, MiniMax, Xiaomi MiMo and Volcengine Ark — which differ only in base
// URL, credential and how the key is presented. Import it for its side effect
// to make llm.Open handle llm.APIAnthropicMessages:
//
//	import _ "github.com/genai-io/sdk-go/pkg/llm/driver/anthropic"
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"regexp"
	"slices"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Name is the driver's identifier, matching the protocol it speaks.
const Name = string(llm.APIAnthropicMessages)

// defaultMaxTokens is what the protocol requires and the caller did not
// supply. Anthropic rejects a request without max_tokens, so unlike the
// OpenAI protocols there is no "leave it to the server" option.
const defaultMaxTokens = 8192

// extendedContextSuffix marks a model ID as the 1M-context variant of a model
// that is otherwise identical, e.g. "claude-opus-4-6[1m]". The suffix is not
// part of the ID Anthropic knows; it selects a beta header.
const extendedContextSuffix = "[1m]"

const extendedContextBeta = "context-1m-2025-08-07"

func init() { llm.RegisterAPI(llm.APIAnthropicMessages, New) }

// Driver talks to one Anthropic-protocol endpoint.
type Driver struct {
	client  sdk.Client
	model   llm.Model
	modelID string
	reqOpts []option.RequestOption
	compat  llm.AnthropicCompat
}

// New builds a driver from a Config. It is registered as the factory for
// llm.APIAnthropicMessages, so llm.Open reaches it without an explicit call.
func New(cfg llm.Config) (llm.Driver, error) {
	return NewWithClient(sdk.NewClient(ClientOptions(cfg)...), cfg)
}

// ClientOptions builds the SDK options a Config asks for: endpoint,
// credential, transport and headers.
//
// It is exported for the sake of a driver that speaks this protocol over a
// different transport — Vertex AI authenticates with Google credentials rather
// than an API key, but everything downstream of client construction is
// identical. Such a driver builds its own client from these options plus its
// own, then hands it to NewWithClient.
func ClientOptions(cfg llm.Config) []option.RequestOption {
	opts := []option.RequestOption{
		// Retries belong to the caller, which alone knows the budget for the
		// whole turn. An SDK retrying underneath would multiply it silently.
		option.WithMaxRetries(0),
	}
	if url := cfg.Endpoint(); url != "" {
		opts = append(opts, option.WithBaseURL(url))
	}
	if cfg.APIKey != "" {
		// Ark and other re-hosts take the key as a bearer token; Anthropic
		// itself wants x-api-key.
		if llm.CompatOf[llm.AnthropicCompat](cfg.Model).BearerAuth {
			opts = append(opts, option.WithAuthToken(cfg.APIKey))
		} else {
			opts = append(opts, option.WithAPIKey(cfg.APIKey))
		}
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
	return opts
}

// NewWithClient builds a driver over an already-constructed SDK client.
//
// Everything this driver does after construction — message conversion,
// thinking configuration, streaming, error classification — is the same
// whatever produced the client, so a transport that the Config cannot express
// supplies its own client rather than forking the driver.
func NewWithClient(client sdk.Client, cfg llm.Config) (llm.Driver, error) {
	modelID, extended := strings.CutSuffix(cfg.Model.ID, extendedContextSuffix)
	if modelID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}

	d := &Driver{
		client:  client,
		model:   cfg.Model,
		modelID: modelID,
		compat:  llm.CompatOf[llm.AnthropicCompat](cfg.Model),
	}
	if extended {
		d.reqOpts = append(d.reqOpts, option.WithHeader("anthropic-beta", extendedContextBeta))
	}
	return d, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Generate runs one Messages call.
func (d *Driver) Generate(ctx context.Context, p *llm.Prompt, opts llm.Options) iter.Seq2[llm.Delta, error] {
	return func(yield func(llm.Delta, error) bool) {
		params, err := d.buildParams(p, opts)
		if err != nil {
			yield(llm.Delta{}, err)
			return
		}

		reqOpts := d.reqOpts
		if betas := llm.NativeOf[Native](opts).Betas; len(betas) > 0 {
			reqOpts = append(slices.Clone(reqOpts), option.WithHeader("anthropic-beta", strings.Join(betas, ",")))
		}

		stream := d.client.Messages.NewStreaming(ctx, *params, reqOpts...)
		defer stream.Close()

		var toolID, toolName string
		var toolInput strings.Builder

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "message_start":
				start := event.AsMessageStart()
				if !yield(llm.Delta{
					Model: string(start.Message.Model),
					ID:    start.Message.ID,
					Usage: &llm.Usage{
						Input:      int(start.Message.Usage.InputTokens),
						CacheWrite: int(start.Message.Usage.CacheCreationInputTokens),
						// The 1-hour slice is billed at twice the input rate,
						// so it has to travel separately from the total.
						CacheWrite1h: int(start.Message.Usage.CacheCreation.Ephemeral1hInputTokens),
						CacheRead:    int(start.Message.Usage.CacheReadInputTokens),
					},
				}, nil) {
					return
				}

			case "content_block_start":
				block := event.AsContentBlockStart()
				if block.ContentBlock.Type == "tool_use" {
					toolID = block.ContentBlock.ID
					toolName = block.ContentBlock.Name
					toolInput.Reset()
				}

			case "content_block_delta":
				delta := event.AsContentBlockDelta()
				var out llm.Delta
				switch delta.Delta.Type {
				case "text_delta":
					out.Text = delta.Delta.Text
				case "thinking_delta":
					out.Thinking = delta.Delta.Thinking
				case "signature_delta":
					out.ThinkingSignature = delta.Delta.Signature
				case "input_json_delta":
					toolInput.WriteString(delta.Delta.PartialJSON)
					continue
				default:
					continue
				}
				if !yield(out, nil) {
					return
				}

			case "content_block_stop":
				if toolID == "" || toolName == "" {
					// A text or thinking block ended. Anthropic is one of the
					// few protocols that says so, which is what lets two
					// adjacent blocks of the same kind be told apart.
					if !yield(llm.Delta{EndBlock: true}, nil) {
						return
					}
					continue
				}
				call := llm.ToolCall{ID: toolID, Name: toolName, Input: toolInput.String()}
				toolID, toolName = "", ""
				toolInput.Reset()
				if !yield(llm.Delta{ToolCall: &call}, nil) {
					return
				}

			case "message_delta":
				md := event.AsMessageDelta()
				// Anthropic reports only output tokens here. Some
				// Anthropic-compatible endpoints (SenseNova) instead send
				// input tokens in message_delta rather than message_start;
				// zeros are ignored on merge, so passing both is safe either
				// way.
				if !yield(llm.Delta{
					StopReason: mapStopReason(string(md.Delta.StopReason)),
					Usage: &llm.Usage{
						Input:      int(md.Usage.InputTokens),
						Output:     int(md.Usage.OutputTokens),
						CacheWrite: int(md.Usage.CacheCreationInputTokens),
						CacheRead:  int(md.Usage.CacheReadInputTokens),
					},
				}, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(llm.Delta{}, d.wrapStream(err))
		}
	}
}

// mapStopReason translates Anthropic's stop reasons.
func mapStopReason(reason string) llm.StopReason {
	switch reason {
	case "end_turn":
		return llm.StopEndTurn
	case "tool_use":
		return llm.StopToolUse
	case "max_tokens":
		return llm.StopMaxTokens
	case "stop_sequence":
		return llm.StopSequence
	case "refusal":
		return llm.StopRefusal
	case "":
		return ""
	default:
		return llm.StopReason(reason)
	}
}

// CountTokens asks the endpoint how large a prompt is, without generating
// from it. Anthropic publishes this, so a caller never has to estimate.
func (d *Driver) CountTokens(ctx context.Context, p *llm.Prompt, opts llm.Options) (int, error) {
	params, err := d.buildParams(p, opts)
	if err != nil {
		return 0, err
	}
	count := sdk.MessageCountTokensParams{
		Model:    params.Model,
		Messages: params.Messages,
		System: sdk.MessageCountTokensParamsSystemUnion{
			OfTextBlockArray: params.System,
		},
		Tools: countTokenTools(params.Tools),
	}
	if !param.IsOmitted(params.Thinking) {
		count.Thinking = params.Thinking
	}
	res, err := d.client.Messages.CountTokens(ctx, count, d.reqOpts...)
	if err != nil {
		return 0, d.wrap(err)
	}
	return int(res.InputTokens), nil
}

// countTokenTools re-types the tool params for the counting endpoint, which
// takes the same shapes under its own union.
func countTokenTools(tools []sdk.ToolUnionParam) []sdk.MessageCountTokensToolUnionParam {
	out := make([]sdk.MessageCountTokensToolUnionParam, 0, len(tools))
	for _, t := range tools {
		if t.OfTool == nil {
			continue
		}
		out = append(out, sdk.MessageCountTokensToolUnionParam{OfTool: t.OfTool})
	}
	return out
}

// Models lists the models the endpoint serves. Anthropic's listing carries IDs
// and display names only — no limits — so callers wanting context windows
// should merge catalog data over the result (see catalog.Enrich).
func (d *Driver) Models(ctx context.Context) ([]llm.Model, error) {
	pager := d.client.Models.ListAutoPaging(ctx, sdk.ModelListParams{})
	var out []llm.Model
	for pager.Next() {
		m := pager.Current()
		name := m.DisplayName
		if name == "" {
			name = m.ID
		}
		out = append(out, llm.Model{
			ID:     m.ID,
			Name:   name,
			API:    llm.APIAnthropicMessages,
			Vendor: d.model.Vendor,
		})
	}
	if err := pager.Err(); err != nil {
		return nil, d.wrap(err)
	}
	return out, nil
}

func (d *Driver) wrap(err error) error {
	if err == nil {
		return nil
	}
	status, code, msg := errorDetails(err)
	return llm.Classify(Name, status, httpResponse(err), code, msg, err)
}

func (d *Driver) wrapStream(err error) error {
	if err == nil {
		return nil
	}
	status, code, msg := errorDetails(err)
	return llm.StreamError(Name, status, httpResponse(err), code, msg, err)
}

func errorDetails(err error) (status int, code, message string) {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode, "", strings.TrimSpace(apiErr.Error())
	}
	return 0, "", ""
}

func httpResponse(err error) *http.Response {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		return apiErr.Response
	}
	return nil
}

// validToolID matches the tool_use ID shape the Messages API accepts.
var validToolID = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// toolIDs rewrites tool call IDs another provider produced into ones Anthropic
// accepts, keeping the mapping stable within a request so calls and results
// still line up. A conversation that started on a different provider and was
// handed to Claude is otherwise rejected outright.
type toolIDs struct {
	m map[string]string
	n int
}

func (t *toolIDs) resolve(id string) string {
	if validToolID.MatchString(id) {
		return id
	}
	if t.m == nil {
		t.m = make(map[string]string)
	}
	if mapped, ok := t.m[id]; ok {
		return mapped
	}
	t.n++
	mapped := fmt.Sprintf("toolu_compat_%d", t.n)
	t.m[id] = mapped
	return mapped
}

func (d *Driver) buildParams(p *llm.Prompt, opts llm.Options) (*sdk.MessageNewParams, error) {
	var ids toolIDs

	// The rung carries both halves of the mapping: Value is what
	// output_config.effort wants under adaptive thinking, Budget is what
	// budget_tokens wants under the older shape. Neither is computed here —
	// which spelling this endpoint uses is model data.
	native := llm.NativeOf[Native](opts)
	level, hasLevel := d.model.ResolveLevel(opts.Effort)
	var budget int64
	if !d.compat.ForceAdaptiveThinking {
		budget = int64(level.Budget)
	}
	// Whether reasoning runs at all, under either shape. It gates replaying a
	// prior thinking block: the API rejects one sent into a request that is
	// not itself thinking.
	thinkingOn := budget > 0 || (d.compat.ForceAdaptiveThinking && level.Value != "")

	msgs := make([]sdk.MessageParam, 0, len(p.Messages))
	for _, msg := range llm.PrepareMessages(p.Messages) {
		if msg.IsToolResult() {
			blocks := make([]sdk.ContentBlockParamUnion, 0, len(msg.ToolResults))
			for _, r := range msg.ToolResults {
				blocks = append(blocks, sdk.NewToolResultBlock(ids.resolve(r.ToolCallID), r.Content, r.IsError))
			}
			msgs = append(msgs, sdk.NewUserMessage(blocks...))
			continue
		}

		switch msg.Role {
		case llm.RoleUser:
			if blocks := userBlocks(msg.Content); len(blocks) > 0 {
				msgs = append(msgs, sdk.NewUserMessage(blocks...))
			}
		case llm.RoleAssistant:
			blocks := assistantBlocks(msg, thinkingOn)
			for _, tc := range msg.ToolCalls {
				blocks = append(blocks, sdk.NewToolUseBlock(ids.resolve(tc.ID), toolInputValue(tc.Input), tc.Name))
			}
			if len(blocks) > 0 {
				msgs = append(msgs, sdk.NewAssistantMessage(blocks...))
			}
		}
	}
	msgs = mergeConsecutive(msgs)

	maxTokens := int64(opts.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	// budget_tokens must leave room for an answer, so max_tokens has to exceed
	// it. Raising the cap is the only fix that keeps the requested budget.
	if budget > 0 && maxTokens <= budget {
		maxTokens = budget + defaultMaxTokens
	}

	params := &sdk.MessageNewParams{
		Model:     sdk.Model(d.modelID),
		MaxTokens: maxTokens,
		Messages:  msgs,
	}
	switch {
	case d.compat.ForceAdaptiveThinking && thinkingOn:
		adaptive := &sdk.ThinkingConfigAdaptiveParam{
			Display: sdk.ThinkingConfigAdaptiveDisplay(native.ThinkingDisplay),
		}
		params.Thinking = sdk.ThinkingConfigParamUnion{OfAdaptive: adaptive}
		params.OutputConfig.Effort = sdk.OutputConfigEffort(level.Value)
	case d.compat.ForceAdaptiveThinking && hasLevel:
		// These models think by default, so "off" has to be said out loud. No
		// effort is paired with it: Opus 5 rejects disabled thinking above
		// effort "high".
		params.Thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}
	case budget > 0:
		params.Thinking = sdk.ThinkingConfigParamOfEnabled(budget)
	}
	// Claude Opus 4.7 and later reject a non-default temperature outright.
	if opts.Temperature > 0 && !d.compat.NoTemperature {
		params.Temperature = sdk.Float(opts.Temperature)
	}
	if len(opts.StopSequences) > 0 {
		params.StopSequences = opts.StopSequences
	}
	if choice := toolChoice(opts, native); choice != nil {
		params.ToolChoice = *choice
	}
	if p.System != "" {
		block := sdk.TextBlockParam{Text: p.System}
		// One cache breakpoint at the end of the system block. Anthropic
		// renders a request as tools → system → messages, so the cached prefix
		// is exactly the tool definitions plus the system prompt — which makes
		// the reported cache tokens an exact measurement of those two. Moving
		// or adding a breakpoint invalidates that reading.
		if cc := d.cacheControl(opts.CacheRetention); cc != nil {
			block.CacheControl = *cc
		}
		params.System = []sdk.TextBlockParam{block}
	}
	if len(p.Tools) > 0 {
		params.Tools = convertTools(p.Tools)
	}
	// Structured output shares output_config with the effort level, so it is
	// set on the same struct rather than replacing it.
	if def := schemaDefinition(opts.Schema); def != nil {
		params.OutputConfig.Format = sdk.JSONOutputFormatParam{Schema: def}
	}
	return params, nil
}

// schemaDefinition renders a schema as the map every protocol's parameter
// type takes, or nil when there is none.
func schemaDefinition(s *llm.Schema) map[string]any {
	if s == nil {
		return nil
	}
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

// Native carries the Anthropic-only settings the normalized Options do not
// model. Pass it as llm.Options.Native; the zero value changes nothing.
type Native struct {
	// ThinkingDisplay controls how thinking comes back: "summarized" returns
	// readable reasoning, "omitted" returns an empty thinking field with the
	// signature still attached for multi-turn continuity. Omitted is faster to
	// first text token, which is what a UI that never shows reasoning wants.
	// Empty leaves the API's own default in place.
	ThinkingDisplay string

	// Betas are extra anthropic-beta header values for this driver's requests.
	Betas []string

	// DisableParallelToolUse limits the model to one tool call per turn. It
	// only applies alongside a tool constraint.
	DisableParallelToolUse bool
}

// cacheControl maps the requested retention onto a cache_control marker. A nil
// result means no breakpoint, which is what CacheNone asks for and what an
// endpoint with no prompt cache gets.
func (d *Driver) cacheControl(retention llm.CacheRetention) *sdk.CacheControlEphemeralParam {
	if d.compat.NoPromptCache || retention == llm.CacheNone {
		return nil
	}
	cc := sdk.NewCacheControlEphemeralParam()
	// A 1-hour write costs twice the input rate against 1.25x for five
	// minutes, so it only pays off from the second read — the caller asks for
	// it deliberately, and an endpoint that rejects the field falls back
	// rather than failing.
	if retention == llm.CacheLong && !d.compat.NoLongCacheRetention {
		cc.TTL = sdk.CacheControlEphemeralTTLTTL1h
	}
	return &cc
}

// toolChoice maps the neutral constraint onto Anthropic's union. A nil result
// leaves the field off, which is the API's own default.
func toolChoice(opts llm.Options, native Native) *sdk.ToolChoiceUnionParam {
	noParallel := sdk.Bool(true)
	if !native.DisableParallelToolUse {
		noParallel = sdk.Bool(false)
	}
	if opts.ForceTool != "" {
		return &sdk.ToolChoiceUnionParam{OfTool: &sdk.ToolChoiceToolParam{
			Name:                   opts.ForceTool,
			DisableParallelToolUse: noParallel,
		}}
	}
	switch opts.ToolChoice {
	case llm.ToolChoiceNone:
		return &sdk.ToolChoiceUnionParam{OfNone: &sdk.ToolChoiceNoneParam{}}
	case llm.ToolChoiceRequired:
		return &sdk.ToolChoiceUnionParam{OfAny: &sdk.ToolChoiceAnyParam{
			DisableParallelToolUse: noParallel,
		}}
	default:
		return nil
	}
}

// toolInputValue decodes a tool call's arguments for the wire. Anthropic wants
// a JSON value, not the raw string; content that will not parse is sent
// verbatim so a malformed call surfaces as the model's own mistake rather than
// as a client error.
func toolInputValue(input string) any {
	if strings.TrimSpace(input) == "" {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal([]byte(input), &v); err != nil {
		return input
	}
	return v
}

func userBlocks(content llm.Content) []sdk.ContentBlockParamUnion {
	blocks := make([]sdk.ContentBlockParamUnion, 0, len(content))
	for _, part := range content {
		switch part.Type {
		case llm.PartText:
			if part.Text != "" {
				blocks = append(blocks, sdk.NewTextBlock(part.Text))
			}
		case llm.PartImage:
			if part.Image != nil {
				blocks = append(blocks, sdk.NewImageBlockBase64(part.Image.MediaType, part.Image.Data))
			}
		}
	}
	return blocks
}

// assistantBlocks rebuilds a prior assistant turn.
//
// A thinking block is replayed only when its signature came back with it and
// thinking is on for this request: the API rejects an unsigned thinking block,
// and replaying one into a non-thinking request is equally invalid.
func assistantBlocks(msg llm.Message, thinkingOn bool) []sdk.ContentBlockParamUnion {
	blocks := make([]sdk.ContentBlockParamUnion, 0, 2)
	if msg.Thinking != "" && msg.ThinkingSignature != "" && thinkingOn {
		blocks = append(blocks, sdk.NewThinkingBlock(msg.ThinkingSignature, msg.Thinking))
	}
	if text := msg.Content.String(); text != "" {
		blocks = append(blocks, sdk.NewTextBlock(text))
	}
	return blocks
}

// mergeConsecutive combines adjacent same-role messages. The API requires all
// tool_result blocks answering one assistant turn to arrive in a single user
// message, and rejects two user turns in a row.
func mergeConsecutive(msgs []sdk.MessageParam) []sdk.MessageParam {
	if len(msgs) <= 1 {
		return msgs
	}
	out := make([]sdk.MessageParam, 0, len(msgs))
	out = append(out, msgs[0])
	for _, msg := range msgs[1:] {
		last := &out[len(out)-1]
		if msg.Role == last.Role {
			last.Content = append(last.Content, msg.Content...)
			continue
		}
		out = append(out, msg)
	}
	return out
}

func convertTools(tools []llm.Tool) []sdk.ToolUnionParam {
	out := make([]sdk.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := sdk.ToolInputSchemaParam{}
		if props, ok := t.Parameters.(map[string]any); ok {
			if properties, ok := props["properties"]; ok {
				schema.Properties = properties
			}
			schema.Required = stringSlice(props["required"])
			schema.ExtraFields = schemaExtras(props)
		}
		out = append(out, sdk.ToolUnionParam{
			OfTool: &sdk.ToolParam{
				Name:        t.Name,
				Description: sdk.String(t.Description),
				InputSchema: schema,
			},
		})
	}
	return out
}

// schemaExtras forwards the JSON Schema keywords the SDK's typed fields do not
// cover, so a schema keeps its $defs, enums and descriptions.
func schemaExtras(schema map[string]any) map[string]any {
	extras := make(map[string]any, len(schema))
	for k, v := range schema {
		switch k {
		case "type", "properties", "required":
			continue
		default:
			extras[k] = v
		}
	}
	if len(extras) == 0 {
		return nil
	}
	return extras
}

// stringSlice normalizes a JSON Schema "required" list, which arrives as
// []string when built in Go and []any when decoded from JSON.
func stringSlice(v any) []string {
	switch list := v.(type) {
	case []string:
		return list
	case []any:
		out := make([]string, 0, len(list))
		for _, item := range list {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

var (
	_ llm.Driver       = (*Driver)(nil)
	_ llm.TokenCounter = (*Driver)(nil)
)
