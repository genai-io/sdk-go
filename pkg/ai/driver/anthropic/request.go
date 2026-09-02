package anthropic

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// validToolID matches the tool_use ID shape the Messages API accepts.
var validToolID = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// toolIDs rewrites tool call IDs another provider produced into ones Anthropic
// accepts, keeping the mapping stable within a request so calls and results
// still line up. A conversation that started on a different provider and was
// handed to Claude is otherwise rejected outright.
type toolIDs struct {
	m map[string]string
	n int

	// unnamed holds the IDs minted for calls that arrived without one, and
	// answered counts how many results have claimed one. Gemini leaves a
	// function call's ID empty, so there is nothing to key a map on and two
	// parallel calls would otherwise become the same Anthropic ID; matching by
	// position instead pairs the nth unnamed result with the nth unnamed call,
	// which is the order both arrive in.
	unnamed  []string
	answered int
}

// call resolves the ID of a tool call the model made.
func (t *toolIDs) call(id string) string {
	if id == "" {
		mapped := t.mint()
		t.unnamed = append(t.unnamed, mapped)
		return mapped
	}
	return t.named(id)
}

// result resolves the ID of a tool result answering one of those calls.
func (t *toolIDs) result(id string) string {
	if id == "" {
		if t.answered < len(t.unnamed) {
			mapped := t.unnamed[t.answered]
			t.answered++
			return mapped
		}
		// A result with no call in front of it. It will be rejected either
		// way; a fresh ID at least says which one went missing.
		return t.mint()
	}
	return t.named(id)
}

func (t *toolIDs) named(id string) string {
	if validToolID.MatchString(id) {
		return id
	}
	if t.m == nil {
		t.m = make(map[string]string)
	}
	if mapped, ok := t.m[id]; ok {
		return mapped
	}
	mapped := t.mint()
	t.m[id] = mapped
	return mapped
}

func (t *toolIDs) mint() string {
	t.n++
	return fmt.Sprintf("toolu_compat_%d", t.n)
}

func (d *Driver) buildParams(req *ai.Request, native Options) (*sdk.MessageNewParams, error) {
	var ids toolIDs

	// The rung carries both halves of the mapping: Value is what
	// output_config.effort wants under adaptive thinking, Budget is what
	// budget_tokens wants under the older shape. Neither is computed here —
	// which spelling this endpoint uses is model data.
	level, hasLevel := d.model.ResolveLevel(req.Effort)
	var budget int64
	if !d.compat.ForceAdaptiveThinking {
		budget = int64(level.Budget)
	}
	// Whether reasoning runs at all, under either shape. It gates replaying a
	// prior thinking block: the API rejects one sent into a request that is
	// not itself thinking.
	thinkingOn := budget > 0 || (d.compat.ForceAdaptiveThinking && level.Value != "")

	msgs := make([]sdk.MessageParam, 0, len(req.Messages))
	for _, msg := range req.Messages {
		blocks := messageBlocks(msg, thinkingOn, &ids)
		if len(blocks) == 0 {
			continue
		}
		switch msg.Role {
		case ai.RoleUser:
			msgs = append(msgs, sdk.NewUserMessage(blocks...))
		case ai.RoleAssistant:
			msgs = append(msgs, sdk.NewAssistantMessage(blocks...))
		}
	}
	msgs = mergeConsecutive(msgs)

	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	// budget_tokens must leave room for an answer, so max_tokens has to exceed
	// it. Raising the cap is the only fix that keeps the requested budget.
	if budget > 0 && maxTokens <= budget {
		maxTokens = budget + defaultMaxTokens
	}

	params := &sdk.MessageNewParams{
		Model:     sdk.Model(d.model.ID),
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
	if req.Temperature != nil && !d.compat.NoTemperature {
		params.Temperature = sdk.Float(*req.Temperature)
	}
	if len(req.StopSequences) > 0 {
		params.StopSequences = req.StopSequences
	}
	if choice := toolChoice(req, native); choice != nil {
		params.ToolChoice = *choice
	}
	if req.System != "" {
		block := sdk.TextBlockParam{Text: req.System}
		// One cache breakpoint at the end of the system block. Anthropic
		// renders a request as tools → system → messages, so the cached prefix
		// is exactly the tool definitions plus the system prompt — which makes
		// the reported cache tokens an exact measurement of those two. Moving
		// or adding a breakpoint invalidates that reading.
		if cc := d.cacheControl(req.CacheRetention); cc != nil {
			block.CacheControl = *cc
		}
		params.System = []sdk.TextBlockParam{block}
	}
	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
	}
	// Structured output shares output_config with the effort level, so it is
	// set on the same struct rather than replacing it.
	if req.Schema != nil {
		if def := req.Schema.DefinitionMap(); def != nil {
			params.OutputConfig.Format = sdk.JSONOutputFormatParam{Schema: def}
		}
	}
	return params, nil
}

// Options carries the Anthropic-only settings the normalized Options do not
// model. Pass it with ai.WithProtocolOptions; the zero value changes nothing.
type Options struct {
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

// ProtocolOptions marks this as one driver's request settings.
func (Options) ProtocolOptions() {}

// cacheControl maps the requested retention onto a cache_control marker. A nil
// result means no breakpoint, which is what CacheNone asks for and what an
// endpoint with no prompt cache gets.
func (d *Driver) cacheControl(retention ai.CacheRetention) *sdk.CacheControlEphemeralParam {
	if d.compat.NoPromptCache || retention == ai.CacheNone {
		return nil
	}
	cc := sdk.NewCacheControlEphemeralParam()
	// A 1-hour write costs twice the input rate against 1.25x for five
	// minutes, so it only pays off from the second read — the caller asks for
	// it deliberately, and an endpoint that rejects the field falls back
	// rather than failing.
	if retention == ai.CacheLong && !d.compat.NoLongCacheRetention {
		cc.TTL = sdk.CacheControlEphemeralTTLTTL1h
	}
	return &cc
}

// toolChoice maps the neutral constraint onto Anthropic's union. A nil result
// leaves the field off, which is the API's own default.
func toolChoice(req *ai.Request, native Options) *sdk.ToolChoiceUnionParam {
	// disable_parallel_tool_use is sent only when it is on. Its zero value
	// changes nothing, as the option documents, and an Anthropic-compatible
	// host may reject a property it has never heard of.
	switch name, forced := req.ToolChoice.Tool(); {
	case forced:
		choice := &sdk.ToolChoiceToolParam{Name: name}
		if native.DisableParallelToolUse {
			choice.DisableParallelToolUse = sdk.Bool(true)
		}
		return &sdk.ToolChoiceUnionParam{OfTool: choice}
	case req.ToolChoice == ai.ToolChoiceNone:
		return &sdk.ToolChoiceUnionParam{OfNone: &sdk.ToolChoiceNoneParam{}}
	case req.ToolChoice == ai.ToolChoiceRequired:
		choice := &sdk.ToolChoiceAnyParam{}
		if native.DisableParallelToolUse {
			choice.DisableParallelToolUse = sdk.Bool(true)
		}
		return &sdk.ToolChoiceUnionParam{OfAny: choice}
	}
	return nil
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

// messageBlocks maps the canonical sequence without reordering it. A thinking
// block — readable or redacted — is replayed only when thinking is enabled for
// this request, and a readable one only when signed; Anthropic rejects both
// unsigned and disabled-thinking replay.
func messageBlocks(msg ai.Message, thinkingOn bool, ids *toolIDs) []sdk.ContentBlockParamUnion {
	blocks := make([]sdk.ContentBlockParamUnion, 0, len(msg.Content))
	for _, block := range msg.Content {
		switch block.Type {
		case ai.BlockText:
			if block.Text != "" {
				blocks = append(blocks, sdk.NewTextBlock(block.Text))
			}
		case ai.BlockImage:
			if block.Image != nil {
				blocks = append(blocks, sdk.NewImageBlockBase64(block.Image.MediaType, block.Image.Data))
			}
		case ai.BlockThinking:
			if block.Text != "" && block.Signature != "" && thinkingOn {
				blocks = append(blocks, sdk.NewThinkingBlock(block.Signature, block.Text))
			}
		case ai.BlockReasoning:
			// Redacted thinking, which the model produced but the safety
			// classifier withheld. There is nothing to read, and it has to go
			// back exactly as it arrived or a tool-use continuation is
			// rejected.
			if block.Reasoning != nil && block.Reasoning.EncryptedContent != "" && thinkingOn {
				blocks = append(blocks, sdk.NewRedactedThinkingBlock(block.Reasoning.EncryptedContent))
			}
		case ai.BlockToolCall:
			if block.ToolCall != nil {
				tc := block.ToolCall
				blocks = append(blocks, sdk.NewToolUseBlock(ids.call(tc.ID), toolInputValue(tc.Input), tc.Name))
			}
		case ai.BlockToolResult:
			if block.ToolResult != nil {
				r := block.ToolResult
				blocks = append(blocks, sdk.NewToolResultBlock(ids.result(r.ToolCallID), r.Content, r.IsError))
			}
		}
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

func convertTools(tools []ai.Tool) []sdk.ToolUnionParam {
	out := make([]sdk.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := sdk.ToolInputSchemaParam{}
		if props := t.Schema.DefinitionMap(); props != nil {
			if properties, ok := props["properties"]; ok {
				schema.Properties = properties
			}
			schema.Required = stringSlice(props["required"])
			schema.ExtraFields = schemaExtras(props)
		}
		out = append(out, sdk.ToolUnionParam{
			OfTool: &sdk.ToolParam{
				Name:        t.Schema.Name,
				Description: sdk.String(t.Schema.Description),
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
