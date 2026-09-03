// Package chat implements the OpenAI Chat Completions protocol.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
//
// # Where things live
//
//	chat.go     construction, Stream and Models
//	request.go  an ai.Request translated into Chat params
package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"slices"

	sdk "github.com/openai/openai-go/v3"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/oai"
)

// Name is the driver's identifier.
const Name = string(ai.APIOpenAIChat)

// fail classifies this protocol's failures.
var fail = errs.For(Name, oai.Details)

func init() { ai.RegisterAPI(ai.APIOpenAIChat, New) }

// Driver talks to one Chat Completions endpoint.
type Driver struct {
	client sdk.Client
	model  ai.Model
	compat ai.OpenAIChatCompat
}

// New builds a driver from a Config. Registered as the factory for
// ai.APIOpenAIChat.
func New(cfg ai.Config) (ai.Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}
	if err := ai.RejectProtocolConfig(cfg, Name); err != nil {
		return nil, err
	}

	return &Driver{
		client: oai.NewClient(cfg),
		model:  cfg.Model,
		compat: ai.CompatOf[ai.OpenAIChatCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Stream runs one Chat Completions call.
func (d *Driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		if err := ai.RejectProtocolOptions(req, Name); err != nil {
			yield(ai.Delta{}, err)
			return
		}
		level, _ := d.model.ResolveLevel(req.Effort)
		params := d.buildParams(req, level)

		stream := d.client.Chat.Completions.NewStreaming(ctx, params, oai.RequestOptions(req)...)
		defer func() { _ = stream.Close() }() // the request is over; a close error changes nothing

		// Tool calls arrive as indexed argument fragments spread across
		// chunks, so they can only be emitted once the stream ends.
		calls := make(map[int]*ai.ToolCall)

		for stream.Next() {
			chunk := stream.Current()

			for _, choice := range chunk.Choices {
				// Reasoning is read whatever the request asked for: an endpoint
				// may reason with no switch to declare — a local model, or a
				// gateway deciding for itself.
				if text := reasoningText(choice.Delta.RawJSON()); text != "" {
					if !yield(ai.Delta{Block: ai.ThinkingBlock(text, "")}, nil) {
						return
					}
				}
				if choice.Delta.Content != "" {
					if !yield(ai.Delta{Block: ai.TextBlock(choice.Delta.Content)}, nil) {
						return
					}
				}
				var out ai.Delta
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
						calls[idx] = &ai.ToolCall{ID: tc.ID, Name: tc.Function.Name}
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

				if out.StopReason != "" || out.Model != "" || out.ID != "" {
					if !yield(out, nil) {
						return
					}
				}
			}

			// prompt_tokens is the whole prompt; the cached slice sits under
			// prompt_tokens_details. Split it so Usage.Input stays "fresh
			// tokens only" across every protocol.
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				fresh, cached := ai.SplitPromptTokens(
					int(chunk.Usage.PromptTokens),
					int(chunk.Usage.PromptTokensDetails.CachedTokens),
				)
				if !yield(ai.Delta{Usage: &ai.Usage{
					Input:     fresh,
					Output:    int(chunk.Usage.CompletionTokens),
					CacheRead: cached,
				}}, nil) {
					return
				}
			}
		}

		flush := func() bool {
			for _, idx := range slices.Sorted(maps.Keys(calls)) {
				if !yield(ai.Delta{Block: ai.ToolCallBlock(*calls[idx])}, nil) {
					return false
				}
			}
			return true
		}

		if err := stream.Err(); err != nil {
			// Everything produced stays on the Response, so the calls collected
			// before the cut go out ahead of the failure that ended the stream.
			if flush() {
				yield(ai.Delta{}, fail.WrapStream(err))
			}
			return
		}
		flush()
	}
}

// Models lists what the endpoint serves. Most OpenAI-compatible endpoints
// return the bare shape — id, object, owned_by — with no limits; where one
// includes context_length it is read out of the raw JSON, since the typed SDK
// struct has no field for a non-standard extension.
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	// One request is the whole listing: the SDK's page type for /models reports
	// no next page ever, because the endpoint does not paginate.
	page, err := d.client.Models.List(ctx)
	if err != nil {
		return nil, fail.Wrap(err)
	}
	out := make([]ai.Model, 0, len(page.Data))
	for _, m := range page.Data {
		model := ai.Model{ID: m.ID, Name: m.ID, API: ai.APIOpenAIChat, Vendor: d.model.Vendor}
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

// reasoningText reads the reasoning a stream delta carries. Neither spelling is
// in the standard schema, so the typed SDK struct has no field for either:
// Moonshot, DeepSeek, Alibaba and Z.ai stream reasoning_content, OpenRouter and
// Ollama stream reasoning, and an endpoint sending both sends the same words.
func reasoningText(rawJSON string) string {
	if rawJSON == "" {
		return ""
	}
	var delta struct {
		ReasoningContent string `json:"reasoning_content"`
		// Reasoning is decoded lazily because some gateways put an object
		// there; only the string form is text to show.
		Reasoning json.RawMessage `json:"reasoning"`
	}
	if err := json.Unmarshal([]byte(rawJSON), &delta); err != nil {
		return ""
	}
	if delta.ReasoningContent != "" {
		return delta.ReasoningContent
	}
	var text string
	if json.Unmarshal(delta.Reasoning, &text) != nil {
		return ""
	}
	return text
}

func mapFinishReason(reason string) ai.StopReason {
	switch reason {
	case "stop":
		return ai.StopEndTurn
	case "tool_calls", "function_call":
		return ai.StopToolUse
	case "length":
		return ai.StopMaxTokens
	case "content_filter":
		return ai.StopRefusal
	default:
		return ai.StopReason(reason)
	}
}

var (
	_ ai.Driver      = (*Driver)(nil)
	_ ai.ModelLister = (*Driver)(nil)
)
