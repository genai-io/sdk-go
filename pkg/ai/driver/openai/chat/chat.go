// Package chat implements the OpenAI Chat Completions protocol.
//
// Chat Completions is the industry's interchange format, so this one driver
// serves most of the catalog: DeepSeek, Moonshot, Alibaba DashScope, Z.ai,
// SenseNova, Agnes-AI, GitHub Copilot and a local Ollama all speak it. What
// separates them is a base URL and a reasoning dialect, both of which arrive
// as catalog data rather than code. Import it for its side effect:
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
//
// # Where things live
//
//	chat.go     construction, Generate and Models
//	request.go  an ai.Request translated into Chat params
//
// Failures are classified by driver/internal/openaierr, shared with the
// Responses driver: this protocol adds nothing of its own to that.
package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"slices"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/errs"
)

// Name is the driver's identifier.
const Name = string(ai.APIOpenAIChat)

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
	if err := ai.RejectConfigNative(cfg, Name); err != nil {
		return nil, err
	}

	opts := []option.RequestOption{option.WithMaxRetries(0)}
	if url := cfg.URL(); url != "" {
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
	for k, v := range cfg.MergedHeaders() {
		opts = append(opts, option.WithHeader(k, v))
	}

	return &Driver{
		client: sdk.NewClient(opts...),
		model:  cfg.Model,
		compat: ai.CompatOf[ai.OpenAIChatCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Generate runs one Chat Completions call.
func (d *Driver) Generate(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		if err := ai.RejectNative(req, Name); err != nil {
			yield(ai.Delta{}, err)
			return
		}
		level, _ := d.model.ResolveLevel(req.Effort)
		params := d.buildParams(req, level)

		stream := d.client.Chat.Completions.NewStreaming(ctx, params)
		defer stream.Close()

		// Tool calls arrive as indexed argument fragments spread across
		// chunks, so they can only be emitted once the stream ends.
		calls := make(map[int]*ai.ToolCall)
		reasoning := d.compat.Thinking != ai.ThinkingNone && thinkingOn(level)

		for stream.Next() {
			chunk := stream.Current()

			for _, choice := range chunk.Choices {
				if reasoning {
					if text := reasoningContent(choice.Delta.RawJSON()); text != "" {
						if !yield(ai.Delta{Block: ai.ThinkingBlock(text, "")}, nil) {
							return
						}
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

		if err := stream.Err(); err != nil {
			yield(ai.Delta{}, errs.WrapStream(Name, err))
			return
		}

		for _, idx := range slices.Sorted(maps.Keys(calls)) {
			if !yield(ai.Delta{Block: ai.ToolCallBlock(*calls[idx])}, nil) {
				return
			}
		}
	}
}

// Models lists what the endpoint serves. Most OpenAI-compatible endpoints
// return the bare shape — id, object, owned_by — with no limits; where one
// includes context_length it is read out of the raw JSON, since the typed SDK
// struct has no field for a non-standard extension.
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	page, err := d.client.Models.List(ctx)
	if err != nil {
		return nil, errs.Wrap(Name, err)
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
	case "":
		return ""
	default:
		return ai.StopReason(reason)
	}
}

var (
	_ ai.Driver      = (*Driver)(nil)
	_ ai.ModelLister = (*Driver)(nil)
)
