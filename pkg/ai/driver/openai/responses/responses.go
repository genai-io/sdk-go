// Package responses implements the OpenAI Responses protocol.
//
// Responses is OpenAI's current API and the only one that round-trips a
// reasoning model's internal state, so it is what the openai vendor uses. The
// older Chat Completions protocol is the sibling package chat, and is what the
// rest of the industry implements.
//
// The vendor SDK's own responses package is imported here as wire, since this
// package carries that name: those identifiers are the protocol's wire types.
//
// Import this one for its side effect:
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
//
// # Where things live
//
//	responses.go  construction, Generate and Models
//	request.go    an ai.Request translated into Responses params
//	errors.go     this protocol's failures, including the ones that arrive
//	              inside a 200
package responses

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	wire "github.com/openai/openai-go/v3/responses"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Name is the driver's identifier.
const Name = string(ai.APIOpenAIResponses)

func init() { ai.RegisterAPI(ai.APIOpenAIResponses, New) }

// Driver talks to one Responses endpoint.
type Driver struct {
	client sdk.Client
	model  ai.Model
	compat ai.OpenAIResponsesCompat
}

// New builds a driver from a Config. Registered as the factory for
// ai.APIOpenAIResponses.
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
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	for k, v := range cfg.MergedHeaders() {
		opts = append(opts, option.WithHeader(k, v))
	}
	return &Driver{
		client: sdk.NewClient(opts...),
		model:  cfg.Model,
		compat: ai.CompatOf[ai.OpenAIResponsesCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Generate runs one Responses call.
func (d *Driver) Generate(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		params, err := d.buildParams(req)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}
		stream := d.client.Responses.NewStreaming(ctx, params)
		defer stream.Close()

		// Responses identifies a function call by output-item ID while its
		// arguments stream, so calls are collected and emitted at the end.
		calls := make(map[string]*ai.ToolCall)
		order := make(map[string]int)
		emitted := make(map[string]bool)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				if !yield(ai.Delta{Block: ai.TextBlock(event.AsResponseOutputTextDelta().Delta)}, nil) {
					return
				}

			case "response.reasoning_summary_part.added":
				// The summary streams as discrete parts, each a self-contained
				// "**headline**\n\nbody" with no separator between them.
				// Without a break, two adjacent bold headlines collide into
				// "…truncation****Updating…".
				if event.AsResponseReasoningSummaryPartAdded().SummaryIndex > 0 {
					if !yield(ai.Delta{Block: ai.ThinkingBlock("\n\n", "")}, nil) {
						return
					}
				}

			case "response.reasoning_summary_text.delta":
				if !yield(ai.Delta{Block: ai.ThinkingBlock(event.AsResponseReasoningSummaryTextDelta().Delta, "")}, nil) {
					return
				}

			case "response.reasoning_text.delta":
				if !yield(ai.Delta{Block: ai.ThinkingBlock(event.AsResponseReasoningTextDelta().Delta, "")}, nil) {
					return
				}

			case "response.output_item.added":
				item := event.AsResponseOutputItemAdded()
				if item.Item.Type != "function_call" {
					continue
				}
				fn := item.Item.AsFunctionCall()
				calls[fn.ID] = &ai.ToolCall{ID: fn.CallID, Name: fn.Name}
				order[fn.ID] = len(order)

			case "response.function_call_arguments.delta":
				delta := event.AsResponseFunctionCallArgumentsDelta()
				if call, ok := calls[delta.ItemID]; ok {
					call.Input += delta.Delta
				}

			case "response.output_item.done":
				item := event.AsResponseOutputItemDone().Item
				switch item.Type {
				case "reasoning":
					if !d.compat.Stateless {
						continue
					}
					if reasoning, ok := extractReasoningItem(item); ok {
						if !yield(ai.Delta{Block: ai.ReasoningBlock(reasoning)}, nil) {
							return
						}
					}
				case "function_call":
					fn := item.AsFunctionCall()
					call := calls[fn.ID]
					if call == nil {
						call = &ai.ToolCall{ID: fn.CallID, Name: fn.Name, Input: fn.Arguments}
					}
					emitted[fn.ID] = true
					if !yield(ai.Delta{Block: ai.ToolCallBlock(*call)}, nil) {
						return
					}
				}

			case "response.completed":
				resp := event.AsResponseCompleted().Response
				out := ai.Delta{Model: resp.Model, ID: resp.ID}

				// input_tokens is the whole prompt; the cached slice sits
				// under input_tokens_details.
				fresh, cached := ai.SplitPromptTokens(
					int(resp.Usage.InputTokens),
					int(resp.Usage.InputTokensDetails.CachedTokens),
				)
				out.Usage = &ai.Usage{
					Input:     fresh,
					Output:    int(resp.Usage.OutputTokens),
					CacheRead: cached,
				}

				switch resp.Status {
				case wire.ResponseStatusCompleted:
					if len(calls) > 0 {
						out.StopReason = ai.StopToolUse
					} else {
						out.StopReason = ai.StopEndTurn
					}
				case wire.ResponseStatusIncomplete:
					out.StopReason = ai.StopMaxTokens
				case wire.ResponseStatusFailed:
					yield(ai.Delta{}, d.responseError(string(resp.Error.Code), resp.Error.Message))
					return
				default:
					out.StopReason = ai.StopReason(resp.Status)
				}

				if !yield(out, nil) {
					return
				}

			case "error":
				e := event.AsError()
				yield(ai.Delta{}, d.responseError(e.Code, e.Message))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(ai.Delta{}, wrapStream(err))
			return
		}

		for _, id := range slices.SortedFunc(maps.Keys(calls), func(a, b string) int { return order[a] - order[b] }) {
			if emitted[id] {
				continue
			}
			if !yield(ai.Delta{Block: ai.ToolCallBlock(*calls[id])}, nil) {
				return
			}
		}
	}
}

// Models lists the models the endpoint serves, filtered to the ones that can
// answer a Responses request — the listing also carries image, audio,
// embedding and moderation models, which would only be noise in a model
// picker.
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	page, err := d.client.Models.List(ctx)
	if err != nil {
		return nil, wrap(err)
	}
	out := make([]ai.Model, 0, len(page.Data))
	for _, m := range page.Data {
		if !isTextModel(m.ID) {
			continue
		}
		out = append(out, ai.Model{ID: m.ID, Name: m.ID, API: ai.APIOpenAIResponses, Vendor: d.model.Vendor})
	}
	slices.SortFunc(out, func(a, b ai.Model) int { return strings.Compare(a.ID, b.ID) })
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

// extractReasoningItem keeps only replayable state. Without encrypted content
// an item cannot restore anything on a stateless backend.
func extractReasoningItem(item wire.ResponseOutputItemUnion) (ai.ReasoningItem, bool) {
	r := item.AsReasoning()
	if r.EncryptedContent == "" {
		return ai.ReasoningItem{}, false
	}
	var summary strings.Builder
	for i, part := range r.Summary {
		if i > 0 {
			summary.WriteString("\n\n")
		}
		summary.WriteString(part.Text)
	}
	return ai.ReasoningItem{
		ID: r.ID, EncryptedContent: r.EncryptedContent, Summary: summary.String(),
	}, true
}

var (
	_ ai.Driver      = (*Driver)(nil)
	_ ai.ModelLister = (*Driver)(nil)
)
