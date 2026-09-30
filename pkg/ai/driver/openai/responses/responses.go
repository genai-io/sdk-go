// Package responses implements the OpenAI Responses protocol.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
//
// # Where things live
//
//	responses.go  construction, Stream and Models
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
	wire "github.com/openai/openai-go/v3/responses"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/oai"
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
	if err := ai.RejectProtocolConfig(cfg, Name); err != nil {
		return nil, err
	}
	return &Driver{
		client: oai.NewClient(cfg),
		model:  cfg.Model,
		compat: ai.CompatOf[ai.OpenAIResponsesCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Stream runs one Responses call.
func (d *Driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		params, err := d.buildParams(req)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}
		stream := d.client.Responses.NewStreaming(ctx, params)
		defer func() { _ = stream.Close() }() // the request is over; a close error changes nothing

		// A call is keyed by its call_id, the one name every event agrees on:
		// argument deltas carry only an output-item ID, which some endpoints
		// re-encode per event, so items map to calls and the finished item's
		// arguments are taken whole. Calls not finished are emitted at the end.
		calls := make(map[string]*ai.ToolCall)
		items := make(map[string]string)
		order := make(map[string]int)
		emitted := make(map[string]bool)
		// A refusal is finished off by an ordinary response.completed, whose
		// stop reason would otherwise overwrite the refusal.
		refused := false

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				if !yield(ai.Delta{Block: ai.TextBlock(event.AsResponseOutputTextDelta().Delta)}, nil) {
					return
				}

			case "response.refusal.delta":
				// A refusal is an answer, not a failure: the text is what the
				// model produced, and the stop reason is what says it declined.
				refused = true
				if !yield(ai.Delta{
					Block:      ai.TextBlock(event.AsResponseRefusalDelta().Delta),
					StopReason: ai.StopRefusal,
				}, nil) {
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
				key := callKey(fn)
				calls[key] = &ai.ToolCall{ID: fn.CallID, Name: fn.Name}
				items[fn.ID] = key
				order[key] = len(order)

			case "response.function_call_arguments.delta":
				delta := event.AsResponseFunctionCallArgumentsDelta()
				if call, ok := calls[items[delta.ItemID]]; ok {
					call.Input += delta.Delta
				}

			case "response.output_item.done":
				item := event.AsResponseOutputItemDone().Item
				switch item.Type {
				case "reasoning":
					if reasoning, ok := extractReasoningItem(item); ok {
						if !yield(ai.Delta{Block: ai.ReasoningBlock(reasoning)}, nil) {
							return
						}
					}
				case "function_call":
					fn := item.AsFunctionCall()
					key := callKey(fn)
					call := calls[key]
					if call == nil {
						call = &ai.ToolCall{ID: fn.CallID, Name: fn.Name}
					}
					if fn.Arguments != "" {
						call.Input = fn.Arguments
					}
					emitted[key] = true
					if !yield(ai.Delta{Block: ai.ToolCallBlock(*call)}, nil) {
						return
					}
				}

			case "response.completed":
				resp := event.AsResponseCompleted().Response
				// A Responses-compatible endpoint that emits only this event
				// says which outcome it means in the status.
				switch resp.Status {
				case wire.ResponseStatusIncomplete:
					if !yield(finished(resp, incompleteStop(resp)), nil) {
						return
					}
				case wire.ResponseStatusFailed:
					yield(ai.Delta{}, d.responseError(string(resp.Error.Code), resp.Error.Message))
					return
				case wire.ResponseStatusCompleted, "":
					if !yield(finished(resp, endOfTurn(refused, len(calls) > 0)), nil) {
						return
					}
				default:
					if !yield(finished(resp, ai.StopReason(resp.Status)), nil) {
						return
					}
				}

			case "response.incomplete":
				// Incomplete and failed are event types of their own, not a
				// status inside response.completed.
				resp := event.AsResponseIncomplete().Response
				if !yield(finished(resp, incompleteStop(resp)), nil) {
					return
				}

			case "response.failed":
				resp := event.AsResponseFailed().Response
				yield(ai.Delta{}, d.responseError(string(resp.Error.Code), resp.Error.Message))
				return

			case "error":
				e := event.AsError()
				yield(ai.Delta{}, d.responseError(e.Code, e.Message))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(ai.Delta{}, fail.WrapStream(err))
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

// endOfTurn says why a response that ran to completion stopped.
func endOfTurn(refused, calledTools bool) ai.StopReason {
	switch {
	case refused:
		return ai.StopRefusal
	case calledTools:
		return ai.StopToolUse
	default:
		return ai.StopEndTurn
	}
}

// incompleteStop says why a response stopped short. A reason this driver does
// not know is taken as the cap, the common case.
func incompleteStop(resp wire.Response) ai.StopReason {
	if resp.IncompleteDetails.Reason == "content_filter" {
		return ai.StopRefusal
	}
	return ai.StopMaxTokens
}

// finished reads the token accounting off a terminal response. All three
// terminal events carry the same Response, so the three of them share this.
func finished(resp wire.Response, stop ai.StopReason) ai.Delta {
	// input_tokens is the whole prompt; the cached and newly cached slices
	// sit under input_tokens_details.
	details := resp.Usage.InputTokensDetails
	fresh, cached := ai.SplitPromptTokens(int(resp.Usage.InputTokens), int(details.CachedTokens))
	fresh, written := ai.SplitPromptTokens(fresh, int(details.CacheWriteTokens))
	return ai.Delta{
		Model:      string(resp.Model),
		ID:         resp.ID,
		StopReason: stop,
		Usage: &ai.Usage{
			Input:  fresh,
			Output: int(resp.Usage.OutputTokens),
			// Reasoning is already inside output_tokens; it travels separately
			// only so a caller can see what the thinking cost.
			Reasoning:  int(resp.Usage.OutputTokensDetails.ReasoningTokens),
			CacheRead:  cached,
			CacheWrite: written,
		},
	}
}

// Models lists the models the endpoint serves, filtered to the ones that can
// answer a Responses request — the listing also carries image, audio,
// embedding and moderation models, which would only be noise in a model
// picker.
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	page, err := d.client.Models.List(ctx)
	if err != nil {
		return nil, fail.Wrap(err)
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
	return !slices.ContainsFunc(nonTextPrefixes, func(p string) bool { return strings.HasPrefix(id, p) }) &&
		!slices.ContainsFunc(nonTextFragments, func(f string) bool { return strings.Contains(id, f) }) &&
		!strings.HasSuffix(id, "-instruct")
}

// callKey names a function call across its events: its call_id, or the item
// ID for an endpoint that leaves call_id out.
func callKey(fn wire.ResponseFunctionToolCall) string {
	if fn.CallID != "" {
		return fn.CallID
	}
	return fn.ID
}

// extractReasoningItem keeps only replayable state. Without encrypted content
// an item cannot restore anything.
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
