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
	"cmp"
	"context"
	"io"
	"iter"
	"slices"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	wire "github.com/openai/openai-go/v3/responses"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/oai"
)

// name is the driver's identifier.
const name = string(ai.APIOpenAIResponses)

func init() { ai.RegisterAPI(ai.APIOpenAIResponses, New) }

// driver talks to one Responses endpoint.
type driver struct {
	client sdk.Client
	model  ai.Model
	compat ai.OpenAIResponsesCompat
}

// New builds a driver from a Config. Registered as the factory for
// ai.APIOpenAIResponses.
func New(cfg ai.Config) (ai.Driver, error) {
	return &driver{
		client: oai.NewClient(cfg),
		model:  cfg.Model,
		compat: ai.CompatOf[ai.OpenAIResponsesCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *driver) Name() string { return name }

// Stream runs one Responses call.
func (d *driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		stream := d.client.Responses.NewStreaming(ctx, d.buildParams(req))
		defer func() { _ = stream.Close() }() // the request is over; a close error changes nothing

		// A call is keyed by its call_id, the one name every event agrees on:
		// argument deltas carry only an output-item ID, which some endpoints
		// re-encode per event, so at maps both to the call and the finished
		// item's arguments are taken whole. Calls not finished are emitted at
		// the end, in the order they began.
		var calls []streamedCall
		at := make(map[string]int)
		track := func(fn wire.ResponseFunctionToolCall) *streamedCall {
			key := callKey(fn)
			i, ok := at[key]
			if !ok {
				i = len(calls)
				at[key] = i
				calls = append(calls, streamedCall{ToolCall: ai.ToolCall{ID: fn.CallID, Name: fn.Name}})
			}
			if fn.ID != "" {
				at[fn.ID] = i
			}
			return &calls[i]
		}
		// A refusal is finished off by an ordinary response.completed, whose
		// stop reason would otherwise overwrite the refusal.
		refused := false
		complete := false
		thought := false
		var terminalErr error

	events:
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
				// "**headline**\n\nbody" with no separator between them, and a
				// later reasoning item's parts count from 0 again. Without a
				// break, two bold headlines collide into "…truncation****Updating…".
				if thought {
					if !yield(ai.Delta{Block: ai.ThinkingBlock("\n\n", "")}, nil) {
						return
					}
				}

			case "response.reasoning_summary_text.delta":
				thought = true
				if !yield(ai.Delta{Block: ai.ThinkingBlock(event.AsResponseReasoningSummaryTextDelta().Delta, "")}, nil) {
					return
				}

			case "response.reasoning_text.delta":
				thought = true
				if !yield(ai.Delta{Block: ai.ThinkingBlock(event.AsResponseReasoningTextDelta().Delta, "")}, nil) {
					return
				}

			case "response.output_item.added":
				item := event.AsResponseOutputItemAdded()
				if item.Item.Type != "function_call" {
					continue
				}
				track(item.Item.AsFunctionCall())

			case "response.function_call_arguments.delta":
				delta := event.AsResponseFunctionCallArgumentsDelta()
				if i, ok := at[delta.ItemID]; ok {
					calls[i].Input += delta.Delta
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
					call := track(fn)
					if fn.Arguments != "" {
						call.Input = fn.Arguments
					}
					call.emitted = true
					if !yield(ai.Delta{Block: ai.ToolCallBlock(call.ToolCall)}, nil) {
						return
					}
				}

			case "response.completed":
				resp := event.AsResponseCompleted().Response
				complete = true
				// A Responses-compatible endpoint that emits only this event
				// says which outcome it means in the status.
				switch resp.Status {
				case wire.ResponseStatusIncomplete:
					if !yield(finalDelta(resp, incompleteStop(resp)), nil) {
						return
					}
				case wire.ResponseStatusFailed:
					if !yield(finalDelta(resp, ai.StopError), nil) {
						return
					}
					terminalErr = d.responseError(string(resp.Error.Code), resp.Error.Message)
					break events
				case wire.ResponseStatusCompleted, "":
					if !yield(finalDelta(resp, endOfTurn(refused, len(calls) > 0)), nil) {
						return
					}
				default:
					if !yield(finalDelta(resp, ai.StopReason(resp.Status)), nil) {
						return
					}
				}

			case "response.incomplete":
				complete = true
				// Incomplete and failed are event types of their own, not a
				// status inside response.completed.
				resp := event.AsResponseIncomplete().Response
				if !yield(finalDelta(resp, incompleteStop(resp)), nil) {
					return
				}

			case "response.failed":
				resp := event.AsResponseFailed().Response
				if !yield(finalDelta(resp, ai.StopError), nil) {
					return
				}
				terminalErr = d.responseError(string(resp.Error.Code), resp.Error.Message)
				break events

			case "error":
				e := event.AsError()
				terminalErr = d.responseError(e.Code, e.Message)
				break events
			}
		}

		for _, call := range calls {
			if call.emitted {
				continue
			}
			if !yield(ai.Delta{Block: ai.ToolCallBlock(call.ToolCall)}, nil) {
				return
			}
		}
		switch {
		case terminalErr != nil:
			yield(ai.Delta{}, terminalErr)
		case stream.Err() != nil:
			yield(ai.Delta{}, fail.WrapStream(stream.Err()))
		case !complete:
			yield(ai.Delta{}, fail.WrapStream(io.ErrUnexpectedEOF))
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

// finalDelta reads the token accounting off a terminal response. All three
// terminal events carry the same Response, so the three of them share this.
func finalDelta(resp wire.Response, stop ai.StopReason) ai.Delta {
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
func (d *driver) Models(ctx context.Context) ([]ai.Model, error) {
	if d.compat.CodexModelListing {
		return d.codexModels(ctx)
	}
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

// codexClientVersion is the Codex CLI release this driver presents itself as
// when listing. The backend returns the lineup for the version it is told;
// bump it if the list goes stale.
const codexClientVersion = "0.161.0"

// codexModels reads the lineup a ChatGPT subscription is entitled to.
func (d *driver) codexModels(ctx context.Context) ([]ai.Model, error) {
	var listing struct {
		Models []struct {
			Slug                     string `json:"slug"`
			DisplayName              string `json:"display_name"`
			ContextWindow            int    `json:"context_window"`
			ShowInPicker             *bool  `json:"show_in_picker"`
			DefaultReasoningLevel    string `json:"default_reasoning_level"`
			SupportedReasoningLevels []struct {
				Effort string `json:"effort"`
			} `json:"supported_reasoning_levels"`
		} `json:"models"`
	}
	err := d.client.Get(ctx, "models", nil, &listing, option.WithQuery("client_version", codexClientVersion))
	if err != nil {
		return nil, fail.Wrap(err)
	}
	out := make([]ai.Model, 0, len(listing.Models))
	for _, m := range listing.Models {
		// A null show_in_picker means shown; only an explicit false hides one.
		if m.Slug == "" || (m.ShowInPicker != nil && !*m.ShowInPicker) {
			continue
		}
		model := ai.Model{
			ID:            m.Slug,
			Name:          cmp.Or(m.DisplayName, m.Slug),
			API:           ai.APIOpenAIResponses,
			Vendor:        d.model.Vendor,
			ContextWindow: m.ContextWindow,
		}
		for _, level := range m.SupportedReasoningLevels {
			if level.Effort == "" {
				continue
			}
			model.Reasoning = append(model.Reasoning, ai.ReasoningLevel{
				Effort:  ai.Effort(level.Effort),
				Value:   level.Effort,
				Default: level.Effort == m.DefaultReasoningLevel,
			})
		}
		out = append(out, model)
	}
	if len(out) == 0 {
		return nil, &ai.Error{Driver: name, Kind: ai.KindUnknown, Message: "the subscription catalog listed no models"}
	}
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

// streamedCall is one function call as its events arrive.
type streamedCall struct {
	ai.ToolCall
	emitted bool
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
		API: ai.APIOpenAIResponses,
		ID:  r.ID, EncryptedContent: r.EncryptedContent, Summary: summary.String(),
	}, true
}

var (
	_ ai.Driver      = (*driver)(nil)
	_ ai.ModelLister = (*driver)(nil)
)
