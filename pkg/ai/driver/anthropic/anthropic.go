// Package anthropic implements the Anthropic Messages protocol.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
//
// # Where things live
//
//	anthropic.go  construction, Stream and Models
//	request.go    an ai.Request translated into Messages params
//	errors.go     this protocol's failures classified into ai.Error kinds
//	vertex/       the same protocol served through Google Cloud Vertex AI,
//	              kept a separate package so its heavy auth stack is opt-in
package anthropic

import (
	"cmp"
	"context"
	"io"
	"iter"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
)

// defaultMaxTokens is what the protocol requires and the caller did not
// supply. Anthropic rejects a request without max_tokens, so unlike the
// OpenAI protocols there is no "leave it to the server" option.
const defaultMaxTokens = 8192

func init() { ai.RegisterAPI(ai.APIAnthropicMessages, New) }

// driver talks to one Anthropic-protocol endpoint.
type driver struct {
	client sdk.Client
	model  ai.Model
	compat ai.AnthropicCompat
	// api is the protocol as the caller reached it. Vertex serves this same
	// wire format from Google's cloud, and says so.
	api  ai.API
	fail errs.Classifier
}

// New builds a driver from a Config. It is registered as the factory for
// ai.APIAnthropicMessages, so ai.New reaches it without an explicit call.
func New(cfg ai.Config) (ai.Driver, error) {
	return NewDeployment(cfg, ai.APIAnthropicMessages)
}

// NewDeployment builds a driver for a deployment that reaches this protocol
// some other way. api is the protocol as the caller reaches it: Vertex speaks
// this wire format from a different host under different credentials, and an
// error or a model listing labelled with the direct API would send a caller to
// the wrong console. opts — the Vertex auth option is the only one — are
// applied ahead of the Config's own, so a Config's endpoint still wins over
// the host such an option carries with it.
//
// The SDK client is assembled here rather than by sdk.NewClient, which
// prepends DefaultClientOptions — and that reads ANTHROPIC_API_KEY,
// ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL out of the process environment.
// A Config is meant to be the whole truth about where a request goes and what
// it presents, so none of them is ever read.
func NewDeployment(cfg ai.Config, api ai.API, opts ...option.RequestOption) (ai.Driver, error) {
	opts = append([]option.RequestOption{
		// The SDK's own host, which sdk.NewClient would otherwise supply.
		option.WithEnvironmentProduction(),
		// Retries belong to the caller, which alone knows the budget for the
		// whole turn. An SDK retrying underneath would multiply it silently.
		option.WithMaxRetries(0),
	}, opts...)
	if url := cfg.MergedBaseURL(); url != "" {
		opts = append(opts, option.WithBaseURL(url))
	}
	if cfg.APIKey != "" {
		// Ark and other re-hosts take the key as a bearer token; Anthropic
		// itself wants x-api-key.
		if ai.CompatOf[ai.AnthropicCompat](cfg.Model).BearerAuth {
			opts = append(opts, option.WithAuthToken(cfg.APIKey))
		} else {
			opts = append(opts, option.WithAPIKey(cfg.APIKey))
		}
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	for k, v := range cfg.MergedHeaders() {
		opts = append(opts, option.WithHeader(k, v))
	}

	// Only the two services the driver uses are wired.
	client := sdk.Client{Options: opts}
	client.Messages = sdk.NewMessageService(opts...)
	client.Models = sdk.NewModelService(opts...)

	return &driver{
		client: client,
		model:  cfg.Model,
		compat: ai.CompatOf[ai.AnthropicCompat](cfg.Model),
		api:    api,
		fail:   errs.For(string(api), details),
	}, nil
}

// Name identifies the driver: the protocol it speaks, from the deployment it
// speaks it to.
func (d *driver) Name() string { return string(d.api) }

// Stream runs one Messages call.
func (d *driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		params, err := d.buildParams(req)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}

		stream := d.client.Messages.NewStreaming(ctx, *params)
		defer func() { _ = stream.Close() }() // the request is over; a close error changes nothing

		// Tool input is kept per content-block index: a compatible endpoint
		// may interleave two calls' fragments, which Anthropic itself never does.
		tools := map[int64]*toolBlock{}
		complete := false

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "message_start":
				start := event.AsMessageStart()
				if !yield(ai.Delta{
					Model: string(start.Message.Model),
					ID:    start.Message.ID,
					Usage: &ai.Usage{
						Input:      int(start.Message.Usage.InputTokens),
						CacheWrite: int(start.Message.Usage.CacheCreationInputTokens),
						// The 1-hour slice is billed at twice the input rate,
						// so it has to travel separately from the total.
						CacheWriteLong: int(start.Message.Usage.CacheCreation.Ephemeral1hInputTokens),
						CacheRead:      int(start.Message.Usage.CacheReadInputTokens),
					},
				}, nil) {
					return
				}

			case "content_block_start":
				block := event.AsContentBlockStart()
				switch block.ContentBlock.Type {
				case "tool_use":
					tools[block.Index] = &toolBlock{id: block.ContentBlock.ID, name: block.ContentBlock.Name}
				case "redacted_thinking":
					// Thinking the safety classifier withheld: unreadable, but
					// the API rejects a tool-use turn whose history drops it.
					if !yield(ai.Delta{Block: ai.ReasoningBlock(ai.ReasoningItem{
						API:              d.api,
						EncryptedContent: block.ContentBlock.Data,
					})}, nil) {
						return
					}
				}

			case "content_block_delta":
				delta := event.AsContentBlockDelta()
				var out ai.Delta
				switch delta.Delta.Type {
				case "text_delta":
					out.Block = ai.TextBlock(delta.Delta.Text)
				case "thinking_delta":
					out.Block = ai.ThinkingBlock(delta.Delta.Thinking, "")
				case "signature_delta":
					out.Block = ai.ThinkingBlock("", delta.Delta.Signature)
				case "input_json_delta":
					if tool := tools[delta.Index]; tool != nil {
						tool.input.WriteString(delta.Delta.PartialJSON)
					}
					continue
				default:
					continue
				}
				if !yield(out, nil) {
					return
				}

			case "content_block_stop":
				index := event.AsContentBlockStop().Index
				tool := tools[index]
				delete(tools, index)
				if tool == nil || tool.id == "" || tool.name == "" {
					// A text or thinking block ended. Anthropic is one of the
					// few protocols that says so, which is what lets two
					// adjacent blocks of the same kind be told apart.
					if !yield(ai.Delta{EndBlock: true}, nil) {
						return
					}
					continue
				}
				call := ai.ToolCall{ID: tool.id, Name: tool.name, Input: tool.input.String()}
				if !yield(ai.Delta{Block: ai.ToolCallBlock(call)}, nil) {
					return
				}

			case "message_delta":
				md := event.AsMessageDelta()
				complete = complete || md.Delta.StopReason != ""
				// Every figure here is the running total for the call. Input
				// tokens repeat what message_start already sent, but some
				// compatible endpoints (SenseNova) send them only here.
				if !yield(ai.Delta{
					StopReason: mapStopReason(string(md.Delta.StopReason)),
					Usage: &ai.Usage{
						Input:      int(md.Usage.InputTokens),
						Output:     int(md.Usage.OutputTokens),
						CacheWrite: int(md.Usage.CacheCreationInputTokens),
						CacheRead:  int(md.Usage.CacheReadInputTokens),
					},
				}, nil) {
					return
				}
			case "message_stop":
				complete = true
			}
		}

		if err := stream.Err(); err != nil {
			yield(ai.Delta{}, d.fail.WrapStream(err))
			return
		}
		if !complete {
			yield(ai.Delta{}, d.fail.WrapStream(io.ErrUnexpectedEOF))
		}
	}
}

// toolBlock is one tool_use block whose input is still streaming.
type toolBlock struct {
	id, name string
	input    strings.Builder
}

// mapStopReason translates Anthropic's stop reasons.
func mapStopReason(reason string) ai.StopReason {
	switch reason {
	case "end_turn":
		return ai.StopEndTurn
	case "tool_use":
		return ai.StopToolUse
	case "max_tokens":
		return ai.StopMaxTokens
	case "stop_sequence":
		return ai.StopSequence
	case "refusal":
		return ai.StopRefusal
	default:
		return ai.StopReason(reason)
	}
}

// Models lists the models the endpoint serves. Anthropic's listing carries IDs
// and display names only — no limits — so callers wanting context windows
// should merge catalog data over the result (see endpoint.Endpoint, which
// does exactly that around a refresh).
func (d *driver) Models(ctx context.Context) ([]ai.Model, error) {
	pager := d.client.Models.ListAutoPaging(ctx, sdk.ModelListParams{})
	var out []ai.Model
	for pager.Next() {
		m := pager.Current()
		name := cmp.Or(m.DisplayName, m.ID)
		out = append(out, ai.Model{
			ID:     m.ID,
			Name:   name,
			API:    d.api,
			Vendor: d.model.Vendor,
		})
	}
	if err := pager.Err(); err != nil {
		return nil, d.fail.Wrap(err)
	}
	return out, nil
}

var (
	_ ai.Driver      = (*driver)(nil)
	_ ai.ModelLister = (*driver)(nil)
)
