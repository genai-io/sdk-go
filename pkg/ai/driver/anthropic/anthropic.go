// Package anthropic implements the Anthropic Messages protocol.
//
// It backs every vendor whose endpoint speaks that protocol — Anthropic
// itself, MiniMax, Xiaomi MiMo and Volcengine Ark — which differ only in base
// URL, credential and how the key is presented. Import it for its side effect
// to make ai.Open handle ai.APIAnthropicMessages:
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
//
// # Where things live
//
//	anthropic.go  construction, Stream, Models and CountTokens
//	request.go    an ai.Request translated into Messages params
//	errors.go     this protocol's failures classified into ai.Error kinds
//	vertex/       the same protocol served through Google Cloud Vertex AI,
//	              kept a separate package so its heavy auth stack is opt-in
package anthropic

import (
	"context"
	"fmt"
	"iter"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Name is the driver's identifier, matching the protocol it speaks.
const Name = string(ai.APIAnthropicMessages)

// defaultMaxTokens is what the protocol requires and the caller did not
// supply. Anthropic rejects a request without max_tokens, so unlike the
// OpenAI protocols there is no "leave it to the server" option.
const defaultMaxTokens = 8192

func init() { ai.RegisterAPI(ai.APIAnthropicMessages, New) }

// Driver talks to one Anthropic-protocol endpoint.
type Driver struct {
	client sdk.Client
	model  ai.Model
	compat ai.AnthropicCompat
}

// New builds a driver from a Config. It is registered as the factory for
// ai.APIAnthropicMessages, so ai.Open reaches it without an explicit call.
func New(cfg ai.Config) (ai.Driver, error) {
	if err := ai.RejectProtocolConfig(cfg, Name); err != nil {
		return nil, err
	}
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
func ClientOptions(cfg ai.Config) []option.RequestOption {
	opts := []option.RequestOption{
		// Retries belong to the caller, which alone knows the budget for the
		// whole turn. An SDK retrying underneath would multiply it silently.
		option.WithMaxRetries(0),
	}
	if url := cfg.URL(); url != "" {
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
	return opts
}

// NewWithClient builds a driver over an already-constructed SDK client.
//
// Everything this driver does after construction — message conversion,
// thinking configuration, streaming, error classification — is the same
// whatever produced the client, so a transport that the Config cannot express
// supplies its own client rather than forking the driver.
func NewWithClient(client sdk.Client, cfg ai.Config) (ai.Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}

	return &Driver{
		client: client,
		model:  cfg.Model,
		compat: ai.CompatOf[ai.AnthropicCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Stream runs one Messages call.
func (d *Driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		native, err := ai.ProtocolOptionsAs[Options](req)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}
		params, err := d.buildParams(req, native)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}

		// Betas are the only per-request header this protocol takes; there is
		// nothing set at construction to layer them over.
		var reqOpts []option.RequestOption
		if betas := native.Betas; len(betas) > 0 {
			reqOpts = append(reqOpts, option.WithHeader("anthropic-beta", strings.Join(betas, ",")))
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
				if !yield(ai.Delta{
					Model: string(start.Message.Model),
					ID:    start.Message.ID,
					Usage: &ai.Usage{
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
				var out ai.Delta
				switch delta.Delta.Type {
				case "text_delta":
					out.Block = ai.TextBlock(delta.Delta.Text)
				case "thinking_delta":
					out.Block = ai.ThinkingBlock(delta.Delta.Thinking, "")
				case "signature_delta":
					out.Block = ai.ThinkingBlock("", delta.Delta.Signature)
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
					if !yield(ai.Delta{EndBlock: true}, nil) {
						return
					}
					continue
				}
				call := ai.ToolCall{ID: toolID, Name: toolName, Input: toolInput.String()}
				toolID, toolName = "", ""
				toolInput.Reset()
				if !yield(ai.Delta{Block: ai.ToolCallBlock(call)}, nil) {
					return
				}

			case "message_delta":
				md := event.AsMessageDelta()
				// Anthropic reports only output tokens here. Some
				// Anthropic-compatible endpoints (SenseNova) instead send
				// input tokens in message_delta rather than message_start;
				// zeros are ignored on merge, so passing both is safe either
				// way.
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
			}
		}

		if err := stream.Err(); err != nil {
			yield(ai.Delta{}, d.wrapStream(err))
		}
	}
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
	case "":
		return ""
	default:
		return ai.StopReason(reason)
	}
}

// CountTokens asks the endpoint how large a prompt is, without generating
// from it. Anthropic publishes this, so a caller never has to estimate.
func (d *Driver) CountTokens(ctx context.Context, req *ai.Request) (int, error) {
	native, err := ai.ProtocolOptionsAs[Options](req)
	if err != nil {
		return 0, err
	}
	params, err := d.buildParams(req, native)
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
	res, err := d.client.Messages.CountTokens(ctx, count)
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
// should merge catalog data over the result (see provider.Provider, which
// does exactly that around a refresh).
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	pager := d.client.Models.ListAutoPaging(ctx, sdk.ModelListParams{})
	var out []ai.Model
	for pager.Next() {
		m := pager.Current()
		name := m.DisplayName
		if name == "" {
			name = m.ID
		}
		out = append(out, ai.Model{
			ID:     m.ID,
			Name:   name,
			API:    ai.APIAnthropicMessages,
			Vendor: d.model.Vendor,
		})
	}
	if err := pager.Err(); err != nil {
		return nil, d.wrap(err)
	}
	return out, nil
}

var (
	_ ai.Driver       = (*Driver)(nil)
	_ ai.ModelLister  = (*Driver)(nil)
	_ ai.TokenCounter = (*Driver)(nil)
)
