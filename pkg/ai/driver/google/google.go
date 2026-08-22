// Package google implements the Google Gemini generateContent protocol.
//
// Import it for its side effect to make ai.NewClient handle ai.APIGoogleGenAI:
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/google"
//
// It speaks the REST API directly rather than through google.golang.org/genai.
// That SDK also serves Vertex AI, so it carries gRPC, protobuf, OpenTelemetry
// and Google's cloud credential stack — around 170 packages a program reaches
// for Gemini with an API key never uses. The wire format it encodes is
// transcribed in wire.go.
//
// This is the Gemini protocol, not "Google's models". Claude served through
// Google Cloud Vertex AI is a deployment of the Anthropic Messages protocol
// and lives in ai/driver/anthropic/vertex — its wire format, its SDK and its
// conversion code are Anthropic's, and only the host and the credentials are
// Google's. Keeping it out of here is also what lets this package cost 19
// packages instead of 271: it speaks REST with an API key and links no Google
// credential stack at all.
//
// # Where things live
//
//	google.go     construction, Stream, Models and CountTokens
//	request.go    an ai.Request translated into a generate call
//	wire.go       the protocol's own types, transcribed from the API definition
//	transport.go  URLs, headers and the server-sent-event stream
//	errors.go     this protocol's failures classified into ai.Error kinds
package google

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Name is the driver's identifier.
const Name = string(ai.APIGoogleGenAI)

// defaultBaseURL is the Gemini API host; apiVersion is the path segment every
// method sits under.
const (
	defaultBaseURL = "https://generativelanguage.googleapis.com"
	apiVersion     = "v1beta"
)

func init() { ai.RegisterAPI(ai.APIGoogleGenAI, New) }

// Driver talks to one Gemini endpoint.
type Driver struct {
	client  *http.Client
	baseURL string
	apiKey  string
	headers map[string]string
	model   ai.Model
	compat  ai.GoogleCompat
}

// New builds a driver from a Config. Registered as the factory for
// ai.APIGoogleGenAI.
func New(cfg ai.Config) (ai.Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}
	if err := ai.RejectProtocolConfig(cfg, Name); err != nil {
		return nil, err
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	base := cfg.URL()
	if base == "" {
		base = defaultBaseURL
	}

	return &Driver{
		client:  client,
		baseURL: strings.TrimSuffix(base, "/"),
		apiKey:  cfg.APIKey,
		headers: cfg.MergedHeaders(),
		model:   cfg.Model,
		compat:  ai.CompatOf[ai.GoogleCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Stream runs one streamGenerateContent call.
func (d *Driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		body, err := d.request(req)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}

		res, err := d.post(ctx, d.methodURL("streamGenerateContent", "alt=sse"), body)
		if err != nil {
			yield(ai.Delta{}, d.wrapStream(err))
			return
		}
		defer res.Body.Close()

		for chunk, err := range sseEvents(res.Body) {
			if err != nil {
				yield(ai.Delta{}, d.wrapStream(err))
				return
			}
			var out generateResponse
			if err := json.Unmarshal(chunk, &out); err != nil {
				yield(ai.Delta{}, d.wrapStream(fmt.Errorf("undecodable stream chunk: %w", err)))
				return
			}
			if !d.emit(out, yield) {
				return
			}
		}
	}
}

// emit turns one decoded chunk into deltas. It reports false when the consumer
// stopped iterating.
func (d *Driver) emit(out generateResponse, yield func(ai.Delta, error) bool) bool {
	for _, c := range out.Candidates {
		if c.Content != nil {
			for _, part := range c.Content.Parts {
				delta, ok := convertPart(part)
				if !ok {
					continue
				}
				if !yield(delta, nil) {
					return false
				}
			}
		}
		if c.FinishReason != "" {
			if !yield(ai.Delta{StopReason: mapFinishReason(c.FinishReason)}, nil) {
				return false
			}
		}
	}
	if out.ResponseID != "" {
		if !yield(ai.Delta{ID: out.ResponseID}, nil) {
			return false
		}
	}
	if u := out.UsageMetadata; u != nil {
		// Gemini reports the cached prefix inside the prompt count, so it is
		// split out the same way the OpenAI protocols are.
		fresh, cached := ai.SplitPromptTokens(int(u.PromptTokenCount), int(u.CachedContentTokenCount))
		if !yield(ai.Delta{Usage: &ai.Usage{
			Input:     fresh,
			Output:    int(u.CandidatesTokenCount),
			CacheRead: cached,
		}}, nil) {
			return false
		}
	}
	return true
}

// convertPart turns one response part into a delta. Gemini distinguishes
// reasoning from the answer with a flag on the text part rather than with a
// separate event type.
func convertPart(p *part) (ai.Delta, bool) {
	switch {
	case p == nil:
		return ai.Delta{}, false
	case p.Text != "":
		if p.Thought {
			return ai.Delta{Block: ai.ThinkingBlock(p.Text, "")}, true
		}
		return ai.Delta{Block: ai.TextBlock(p.Text)}, true
	case p.FunctionCall != nil:
		args, err := json.Marshal(p.FunctionCall.Args)
		if err != nil {
			args = []byte("{}")
		}
		return ai.Delta{Block: ai.ToolCallBlock(ai.ToolCall{
			ID:    p.FunctionCall.ID,
			Name:  p.FunctionCall.Name,
			Input: string(args),
			// Gemini requires the thought signature to come back with the call
			// on the next turn, or it rejects the conversation.
			Signature: p.ThoughtSignature,
		})}, true
	}
	return ai.Delta{}, false
}

func mapFinishReason(reason string) ai.StopReason {
	switch reason {
	case "STOP":
		return ai.StopEndTurn
	case "MAX_TOKENS":
		return ai.StopMaxTokens
	case "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST":
		return ai.StopRefusal
	case "":
		return ""
	default:
		return ai.StopReason(reason)
	}
}

// CountTokens asks the endpoint how large a prompt is, without generating from
// it. Gemini publishes this, so a caller never has to estimate.
func (d *Driver) CountTokens(ctx context.Context, req *ai.Request) (int, error) {
	contents, err := d.convertContents(req)
	if err != nil {
		return 0, err
	}
	inner := &countTokensInner{Model: "models/" + d.model.ID, Contents: contents}
	// The system instruction and the tool declarations count against the
	// window too, and the wrapper is the only form that accepts them.
	if req.System != "" {
		inner.SystemInstruction = &content{Parts: []*part{{Text: req.System}}}
	}
	inner.Tools = declarations(req.Tools)

	res, err := d.post(ctx, d.methodURL("countTokens", ""), &countTokensRequest{GenerateContentRequest: inner})
	if err != nil {
		return 0, d.wrap(err)
	}
	defer res.Body.Close()

	var out countTokensResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return 0, d.wrap(err)
	}
	return int(out.TotalTokens), nil
}

// Models lists the Gemini models the endpoint serves. Experimental and
// "-latest" aliases are dropped: they duplicate a concrete model under a name
// whose meaning changes without notice.
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	var out []ai.Model
	for token := ""; ; {
		query := "pageSize=1000"
		if token != "" {
			query += "&pageToken=" + url.QueryEscape(token)
		}
		res, err := d.get(ctx, fmt.Sprintf("%s/%s/models?%s", d.baseURL, apiVersion, query))
		if err != nil {
			return nil, d.wrap(err)
		}
		var page modelList
		err = json.NewDecoder(res.Body).Decode(&page)
		res.Body.Close()
		if err != nil {
			return nil, d.wrap(err)
		}

		for _, m := range page.Models {
			id, ok := strings.CutPrefix(m.Name, "models/")
			if !ok {
				id = m.Name
			}
			if !strings.Contains(id, "gemini") || strings.Contains(id, "-exp") || strings.Contains(id, "-latest") {
				continue
			}
			name := m.DisplayName
			if name == "" {
				name = id
			}
			out = append(out, ai.Model{
				ID:            id,
				Name:          name,
				API:           ai.APIGoogleGenAI,
				Vendor:        d.model.Vendor,
				ContextWindow: int(m.InputTokenLimit),
				MaxOutput:     int(m.OutputTokenLimit),
			})
		}
		if page.NextPageToken == "" {
			break
		}
		token = page.NextPageToken
	}
	if len(out) == 0 {
		return nil, &ai.Error{Driver: Name, Kind: ai.KindUnknown, Message: "endpoint listed no Gemini models"}
	}
	slices.SortFunc(out, func(a, b ai.Model) int { return strings.Compare(a.ID, b.ID) })
	return out, nil
}

var (
	_ ai.Driver       = (*Driver)(nil)
	_ ai.ModelLister  = (*Driver)(nil)
	_ ai.TokenCounter = (*Driver)(nil)
)
