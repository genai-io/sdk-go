// Package google implements the Google Gemini generateContent protocol.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/google"
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

// generateContentMethod is the entry in a listed model's
// supportedGenerationMethods that says it can answer a prompt.
const generateContentMethod = "generateContent"

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
			yield(ai.Delta{}, fail.WrapStream(err))
			return
		}
		defer func() { _ = res.Body.Close() }() // nothing to do about a failure to close a read body

		for chunk, err := range sseEvents(res.Body) {
			if err != nil {
				yield(ai.Delta{}, fail.WrapStream(err))
				return
			}
			var out generateResponse
			if err := json.Unmarshal(chunk, &out); err != nil {
				yield(ai.Delta{}, fail.WrapStream(fmt.Errorf("undecodable stream chunk: %w", err)))
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
		// Thinking is counted outside candidatesTokenCount here, unlike every
		// other protocol, so it is added in: Output is what the output rate is
		// charged on, and Reasoning is how much of it went unseen.
		reasoning := int(u.ThoughtsTokenCount)
		if !yield(ai.Delta{Usage: &ai.Usage{
			Input:     fresh,
			Output:    int(u.CandidatesTokenCount) + reasoning,
			Reasoning: reasoning,
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
		return 0, fail.Wrap(err)
	}
	defer func() { _ = res.Body.Close() }() // nothing to do about a failure to close a read body

	var out countTokensResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return 0, fail.Wrap(err)
	}
	return int(out.TotalTokens), nil
}

// Models lists the models the endpoint serves that can answer a prompt. The
// listing also carries embedding, image, speech and live models, which say so
// in supportedGenerationMethods. Experimental and "-latest" aliases are dropped
// on top of that: they duplicate a concrete model under a name whose meaning
// changes without notice.
func (d *Driver) Models(ctx context.Context) ([]ai.Model, error) {
	var out []ai.Model
	for token := ""; ; {
		query := "pageSize=1000"
		if token != "" {
			query += "&pageToken=" + url.QueryEscape(token)
		}
		res, err := d.get(ctx, fmt.Sprintf("%s/%s/models?%s", d.baseURL, apiVersion, query))
		if err != nil {
			return nil, fail.Wrap(err)
		}
		var page modelList
		err = json.NewDecoder(res.Body).Decode(&page)
		_ = res.Body.Close() // the decode error above is the one worth reporting
		if err != nil {
			return nil, fail.Wrap(err)
		}

		for _, m := range page.Models {
			id, ok := strings.CutPrefix(m.Name, "models/")
			if !ok {
				id = m.Name
			}
			if !slices.Contains(m.SupportedGenerationMethods, generateContentMethod) {
				continue
			}
			if strings.Contains(id, "-exp") || strings.Contains(id, "-latest") {
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
		return nil, &ai.Error{Driver: Name, Kind: ai.KindUnknown, Message: "endpoint listed no models that generate content"}
	}
	slices.SortFunc(out, func(a, b ai.Model) int { return strings.Compare(a.ID, b.ID) })
	return out, nil
}

var (
	_ ai.Driver       = (*Driver)(nil)
	_ ai.ModelLister  = (*Driver)(nil)
	_ ai.TokenCounter = (*Driver)(nil)
)
