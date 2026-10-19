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
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
)

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

// driver talks to one Gemini endpoint.
type driver struct {
	client *http.Client
	// api is the protocol as the caller reached it: the Gemini API, or the
	// same body through Vertex AI, which counts tokens in its own shape. It
	// is what the driver and its errors answer to, so a failure on Vertex
	// does not send anyone to the Gemini API's console.
	api  ai.API
	fail errs.Classifier
	// modelsURL is the collection the model sits in — the URL up to
	// "/{model}:{method}". On the Gemini API that is "{base}/v1beta/models";
	// on Vertex the project and location come before "publishers/google/models".
	modelsURL string
	apiKey    string
	model     ai.Model
	compat    ai.GoogleCompat
}

// New builds a driver from a Config. Registered as the factory for
// ai.APIGoogleGenAI.
func New(cfg ai.Config) (ai.Driver, error) {
	base := cmp.Or(cfg.Model.BaseURL, defaultBaseURL)
	return NewDeployment(cfg, ai.APIGoogleGenAI, strings.TrimSuffix(base, "/")+"/"+apiVersion+"/models")
}

// NewDeployment builds a driver for a model collection, for a package that
// serves this protocol from another address: google/vertex passes the
// collection its project and location name, with the Config's client carrying
// its Google credential. A nil Config.HTTPClient is http.DefaultClient.
func NewDeployment(cfg ai.Config, api ai.API, modelsURL string) (ai.Driver, error) {
	return &driver{
		client:    cmp.Or(cfg.HTTPClient, http.DefaultClient),
		api:       api,
		fail:      errs.For(string(api), details),
		modelsURL: modelsURL,
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		compat:    ai.CompatOf[ai.GoogleCompat](cfg.Model),
	}, nil
}

// Name identifies the driver by the protocol it was reached through.
func (d *driver) Name() string { return string(d.api) }

// Stream runs one streamGenerateContent call.
func (d *driver) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		body, err := d.buildParams(req)
		if err != nil {
			yield(ai.Delta{}, err)
			return
		}

		res, err := d.post(ctx, d.methodURL("streamGenerateContent", "alt=sse"), body)
		if err != nil {
			yield(ai.Delta{}, d.fail.WrapStream(err))
			return
		}
		defer func() { _ = res.Body.Close() }() // nothing to do about a failure to close a read body

		// Gemini finishes a turn that called a function with STOP, and the call
		// may arrive in an earlier chunk than the finish.
		var called, complete bool
		for chunk, err := range sseEvents(res.Body) {
			if err != nil {
				yield(ai.Delta{}, d.fail.WrapStream(err))
				return
			}
			var out generateResponse
			if err := json.Unmarshal(chunk, &out); err != nil {
				yield(ai.Delta{}, d.fail.WrapStream(fmt.Errorf("undecodable stream chunk: %w", err)))
				return
			}
			var stopped, ok bool
			if called, stopped, ok = d.emit(out, called, yield); !ok {
				return
			}
			complete = complete || stopped
		}
		if !complete {
			yield(ai.Delta{}, d.fail.WrapStream(io.ErrUnexpectedEOF))
		}
	}
}

// emit turns one decoded chunk into deltas. called is whether the turn has
// made a function call so far, and comes back updated; stopped is whether the
// chunk carried a stop reason; ok is false when the consumer stopped iterating.
func (d *driver) emit(out generateResponse, called bool, yield func(ai.Delta, error) bool) (bool, bool, bool) {
	stopped := false
	// A blocked prompt gets no candidate, only the reason, which is carried as
	// the refusal's text so the caller does not see an empty answer.
	if f := out.PromptFeedback; f != nil && f.BlockReason != "" {
		msg := f.BlockReasonMessage
		if msg == "" {
			msg = "prompt blocked: " + f.BlockReason
		}
		stopped = true
		if !yield(ai.Delta{Block: ai.TextBlock(msg), StopReason: ai.StopRefusal}, nil) {
			return called, stopped, false
		}
	}
	for _, c := range out.Candidates {
		if c.Content != nil {
			for _, part := range c.Content.Parts {
				delta, ok := convertPart(part)
				if !ok {
					continue
				}
				if delta.Block.ToolCall != nil {
					called = true
				}
				if !yield(delta, nil) {
					return called, stopped, false
				}
			}
		}
		if c.FinishReason != "" {
			stop := mapFinishReason(c.FinishReason)
			if stop == ai.StopEndTurn && called {
				stop = ai.StopToolUse
			}
			stopped = true
			if !yield(ai.Delta{StopReason: stop}, nil) {
				return called, stopped, false
			}
		}
	}
	if out.ResponseID != "" {
		if !yield(ai.Delta{ID: out.ResponseID}, nil) {
			return called, stopped, false
		}
	}
	if u := out.UsageMetadata; u != nil {
		// Gemini reports the cached prefix inside the prompt count, so it is
		// split out the same way the OpenAI protocols are.
		fresh, cached := ai.SplitPromptTokens(int(u.PromptTokenCount), int(u.CachedContentTokenCount))
		// Gemini counts thinking outside candidatesTokenCount, unlike every other
		// protocol, so it is added into Output — what the output rate charges on.
		reasoning := int(u.ThoughtsTokenCount)
		if !yield(ai.Delta{Usage: &ai.Usage{
			Input:     fresh,
			Output:    int(u.CandidatesTokenCount) + reasoning,
			Reasoning: reasoning,
			CacheRead: cached,
		}}, nil) {
			return called, stopped, false
		}
	}
	return called, stopped, true
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
func (d *driver) CountTokens(ctx context.Context, req *ai.Request) (int, error) {
	contents, err := d.convertContents(req)
	if err != nil {
		return 0, err
	}
	inner := &countTokensInner{Contents: contents}
	// The system instruction and the tool declarations count against the
	// window too.
	if req.System != "" {
		inner.SystemInstruction = &content{Parts: []*part{{Text: req.System}}}
	}
	inner.Tools = declarations(req.Tools)

	// Vertex takes those fields flat. The Gemini API takes them only inside a
	// generateContentRequest wrapper, which also has to name the model.
	var body any = inner
	if d.api == ai.APIGoogleGenAI {
		inner.Model = "models/" + d.model.ID
		body = &countTokensRequest{GenerateContentRequest: inner}
	}
	res, err := d.post(ctx, d.methodURL("countTokens", ""), body)
	if err != nil {
		return 0, d.fail.Wrap(err)
	}
	defer func() { _ = res.Body.Close() }() // nothing to do about a failure to close a read body

	var out countTokensResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return 0, d.fail.Wrap(err)
	}
	return int(out.TotalTokens), nil
}

// Models lists the models the endpoint serves that can answer a prompt. The
// listing also carries embedding, image, speech and live models, which say so
// in supportedGenerationMethods. Experimental and "-latest" aliases are dropped
// too: they duplicate a concrete model under a name whose meaning can change.
func (d *driver) Models(ctx context.Context) ([]ai.Model, error) {
	var out []ai.Model
	for token := ""; ; {
		query := "pageSize=1000"
		if token != "" {
			query += "&pageToken=" + url.QueryEscape(token)
		}
		res, err := d.get(ctx, d.modelsURL+"?"+query)
		if err != nil {
			return nil, d.fail.Wrap(err)
		}
		var page modelList
		err = json.NewDecoder(res.Body).Decode(&page)
		_ = res.Body.Close() // the decode error above is the one worth reporting
		if err != nil {
			return nil, d.fail.Wrap(err)
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
			name := cmp.Or(m.DisplayName, id)
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
		return nil, &ai.Error{Driver: d.Name(), Kind: ai.KindUnknown, Message: "endpoint listed no models that generate content"}
	}
	slices.SortFunc(out, func(a, b ai.Model) int { return strings.Compare(a.ID, b.ID) })
	return out, nil
}

var (
	_ ai.Driver       = (*driver)(nil)
	_ ai.ModelLister  = (*driver)(nil)
	_ ai.TokenCounter = (*driver)(nil)
)
