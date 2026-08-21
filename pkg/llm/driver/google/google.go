// Package google implements the Google Gemini generateContent protocol.
//
// Import it for its side effect to make llm.Open handle llm.APIGoogleGenAI:
//
//	import _ "github.com/genai-io/sdk-go/pkg/llm/driver/google"
//
// It speaks the REST API directly rather than through google.golang.org/genai.
// That SDK also serves Vertex AI, so it carries gRPC, protobuf, OpenTelemetry
// and Google's cloud credential stack — around 170 packages a program reaches
// for Gemini with an API key never uses. The wire format it encodes is
// transcribed in wire.go.
package google

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Name is the driver's identifier.
const Name = string(llm.APIGoogleGenAI)

// defaultBaseURL is the Gemini API host; apiVersion is the path segment every
// method sits under.
const (
	defaultBaseURL = "https://generativelanguage.googleapis.com"
	apiVersion     = "v1beta"
)

func init() { llm.RegisterAPI(llm.APIGoogleGenAI, New) }

// Driver talks to one Gemini endpoint.
type Driver struct {
	client  *http.Client
	baseURL string
	apiKey  string
	headers map[string]string
	model   llm.Model
	compat  llm.GoogleCompat
}

// New builds a driver from a Config. Registered as the factory for
// llm.APIGoogleGenAI.
func New(cfg llm.Config) (llm.Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("%s: model ID is required", Name)
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	base := cfg.Endpoint()
	if base == "" {
		base = defaultBaseURL
	}

	// Model headers first, so a Config header of the same name wins.
	headers := map[string]string{}
	for k, v := range cfg.Model.Headers {
		headers[k] = v
	}
	for k, v := range cfg.Headers {
		headers[k] = v
	}

	return &Driver{
		client:  client,
		baseURL: strings.TrimSuffix(base, "/"),
		apiKey:  cfg.APIKey,
		headers: headers,
		model:   cfg.Model,
		compat:  llm.CompatOf[llm.GoogleCompat](cfg.Model),
	}, nil
}

// Name identifies the driver.
func (d *Driver) Name() string { return Name }

// Generate runs one streamGenerateContent call.
func (d *Driver) Generate(ctx context.Context, p *llm.Prompt, opts llm.Options) iter.Seq2[llm.Delta, error] {
	return func(yield func(llm.Delta, error) bool) {
		body, err := d.request(p, opts)
		if err != nil {
			yield(llm.Delta{}, err)
			return
		}

		res, err := d.post(ctx, d.methodURL("streamGenerateContent", "alt=sse"), body)
		if err != nil {
			yield(llm.Delta{}, d.wrapStream(err))
			return
		}
		defer res.Body.Close()

		for chunk, err := range sseEvents(res.Body) {
			if err != nil {
				yield(llm.Delta{}, d.wrapStream(err))
				return
			}
			var out generateResponse
			if err := json.Unmarshal(chunk, &out); err != nil {
				yield(llm.Delta{}, d.wrapStream(fmt.Errorf("undecodable stream chunk: %w", err)))
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
func (d *Driver) emit(out generateResponse, yield func(llm.Delta, error) bool) bool {
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
			if !yield(llm.Delta{StopReason: mapFinishReason(c.FinishReason)}, nil) {
				return false
			}
		}
	}
	if out.ResponseID != "" {
		if !yield(llm.Delta{ID: out.ResponseID}, nil) {
			return false
		}
	}
	if u := out.UsageMetadata; u != nil {
		// Gemini reports the cached prefix inside the prompt count, so it is
		// split out the same way the OpenAI protocols are.
		fresh, cached := llm.SplitPromptTokens(int(u.PromptTokenCount), int(u.CachedContentTokenCount))
		if !yield(llm.Delta{Usage: &llm.Usage{
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
func convertPart(p *part) (llm.Delta, bool) {
	switch {
	case p == nil:
		return llm.Delta{}, false
	case p.Text != "":
		if p.Thought {
			return llm.Delta{Thinking: p.Text}, true
		}
		return llm.Delta{Text: p.Text}, true
	case p.FunctionCall != nil:
		args, err := json.Marshal(p.FunctionCall.Args)
		if err != nil {
			args = []byte("{}")
		}
		return llm.Delta{ToolCall: &llm.ToolCall{
			ID:    p.FunctionCall.ID,
			Name:  p.FunctionCall.Name,
			Input: string(args),
			// Gemini requires the thought signature to come back with the call
			// on the next turn, or it rejects the conversation.
			Signature: p.ThoughtSignature,
		}}, true
	}
	return llm.Delta{}, false
}

func mapFinishReason(reason string) llm.StopReason {
	switch reason {
	case "STOP":
		return llm.StopEndTurn
	case "MAX_TOKENS":
		return llm.StopMaxTokens
	case "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST":
		return llm.StopRefusal
	case "":
		return ""
	default:
		return llm.StopReason(reason)
	}
}

// request assembles the body of a generate call.
func (d *Driver) request(p *llm.Prompt, opts llm.Options) (*generateRequest, error) {
	contents, err := d.convertContents(p)
	if err != nil {
		return nil, err
	}
	req := &generateRequest{
		Contents:         contents,
		GenerationConfig: d.generationConfig(opts),
		ToolConfig:       toolChoiceConfig(opts),
		Tools:            declarations(p.Tools),
	}
	if p.System != "" {
		req.SystemInstruction = &content{Parts: []*part{{Text: p.System}}}
	}
	return req, nil
}

func (d *Driver) generationConfig(opts llm.Options) *generationConfig {
	cfg := &generationConfig{}
	if opts.MaxTokens > 0 {
		cfg.MaxOutputTokens = int32(opts.MaxTokens)
	}
	if opts.Temperature > 0 {
		t := float32(opts.Temperature)
		cfg.Temperature = &t
	}
	if len(opts.StopSequences) > 0 {
		cfg.StopSequences = opts.StopSequences
	}
	// Gemini 3 replaced the token budget with a level; 2.5 still takes a
	// budget. The rung carries whichever this model's endpoint wants.
	if level, ok := d.model.ResolveLevel(opts.Effort); ok {
		switch {
		case d.compat.ThinkingLevel && level.Value != "":
			cfg.ThinkingConfig = &thinkingConfig{IncludeThoughts: true, ThinkingLevel: level.Value}
		case !d.compat.ThinkingLevel && level.Budget > 0:
			budget := int32(level.Budget)
			cfg.ThinkingConfig = &thinkingConfig{IncludeThoughts: true, ThinkingBudget: &budget}
		}
	}
	// Gemini takes the schema as raw JSON Schema alongside a JSON mime type;
	// the mime type alone would only promise valid JSON, not the shape.
	if opts.Schema != nil && opts.Schema.Definition != nil {
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJSONSchema = opts.Schema.Definition
	}
	return cfg
}

func declarations(tools []llm.Tool) []*tool {
	if len(tools) == 0 {
		return nil
	}
	decls := make([]*functionDeclaration, 0, len(tools))
	for _, t := range tools {
		decl := &functionDeclaration{Name: t.Name, Description: t.Description}
		if t.Parameters != nil {
			decl.ParametersJSONSchema = t.Parameters
		}
		decls = append(decls, decl)
	}
	return []*tool{{FunctionDeclarations: decls}}
}

// toolChoiceConfig maps the neutral constraint onto Gemini's function-calling
// mode. A nil result leaves the field off, which is the API's own default.
func toolChoiceConfig(opts llm.Options) *toolConfig {
	// Forcing one tool is ANY mode narrowed to a single allowed name.
	if opts.ForceTool != "" {
		return &toolConfig{FunctionCallingConfig: &functionCallingConfig{
			Mode:                 modeAny,
			AllowedFunctionNames: []string{opts.ForceTool},
		}}
	}
	var mode string
	switch opts.ToolChoice {
	case llm.ToolChoiceNone:
		mode = modeNone
	case llm.ToolChoiceRequired:
		mode = modeAny
	default:
		return nil
	}
	return &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: mode}}
}

func (d *Driver) convertContents(p *llm.Prompt) ([]*content, error) {
	msgs := llm.PrepareMessages(p.Messages)
	out := make([]*content, 0, len(msgs))

	for _, msg := range msgs {
		role := "user"
		if msg.Role == llm.RoleAssistant {
			role = "model"
		}

		var parts []*part
		switch {
		case msg.IsToolResult():
			for _, r := range msg.ToolResults {
				// The protocol wants a JSON object; wrap anything else so a
				// plain-text tool result still round-trips.
				var response map[string]any
				if err := json.Unmarshal([]byte(r.Content), &response); err != nil {
					response = map[string]any{"result": r.Content}
				}
				parts = append(parts, &part{FunctionResponse: &functionResponse{
					ID:       r.ToolCallID,
					Name:     r.ToolName,
					Response: response,
				}})
			}

		case len(msg.ToolCalls) > 0:
			if text := msg.Content.String(); text != "" {
				parts = append(parts, &part{Text: text})
			}
			for _, tc := range msg.ToolCalls {
				var args map[string]any
				if tc.Input != "" {
					_ = json.Unmarshal([]byte(tc.Input), &args)
				}
				parts = append(parts, &part{
					FunctionCall:     &functionCall{ID: tc.ID, Name: tc.Name, Args: args},
					ThoughtSignature: tc.Signature,
				})
			}

		default:
			converted, err := contentParts(msg.Content)
			if err != nil {
				return nil, err
			}
			parts = converted
		}

		if len(parts) == 0 {
			continue
		}
		out = append(out, &content{Role: role, Parts: parts})
	}
	return out, nil
}

// contentParts converts message content. Gemini takes image bytes rather than
// base64 text, so a corrupt attachment is reported here instead of failing as
// an opaque server-side rejection.
func contentParts(c llm.Content) ([]*part, error) {
	parts := make([]*part, 0, len(c))
	for _, p := range c {
		switch p.Type {
		case llm.PartText:
			if p.Text != "" {
				parts = append(parts, &part{Text: p.Text})
			}
		case llm.PartImage:
			if p.Image == nil {
				continue
			}
			decoded, err := base64.StdEncoding.DecodeString(p.Image.Data)
			if err != nil {
				return nil, &llm.Error{
					Driver:  Name,
					Kind:    llm.KindInvalidRequest,
					Message: fmt.Sprintf("image %q is not valid base64", p.Image.FileName),
					Err:     err,
				}
			}
			parts = append(parts, &part{InlineData: &blob{MIMEType: p.Image.MediaType, Data: decoded}})
		}
	}
	return parts, nil
}

// CountTokens asks the endpoint how large a prompt is, without generating from
// it. Gemini publishes this, so a caller never has to estimate.
func (d *Driver) CountTokens(ctx context.Context, p *llm.Prompt, opts llm.Options) (int, error) {
	contents, err := d.convertContents(p)
	if err != nil {
		return 0, err
	}
	inner := &countTokensInner{Model: "models/" + d.model.ID, Contents: contents}
	// The system instruction and the tool declarations count against the
	// window too, and the wrapper is the only form that accepts them.
	if p.System != "" {
		inner.SystemInstruction = &content{Parts: []*part{{Text: p.System}}}
	}
	inner.Tools = declarations(p.Tools)

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
func (d *Driver) Models(ctx context.Context) ([]llm.Model, error) {
	var out []llm.Model
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
			out = append(out, llm.Model{
				ID:            id,
				Name:          name,
				API:           llm.APIGoogleGenAI,
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
		return nil, &llm.Error{Driver: Name, Kind: llm.KindUnknown, Message: "endpoint listed no Gemini models"}
	}
	slices.SortFunc(out, func(a, b llm.Model) int { return strings.Compare(a.ID, b.ID) })
	return out, nil
}

// methodURL builds the URL for a model method, e.g. ":countTokens".
func (d *Driver) methodURL(method, query string) string {
	u := fmt.Sprintf("%s/%s/models/%s:%s", d.baseURL, apiVersion, d.model.ID, method)
	if query != "" {
		u += "?" + query
	}
	return u
}

func (d *Driver) post(ctx context.Context, url string, body any) (*http.Response, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return d.do(req)
}

func (d *Driver) get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	return d.do(req)
}

func (d *Driver) do(req *http.Request) (*http.Response, error) {
	// The key travels in a header rather than the query string the REST
	// examples use: a URL ends up in proxy logs and error reports, and a
	// credential should not go with it.
	if d.apiKey != "" {
		req.Header.Set("x-goog-api-key", d.apiKey)
	}
	for k, v := range d.headers {
		req.Header.Set(k, v)
	}
	res, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= 400 {
		defer res.Body.Close()
		return nil, readAPIError(res)
	}
	return res, nil
}

// readAPIError turns a failed response into the driver's own error, keeping
// the response so a 429's Retry-After is honoured.
func readAPIError(res *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	var envelope apiError
	_ = json.Unmarshal(raw, &envelope)
	message := strings.TrimSpace(envelope.Error.Message)
	if message == "" {
		message = strings.TrimSpace(string(raw))
	}
	return &statusError{status: res.StatusCode, code: envelope.Error.Status, message: message, response: res}
}

// statusError carries what the driver knows about a failed response through to
// classification.
type statusError struct {
	status   int
	code     string
	message  string
	response *http.Response
}

func (e *statusError) Error() string {
	if e.code != "" {
		return fmt.Sprintf("%s: %s", e.code, e.message)
	}
	return e.message
}

// sseEvents yields the payload of each "data:" event until the stream ends.
func sseEvents(r io.Reader) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		scanner := bufio.NewScanner(r)
		// A single event can carry a whole turn's worth of tool arguments; the
		// default 64KiB limit would cut one in half.
		scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		for scanner.Scan() {
			line := strings.TrimRight(scanner.Text(), "\r")
			payload, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			payload = strings.TrimSpace(payload)
			if payload == "" || payload == "[DONE]" {
				continue
			}
			if !yield([]byte(payload), nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
		}
	}
}

func (d *Driver) wrap(err error) error {
	status, resp, code, msg := errorDetails(err)
	return llm.Classify(Name, status, resp, code, msg, err)
}

func (d *Driver) wrapStream(err error) error {
	status, resp, code, msg := errorDetails(err)
	return llm.StreamError(Name, status, resp, code, msg, err)
}

// errorDetails reads what the endpoint reported. Unlike the SDK's error type,
// the response is kept, so a 429 that carries Retry-After is honoured rather
// than falling back to the caller's own backoff.
func errorDetails(err error) (status int, resp *http.Response, code, message string) {
	var se *statusError
	if errors.As(err, &se) {
		return se.status, se.response, se.code, se.message
	}
	return 0, nil, "", ""
}

var (
	_ llm.Driver       = (*Driver)(nil)
	_ llm.TokenCounter = (*Driver)(nil)
)
