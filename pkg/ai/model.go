package ai

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"
)

// API is a wire protocol — the request/response shape an endpoint speaks.
//
// This, not the vendor name, is what decides which driver handles a model.
// Most vendors ship an endpoint that speaks somebody else's protocol
// (DeepSeek, Moonshot and Ollama speak OpenAI Chat Completions; MiniMax,
// Xiaomi MiMo and Volcengine speak Anthropic Messages), so a vendor is a row
// in the catalog and only a protocol needs code.
type API string

const (
	// APIAnthropicMessages is the Anthropic /v1/messages protocol.
	APIAnthropicMessages API = "anthropic-messages"
	// APIOpenAIChat is the OpenAI /v1/chat/completions protocol, the de-facto
	// interchange format.
	APIOpenAIChat API = "openai-chat-completions"
	// APIOpenAIResponses is the OpenAI /v1/responses protocol.
	APIOpenAIResponses API = "openai-responses"
	// APIGoogleGenAI is the Google Gemini generateContent protocol.
	APIGoogleGenAI API = "google-genai"
	// APIAnthropicVertex is the Anthropic Messages protocol served through
	// Google Cloud Vertex AI. The wire format is identical to
	// APIAnthropicMessages; what differs is how the client authenticates and
	// where it points, which is why it is a separate protocol only so far as
	// routing is concerned — the driver reuses the same conversion code.
	APIAnthropicVertex API = "anthropic-vertex"
)

// VertexConfig is the deployment a Vertex-served model lives in. It is passed
// as Config.Native to the anthropic/vertex driver.
//
// It lives here rather than beside that driver so a caller can fill it in —
// from the environment, from a settings file — without importing the driver
// and its Google Cloud auth dependencies.
type VertexConfig struct {
	// Project is the GCP project ID serving the model.
	Project string
	// Region is the serving region. "global" is the usual choice and is what
	// an empty value resolves to.
	Region string
}

// NativeConfig marks VertexConfig as a driver's construction settings.
func (VertexConfig) NativeConfig() {}

// Modality is a kind of content a model accepts as input.
//
// A list rather than a set of booleans: providers keep adding kinds, and a
// caller checking Accepts(ModalityAudio) against a model that predates audio
// gets a correct "no" without the model needing a field for it.
type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
	ModalityVideo Modality = "video"
	ModalityPDF   Modality = "pdf"
)

// ReasoningLevel is one rung of a model's reasoning ladder: what a caller asks
// for, and what this particular endpoint wants sent for it.
//
// Both halves have to be here. A rung that carried only the endpoint's literal
// ("think+", "high", "enabled") would force every caller to learn each
// vendor's vocabulary, and the same prompt would stop running across
// providers — which is the entire point of the normalized ladder. A rung that
// carried only the normalized Effort would push the mapping back into driver
// code, which is where it was before.
//
// It is a slice on Model rather than a map because the order is meaningful
// (clamping walks it), because a rung needs more than one value (Anthropic's
// 4.5 generation wants a token budget where 4.6+ wants an effort string, and
// one type covers both), and because a slice serializes deterministically.
type ReasoningLevel struct {
	// Effort is the normalized rung — the caller's vocabulary.
	Effort Effort `json:"effort"`
	// Value is the literal this endpoint wants. An empty string means the
	// parameter is omitted entirely for this rung, which is how "off" is
	// expressed on an endpoint that reasons only when asked.
	Value string `json:"value,omitempty"`
	// Budget is the token budget for endpoints that take one instead of, or
	// alongside, a level. Zero means none.
	Budget int `json:"budget,omitempty"`
	// Default marks the rung used when a request leaves Effort unset. At most
	// one rung should set it; the first one wins.
	Default bool `json:"default,omitempty"`
}

// Model is everything the SDK needs to talk to one model: which protocol
// serves it, where, and what it can do.
//
// Values come from the catalog, from a provider's live model listing, or from
// a caller who wants neither — a hand-built Model with ID, API and BaseURL set
// is enough to reach any endpoint.
type Model struct {
	ID   string `json:"id"`
	API  API    `json:"api"`
	Name string `json:"name,omitempty"`

	// Vendor is the catalog vendor this model came from, e.g. "deepseek". It
	// is informational: routing keys off API.
	Vendor string `json:"vendor,omitempty"`

	// BaseURL overrides the driver's default endpoint. Empty means the
	// protocol's own default host.
	BaseURL string `json:"base_url,omitempty"`

	// ContextWindow is the maximum input tokens, and MaxOutput the maximum
	// tokens the model will generate in one turn. Zero means unknown: callers
	// must treat that as "cannot size the window" rather than substituting a
	// guess, because acting on a guessed limit fails silently in both
	// directions.
	ContextWindow int `json:"context_window,omitempty"`
	MaxOutput     int `json:"max_output,omitempty"`

	// Input lists the content kinds this model accepts. An empty list means
	// text only.
	Input []Modality `json:"input,omitempty"`

	// Reasoning is the model's ladder, ordered from least to most effort. Nil
	// means the model does not reason.
	Reasoning []ReasoningLevel `json:"reasoning,omitempty"`

	Pricing Pricing `json:"pricing,omitempty"`

	// Unsupported records what this model cannot do. Its zero value is a
	// fully capable model.
	Unsupported Unsupported `json:"unsupported,omitempty"`

	// Stage is where the model sits in its vendor's lifecycle. The zero value
	// is a stable, generally available model.
	Stage Stage `json:"stage,omitempty"`

	// Replacement names the model to move to, for one that is deprecated or
	// retired.
	Replacement string `json:"replacement,omitempty"`

	// SamplingParams are merged into the request body verbatim, after the
	// named fields, so a custom OpenAI-compatible server (llama.cpp, vLLM,
	// SGLang) can receive parameters this SDK does not model — top_p, top_k,
	// min_p, repetition_penalty. Per-request WithSamplingParams overrides
	// these key by key. Only the OpenAI-family drivers apply them.
	SamplingParams map[string]any `json:"sampling_params,omitempty"`

	// Headers are added to every request for this model, on top of whatever
	// the Config carries.
	Headers map[string]string `json:"headers,omitempty"`

	// Compat holds the protocol-specific behavior flags this endpoint needs —
	// one of AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat or
	// GoogleCompat, by value. Read it with CompatOf, which yields the zero
	// value when the model carries none, so "not stated" and "all defaults"
	// are the same thing.
	//
	// It is `any` rather than a type parameter because a []Model has to hold
	// models of different protocols; a generic Model[T] could not.
	Compat any `json:"compat,omitempty"`
}

// Accepts reports whether the model takes the given input kind. A model that
// lists no modalities is text-only.
func (m Model) Accepts(kind Modality) bool {
	if len(m.Input) == 0 {
		return kind == ModalityText
	}
	return slices.Contains(m.Input, kind)
}

// Reasons reports whether the model has a reasoning ladder.
func (m Model) Reasons() bool { return len(m.Reasoning) > 0 }

// Efforts returns the rungs the model advertises, least to most.
func (m Model) Efforts() []Effort {
	out := make([]Effort, len(m.Reasoning))
	for i, level := range m.Reasoning {
		out[i] = level.Effort
	}
	return out
}

// Offers reports whether the model's ladder declares this exact rung. Use it
// to ask a model what it can do before asking it to do it; ResolveLevel is
// what happens when you ask for a rung it does not declare.
func (m Model) Offers(e Effort) bool {
	for _, level := range m.Reasoning {
		if level.Effort == e {
			return true
		}
	}
	return false
}

// DefaultLevel returns the rung used when a request leaves Effort unset, and
// whether the model states one. A model with a ladder but no default sends
// nothing, leaving the provider's own default in place.
func (m Model) DefaultLevel() (ReasoningLevel, bool) {
	for _, level := range m.Reasoning {
		if level.Default {
			return level, true
		}
	}
	return ReasoningLevel{}, false
}

// ResolveLevel picks the rung to send for a requested effort.
//
// An exact rung wins — which is also what carries a rung this package does not
// name, when the model's own ladder declares it. Otherwise the search runs
// *upward* first and only falls back downward when nothing above exists:
// quietly reasoning less than asked is the more surprising failure, so a
// request for "low" against an off/high endpoint turns reasoning on rather
// than off. Ordering comes from the portable ladder, so a rung outside it that
// did not match exactly falls back to the model's default rather than being
// placed by guesswork.
//
// EffortDefault yields the model's default rung, or no rung at all when it
// states none. A model with no ladder never yields a rung.
func (m Model) ResolveLevel(want Effort) (ReasoningLevel, bool) {
	if !m.Reasons() {
		return ReasoningLevel{}, false
	}
	if want == EffortDefault {
		return m.DefaultLevel()
	}
	for _, level := range m.Reasoning {
		if level.Effort == want {
			return level, true
		}
	}

	rank, ok := effortRank(want)
	if !ok {
		return m.DefaultLevel()
	}
	// Upward first.
	for _, level := range m.Reasoning {
		if r, ok := effortRank(level.Effort); ok && r > rank {
			return level, true
		}
	}
	// Nothing above: take the highest rung below.
	for i := len(m.Reasoning) - 1; i >= 0; i-- {
		if r, ok := effortRank(m.Reasoning[i].Effort); ok && r < rank {
			return m.Reasoning[i], true
		}
	}
	return m.DefaultLevel()
}

// String returns "vendor/id", or just the ID for a model with no vendor.
func (m Model) String() string {
	if m.Vendor == "" {
		return m.ID
	}
	return m.Vendor + "/" + m.ID
}

// The JSON codec for Model.
//
// A Model has to survive being written to a session file and read back. Compat
// is an `any`, and the encoding/json default turns it into a map[string]any on
// the way in — so CompatOf yields the zero value and the model silently loses
// its protocol dialect. A DeepSeek model reloaded that way stops sending its
// "reasoning off" field, and reasoning stays on with nothing reporting it.
//
// The API field is the discriminator: it already says which protocol the model
// speaks, so it says which compat type belongs to it. See compat.go.

// modelJSON is Model without the custom marshalling, so the two methods below
// can delegate the ordinary fields to the standard encoder.
//
// It is not named for a wire: nothing sends a Model to a provider. Everywhere
// else in this package "wire" means the request shape a vendor's API speaks,
// and this is only how our own Model is written to a file or a cache.
type modelJSON struct {
	ID             string            `json:"id"`
	API            API               `json:"api"`
	Name           string            `json:"name,omitempty"`
	Vendor         string            `json:"vendor,omitempty"`
	BaseURL        string            `json:"base_url,omitempty"`
	ContextWindow  int               `json:"context_window,omitempty"`
	MaxOutput      int               `json:"max_output,omitempty"`
	Input          []Modality        `json:"input,omitempty"`
	Reasoning      []ReasoningLevel  `json:"reasoning,omitempty"`
	Pricing        Pricing           `json:"pricing,omitempty"`
	Unsupported    Unsupported       `json:"unsupported,omitempty"`
	Stage          Stage             `json:"stage,omitempty"`
	Replacement    string            `json:"replacement,omitempty"`
	SamplingParams map[string]any    `json:"sampling_params,omitempty"`
	Headers        map[string]string `json:"headers,omitempty"`
	Compat         json.RawMessage   `json:"compat,omitempty"`
}

// MarshalJSON writes the model, compat included.
func (m Model) MarshalJSON() ([]byte, error) {
	out := modelJSON{
		ID: m.ID, API: m.API, Name: m.Name, Vendor: m.Vendor, BaseURL: m.BaseURL,
		ContextWindow: m.ContextWindow, MaxOutput: m.MaxOutput,
		Input: m.Input, Reasoning: m.Reasoning, Pricing: m.Pricing,
		Unsupported: m.Unsupported, Stage: m.Stage, Replacement: m.Replacement,
		SamplingParams: m.SamplingParams, Headers: m.Headers,
	}
	if m.Compat != nil {
		raw, err := json.Marshal(m.Compat)
		if err != nil {
			return nil, fmt.Errorf("ai: encoding compat for model %s: %w", m, err)
		}
		out.Compat = raw
	}
	return json.Marshal(out)
}

// UnmarshalJSON reads a model back, rebuilding compat as the concrete type its
// protocol uses.
//
// A protocol with no registered decoder is an error rather than a silent
// downgrade: a model whose quirks were dropped looks fine and misbehaves
// later, which is the worse of the two outcomes.
func (m *Model) UnmarshalJSON(data []byte) error {
	var in modelJSON
	if err := json.Unmarshal(data, &in); err != nil {
		return err
	}
	*m = Model{
		ID: in.ID, API: in.API, Name: in.Name, Vendor: in.Vendor, BaseURL: in.BaseURL,
		ContextWindow: in.ContextWindow, MaxOutput: in.MaxOutput,
		Input: in.Input, Reasoning: in.Reasoning, Pricing: in.Pricing,
		Unsupported: in.Unsupported, Stage: in.Stage, Replacement: in.Replacement,
		SamplingParams: in.SamplingParams, Headers: in.Headers,
	}
	if len(in.Compat) == 0 || string(in.Compat) == "null" {
		return nil
	}

	t, ok := compatType(in.API)
	if !ok {
		return fmt.Errorf("ai: model %s carries compat for unregistered protocol %q; "+
			"call RegisterCompat for it, or the model would load with its quirks silently dropped", m, in.API)
	}
	compat := reflect.New(t)
	if err := json.Unmarshal(in.Compat, compat.Interface()); err != nil {
		return fmt.Errorf("ai: decoding compat for model %s: %w", m, err)
	}
	m.Compat = compat.Elem().Interface()
	return nil
}

// What a model can and cannot do, and where it sits in its vendor's lifecycle.
// These are declarations only; validate.go is what turns them into the errors a
// caller sees.

// Unsupported records what a model cannot do.
//
// It is stated as absences rather than capabilities so that its zero value is
// a fully capable model, which is nearly all of them — an entry says only what
// is missing. That also means a hand-built Model, or one that arrived from a
// live listing with nothing but an ID, is assumed capable rather than assumed
// crippled.
type Unsupported struct {
	// Tools means the endpoint rejects tool definitions outright. Common on
	// small local models and on the older completion-style endpoints.
	Tools bool `json:"tools,omitempty"`
	// ToolChoice means tools work but constraining which one does not.
	ToolChoice bool `json:"tool_choice,omitempty"`
	// System means there is no system role; the instructions have to go in the
	// first user message.
	System bool `json:"system,omitempty"`
	// Multiturn means the endpoint accepts a single message only.
	Multiturn bool `json:"multiturn,omitempty"`
	// Schema means the endpoint cannot constrain output to a JSON schema; the
	// shape has to be asked for in the prompt.
	Schema bool `json:"schema,omitempty"`
	// SchemaWithTools means it can constrain output, but not in the same
	// request that offers tools.
	SchemaWithTools bool `json:"schema_with_tools,omitempty"`
}

// Stage is where a model sits in its vendor's lifecycle.
//
// A retired model is kept in the catalog rather than deleted so that a caller
// still pointing at one gets told what happened and what to move to. Deleting
// the entry turns a clear "retired on 2026-02-19, use claude-sonnet-5" into an
// opaque 404 from the provider.
type Stage string

const (
	// StageStable is the default: generally available and supported.
	StageStable Stage = ""
	// StagePreview is available but subject to change without notice.
	StagePreview Stage = "preview"
	// StageDeprecated still serves requests but has an announced end date.
	StageDeprecated Stage = "deprecated"
	// StageRetired no longer serves requests. Validate refuses to send to one.
	StageRetired Stage = "retired"
)

// Available reports whether the model still serves requests.
func (s Stage) Available() bool { return s != StageRetired }

// Available returns the models that still serve requests, dropping retired
// ones. A model picker wants this; a lookup by ID does not, because answering
// "that one was retired, use this instead" needs the entry to still be there.
func Available(models []Model) []Model {
	out := make([]Model, 0, len(models))
	for _, m := range models {
		if m.Stage.Available() {
			out = append(out, m)
		}
	}
	return out
}

func cloneModel(model Model) Model {
	out := model
	out.Input = slices.Clone(model.Input)
	out.Reasoning = slices.Clone(model.Reasoning)
	out.Pricing.Tiers = slices.Clone(model.Pricing.Tiers)
	out.SamplingParams = cloneStringMap(model.SamplingParams)
	out.Headers = maps.Clone(model.Headers)
	out.Compat = model.Compat
	return out
}

// Clone returns a model snapshot whose mutable fields do not alias m.
// A Model handed to a caller is always one of these: the catalog's rows are a
// package-level table, and a client's model is its own, so returning either
// directly would let one caller corrupt what every other one reads.
func (m Model) Clone() Model { return cloneModel(m) }

func cloneModels(models []Model) []Model {
	out := make([]Model, len(models))
	for i, model := range models {
		out[i] = cloneModel(model)
	}
	return out
}
