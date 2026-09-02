package ai

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"
)

// API is a wire protocol — the request/response shape an endpoint speaks.
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

// anthropicFamily reports whether this protocol carries an Anthropic Messages
// body. Vertex differs only in how the client authenticates and where it
// points, so every rule about that body has to name both.
func (a API) anthropicFamily() bool {
	return a == APIAnthropicMessages || a == APIAnthropicVertex
}

// VertexConfig is the deployment a Vertex-served model lives in. It is passed
// as Config.ProtocolConfig to the anthropic/vertex driver.
type VertexConfig struct {
	// Project is the GCP project ID serving the model.
	Project string
	// Region is the serving region. "global" is the usual choice and is what
	// an empty value resolves to.
	Region string
}

// ProtocolConfig marks VertexConfig as a driver's construction settings.
func (VertexConfig) ProtocolConfig() {}

// Modality is a kind of content a model accepts as input.
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

// plainModel is Model without its two methods, so the codec below can hand the
// ordinary fields to the standard encoder instead of recursing into itself.
type plainModel Model

// modelJSON is the wire shape: every Model field as the standard encoder sees
// it, with only Compat shadowed by its raw form. Embedding rather than
// restating the fields is what keeps the two from drifting apart.
type modelJSON struct {
	plainModel
	Compat json.RawMessage `json:"compat,omitempty"`
}

// MarshalJSON writes the model, compat included.
func (m Model) MarshalJSON() ([]byte, error) {
	out := modelJSON{plainModel: plainModel(m)}
	// The shadowed field is the one the encoder never reaches; clearing it says
	// so, rather than leaving a second copy of compat to wonder about.
	out.plainModel.Compat = nil
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
func (m *Model) UnmarshalJSON(data []byte) error {
	var in modelJSON
	if err := json.Unmarshal(data, &in); err != nil {
		return err
	}
	*m = Model(in.plainModel)
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
	out.SamplingParams = maps.Clone(model.SamplingParams)
	out.Headers = maps.Clone(model.Headers)
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
