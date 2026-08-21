package llm

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
)

// A Model has to survive being written to a session file and read back.
//
// Compat is an `any`, and the encoding/json default turns it into a
// map[string]any on the way in — so CompatOf yields the zero value and the
// model silently loses its protocol dialect. A DeepSeek model reloaded that
// way stops sending its "reasoning off" field, and reasoning stays on with
// nothing reporting it. That is the same failure the provider merge is written
// to avoid, and it has to be closed here too.
//
// The API field is the discriminator: it already says which protocol the model
// speaks, so it says which compat type belongs to it.

// compatRegistry maps a protocol to the compat type it expects.
//
// The type, not a decoder: it is what both jobs need. Reading a model back
// needs to rebuild the value as that type, and validating one needs to catch a
// model carrying another protocol's compat — which CompatOf cannot report,
// because a type assertion that fails yields the zero value and the model
// simply behaves as though it had no dialect at all.
var compatRegistry = struct {
	mu sync.RWMutex
	m  map[API]reflect.Type
}{m: map[API]reflect.Type{
	APIAnthropicMessages: reflect.TypeFor[AnthropicCompat](),
	APIAnthropicVertex:   reflect.TypeFor[AnthropicCompat](),
	APIOpenAIChat:        reflect.TypeFor[OpenAIChatCompat](),
	APIOpenAIResponses:   reflect.TypeFor[OpenAIResponsesCompat](),
	APIGoogleGenAI:       reflect.TypeFor[GoogleCompat](),
}}

// RegisterCompat declares the compat type a protocol this package does not
// define uses. A driver package for a custom protocol calls it from init,
// alongside RegisterAPI.
//
//	llm.RegisterCompat[MyCompat](myAPI)
//
// It takes the type rather than a decode function because every caller wants
// the same decoding — and a type is the thing that also makes a mismatched
// compat detectable, which a decode function is not.
func RegisterCompat[T any](api API) {
	compatRegistry.mu.Lock()
	defer compatRegistry.mu.Unlock()
	compatRegistry.m[api] = reflect.TypeFor[T]()
}

// compatType reports the compat type registered for a protocol.
func compatType(api API) (reflect.Type, bool) {
	compatRegistry.mu.RLock()
	defer compatRegistry.mu.RUnlock()
	t, ok := compatRegistry.m[api]
	return t, ok
}

// checkCompat reports a compat value that does not belong to the model's
// protocol.
//
// This is the one failure CompatOf cannot surface. Setting an OpenAIChatCompat
// on an Anthropic model leaves CompatOf[AnthropicCompat] returning the zero
// value, so the model runs with first-party defaults and nothing says the
// dialect was ignored — the same silent-downgrade shape that UnmarshalJSON
// refuses. A protocol with no registered type is left alone: there is nothing
// to compare against.
func (m Model) checkCompat() error {
	if m.Compat == nil {
		return nil
	}
	want, ok := compatType(m.API)
	if !ok {
		return nil
	}
	if got := reflect.TypeOf(m.Compat); got != want {
		return &Error{Kind: KindUnsupported, Message: fmt.Sprintf(
			"model %s speaks %s, but carries %s as its compat; %s expects %s, "+
				"and the mismatch would be ignored rather than reported",
			m, m.API, got, m.API, want)}
	}
	return nil
}

// modelWire is Model without the custom marshalling, so the two methods below
// can delegate the ordinary fields to the standard encoder.
type modelWire struct {
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
	w := modelWire{
		ID: m.ID, API: m.API, Name: m.Name, Vendor: m.Vendor, BaseURL: m.BaseURL,
		ContextWindow: m.ContextWindow, MaxOutput: m.MaxOutput,
		Input: m.Input, Reasoning: m.Reasoning, Pricing: m.Pricing,
		Unsupported: m.Unsupported, Stage: m.Stage, Replacement: m.Replacement,
		SamplingParams: m.SamplingParams, Headers: m.Headers,
	}
	if m.Compat != nil {
		raw, err := json.Marshal(m.Compat)
		if err != nil {
			return nil, fmt.Errorf("llm: encoding compat for model %s: %w", m, err)
		}
		w.Compat = raw
	}
	return json.Marshal(w)
}

// UnmarshalJSON reads a model back, rebuilding compat as the concrete type its
// protocol uses.
//
// A protocol with no registered decoder is an error rather than a silent
// downgrade: a model whose quirks were dropped looks fine and misbehaves
// later, which is the worse of the two outcomes.
func (m *Model) UnmarshalJSON(data []byte) error {
	var w modelWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	*m = Model{
		ID: w.ID, API: w.API, Name: w.Name, Vendor: w.Vendor, BaseURL: w.BaseURL,
		ContextWindow: w.ContextWindow, MaxOutput: w.MaxOutput,
		Input: w.Input, Reasoning: w.Reasoning, Pricing: w.Pricing,
		Unsupported: w.Unsupported, Stage: w.Stage, Replacement: w.Replacement,
		SamplingParams: w.SamplingParams, Headers: w.Headers,
	}
	if len(w.Compat) == 0 || string(w.Compat) == "null" {
		return nil
	}

	t, ok := compatType(w.API)
	if !ok {
		return fmt.Errorf("llm: model %s carries compat for unregistered protocol %q; "+
			"call RegisterCompat for it, or the model would load with its quirks silently dropped", m, w.API)
	}
	compat := reflect.New(t)
	if err := json.Unmarshal(w.Compat, compat.Interface()); err != nil {
		return fmt.Errorf("llm: decoding compat for model %s: %w", m, err)
	}
	m.Compat = compat.Elem().Interface()
	return nil
}
