package catalog

import (
	"maps"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/provider"
)

// Vendor is one endpoint of models.
type Vendor struct {
	// ID is the short lowercase key used in a "vendor/model" reference.
	ID string
	// DisplayName is the vendor's name as it should be shown.
	DisplayName string
	// Order sorts vendors for display; lower comes first.
	Order int

	// API is the wire protocol this vendor's endpoint speaks — the field that
	// decides which driver handles its models.
	API ai.API

	// BaseURL is the default endpoint. Empty means the driver's own default,
	// which is what the vendor that owns the protocol wants.
	BaseURL string
	// BaseURLEnv names the environment variable that conventionally overrides
	// BaseURL.
	BaseURLEnv string
	// BaseURLSuffix, when set, is appended to an override that lacks it. A
	// local Ollama serves its OpenAI-compatible API under /v1, and the URL
	// people have to hand is the bare host and port.
	BaseURLSuffix string
	// RequiresBaseURL marks a vendor that has no usable default endpoint —
	// one whose host names a tenant's own resource or region, so BaseURLEnv
	// must actually be set.
	RequiresBaseURL bool

	// KeyEnv names the environment variables that conventionally hold the
	// credential, most preferred first. Empty for endpoints needing none.
	KeyEnv []string

	// DeploymentEnv names the environment variables that carry a
	// deployment-scoped setting rather than a credential — a Vertex project
	// and region, for instance. auth reads them into Config.ProtocolConfig.
	DeploymentEnv map[string]string

	// Input lists the content kinds this vendor's models accept, for models
	// that do not declare their own. Empty means text only.
	Input []ai.Modality

	// Reasoning is the ladder for models that do not declare their own,
	// ordered least to most effort.
	Reasoning []ai.ReasoningLevel

	// Compat is the protocol behavior copied onto every model that does not
	// declare its own — one of ai.AnthropicCompat, ai.OpenAIChatCompat,
	// ai.OpenAIResponsesCompat or ai.GoogleCompat, by value.
	Compat any

	// SamplingParams are default sampling parameters for this vendor's models.
	SamplingParams map[string]any

	// Headers are sent with every request to this vendor.
	Headers map[string]string

	// Models is the known catalog. It is not exhaustive: Model resolves an
	// unlisted ID against the vendor's defaults.
	Models []ai.Model

	// Infer fills in what the Models table does not state for an ID — usually
	// limits, sometimes reasoning support. Several vendors encode the context
	// window in the model ID itself ("kimi-...-128k", "glm-5.2-...") and
	// publish nothing through their API, which no static table keeps up with.
	Infer func(ai.Model) ai.Model

	// Verified is when this entry was last checked against the vendor's own
	// published documentation, as YYYY-MM-DD.
	Verified string

	// Note records anything a caller has to know before choosing this vendor,
	// such as a credential that only an interactive login can produce, or a
	// figure that could not be verified.
	Note string
}

// NoReasoning marks a catalog entry as a model that does not reason, which is
// different from one that simply does not say — an omitted Reasoning inherits
// the vendor default.
var NoReasoning = []ai.ReasoningLevel{}

// NeedsDeployment reports whether this vendor requires deployment-scoped
// configuration — a cloud project, a region — beyond a credential.
func (v Vendor) NeedsDeployment() bool { return len(v.DeploymentEnv) > 0 }

func (v Vendor) clone() Vendor {
	out := v
	out.KeyEnv = slices.Clone(v.KeyEnv)
	out.DeploymentEnv = maps.Clone(v.DeploymentEnv)
	out.Input = slices.Clone(v.Input)
	out.Reasoning = slices.Clone(v.Reasoning)
	out.Headers = maps.Clone(v.Headers)
	carrier := (ai.Model{Compat: v.Compat, SamplingParams: v.SamplingParams}).Clone()
	out.Compat = carrier.Compat
	out.SamplingParams = carrier.SamplingParams
	out.Models = make([]ai.Model, len(v.Models))
	for i, model := range v.Models {
		out.Models[i] = model.Clone()
	}
	return out
}

// Model resolves a model ID against this vendor, whether or not it is listed.
func (v Vendor) Model(id string) ai.Model {
	for _, m := range v.Models {
		if strings.EqualFold(m.ID, id) {
			return v.decorate(m)
		}
	}
	return v.decorate(ai.Model{ID: id})
}

// ModelList returns the vendor's known models, fully decorated.
func (v Vendor) ModelList() []ai.Model {
	out := make([]ai.Model, len(v.Models))
	for i, m := range v.Models {
		out[i] = v.decorate(m)
	}
	return out
}

// decorate fills in everything a catalog entry inherits from its vendor. An
// entry only spells out what differs from the vendor's defaults, so a table of
// thirty models stays readable.
func (v Vendor) decorate(m ai.Model) ai.Model {
	m.Vendor = v.ID
	m.API = v.API
	if m.BaseURL == "" {
		m.BaseURL = v.BaseURL
	}
	if m.Name == "" {
		m.Name = m.ID
	}
	if m.Input == nil {
		m.Input = v.Input
	}
	// A nil ladder means "not stated, inherit"; an explicitly empty one
	// (NoReasoning) means "this model does not reason", which a vendor default
	// must not overwrite.
	if m.Reasoning == nil {
		m.Reasoning = v.Reasoning
	}
	if m.Compat == nil {
		m.Compat = v.Compat
	}
	if m.SamplingParams == nil {
		m.SamplingParams = v.SamplingParams
	}
	if m.Headers == nil {
		m.Headers = v.Headers
	}
	if m.Pricing.Known() && m.Pricing.Currency == "" {
		m.Pricing.Currency = ai.USD
	}
	// One clone, here: everything above is a whole-field assignment, so this
	// is the first point at which m can share a slice or map with anyone —
	// either the caller's model or the package-level vendor defaults just
	// inherited. Infer is allowed to edit its argument, never those.
	m = m.Clone()
	// A retired model is a signpost, not an offer: leaving its limits at zero
	// keeps it from looking usable in a picker that only reads the numbers.
	if v.Infer != nil && m.Stage.Available() {
		m = v.Infer(m)
	}
	return m.Clone()
}

// ResolveBaseURL applies an override to a vendor's endpoint.
func (v Vendor) ResolveBaseURL(override string) string {
	override = strings.TrimSpace(override)
	if override == "" {
		return v.BaseURL
	}
	override = strings.TrimRight(override, "/")
	if v.BaseURLSuffix != "" && !strings.HasSuffix(override, v.BaseURLSuffix) {
		override += v.BaseURLSuffix
	}
	return override
}

// Provider builds a live provider for this vendor, seeded with its catalog
// models as the static baseline.
func (v Vendor) Provider(cfg provider.Config) *provider.Provider {
	cfg.ID = v.ID
	if cfg.Name == "" {
		cfg.Name = v.DisplayName
	}
	cfg.API = v.API
	if cfg.BaseURL == "" {
		cfg.BaseURL = v.BaseURL
	}
	if cfg.Models == nil {
		cfg.Models = v.ModelList()
	}
	if cfg.Headers == nil {
		cfg.Headers = v.Headers
	}
	return provider.New(cfg)
}
