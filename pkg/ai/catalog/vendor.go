package catalog

import (
	"maps"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"
)

// A vendor is a row, not a package.
//
// What distinguishes DeepSeek from Moonshot from Ollama is a base URL, an
// environment variable, a reasoning dialect and a list of models — never Go
// code, because all three serve the OpenAI Chat Completions protocol that one
// driver already speaks.
//
// An entry states only what differs from its vendor's defaults, and decorate
// fills in the rest. That is what keeps a table of thirty models readable.

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
	//
	// Without it such an entry is a quiet misroute rather than an error: the
	// endpoint falls back to the protocol owner's host, and an Azure or
	// Bedrock credential is presented to api.openai.com.
	RequiresBaseURL bool

	// KeyEnv names the environment variables that conventionally hold the
	// credential, most preferred first. Empty for endpoints needing none.
	KeyEnv []string

	// DeploymentEnv names the environment variables that carry a
	// deployment-scoped setting rather than a credential — a Vertex project
	// and region, for instance. auth reads them into Config.Native.
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
	//
	// It runs on every resolved model, after vendor defaults, and by
	// convention only fills fields that are still zero.
	Infer func(ai.Model) ai.Model

	// Verified is when this entry was last checked against the vendor's own
	// published documentation, as YYYY-MM-DD.
	//
	// It is recorded because the failure mode of a vendored catalog is silent:
	// a stale context window or price reads exactly like a fresh one. A date
	// lets a caller — or a reviewer — see the age of what they are trusting,
	// and Stale reports entries that have drifted out of date.
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
//
// Slices and maps are cloned on the way out: the tables are package-level
// values shared by every caller, and a caller that appended to a returned
// ladder would corrupt the catalog for everyone else.
func (v Vendor) decorate(m ai.Model) ai.Model {
	m = m.Clone()
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
	// Clone inherited collections before Infer: an inference function is allowed
	// to edit its model argument, never the package-level catalog defaults.
	m = m.Clone()
	// A retired model is a signpost, not an offer: leaving its limits at zero
	// keeps it from looking usable in a picker that only reads the numbers.
	if v.Infer != nil && m.Stage.Available() {
		m = v.Infer(m)
	}
	return m.Clone()
}

// ResolveBaseURL applies an override to a vendor's endpoint.
//
// An empty override leaves the vendor default in place. An override is
// trimmed of its trailing slash and given the vendor's required path suffix if
// it lacks one, so the bare "http://localhost:11434" people paste for a local
// Ollama still reaches its OpenAI-compatible API.
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

// Endpoint builds a live endpoint for this vendor, seeded with its catalog
// models as the static baseline.
//
// The caller supplies the credential and any transport settings; the vendor
// supplies its identity, protocol, endpoint and models. Fields the caller
// leaves unset fall back to the vendor's — so passing a zero endpoint.Config
// yields an endpoint that can already list and open models, just without a key.
func (v Vendor) Endpoint(cfg endpoint.Config) *endpoint.Endpoint {
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
	return endpoint.New(cfg)
}
