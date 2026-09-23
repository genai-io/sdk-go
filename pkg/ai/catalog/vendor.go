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
	// and region, for instance.
	DeploymentEnv map[string]string

	// Deployment turns those variables into the value this vendor's driver
	// expects as ai.Config.ProtocolConfig, and says which one is missing when the
	// endpoint cannot run without it — a fact about the row, not about auth. It
	// is handed DeploymentEnv, so a row names each variable once. Nil for a
	// vendor that needs no deployment.
	Deployment func(vars map[string]string, env func(string) string) (ai.ProtocolConfig, error)

	// Compat is the protocol behavior copied onto every model that does not
	// declare its own — one of ai.AnthropicCompat, ai.OpenAIChatCompat,
	// ai.OpenAIResponsesCompat or ai.GoogleCompat, by value.
	Compat any

	// SamplingParams are default sampling parameters for this vendor's models.
	SamplingParams map[string]any

	// Headers are sent with every request to this vendor.
	Headers map[string]string

	// Verified is when this entry was last checked against the vendor's own
	// published documentation, as YYYY-MM-DD.
	Verified string

	// Note records anything a caller has to know before choosing this vendor,
	// such as a credential that only an interactive login can produce, or a
	// figure that could not be verified.
	Note string
}

// NeedsDeployment reports whether this vendor requires deployment-scoped
// configuration — a cloud project, a region — beyond a credential.
func (v Vendor) NeedsDeployment() bool { return len(v.DeploymentEnv) > 0 }

func (v Vendor) clone() Vendor {
	out := v
	out.KeyEnv = slices.Clone(v.KeyEnv)
	out.DeploymentEnv = maps.Clone(v.DeploymentEnv)
	out.Headers = maps.Clone(v.Headers)
	// Compat needs no clone: every compat is a struct held in an interface, so
	// the field copy above is already a copy of the value.
	out.SamplingParams = maps.Clone(v.SamplingParams)
	return out
}

// Model decorates a model ID with this vendor's protocol facts. It knows
// nothing about the model itself: limits, prices, modalities and reasoning
// efforts are the caller's to set.
func (v Vendor) Model(id string) ai.Model { return v.decorate(ai.Model{ID: id}) }

// Resolve stamps a model with this vendor's protocol facts, overwriting
// nothing the model already states. A live listing is what needs it: a host
// reports an ID and a name and never a protocol quirk.
func (v Vendor) Resolve(m ai.Model) ai.Model { return v.decorate(m) }

// decorate fills in what a model inherits from its vendor's protocol.
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
	if m.Compat == nil {
		m.Compat = v.Compat
	}
	if m.SamplingParams == nil {
		m.SamplingParams = maps.Clone(v.SamplingParams)
	}
	if m.Headers == nil {
		m.Headers = maps.Clone(v.Headers)
	}
	if m.Pricing.Known() && m.Pricing.Currency == "" {
		m.Pricing.Currency = ai.USD
	}
	return m
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

// Provider builds a live provider for this vendor. Its models come from
// cfg.Models and the endpoint's own listing.
func (v Vendor) Provider(cfg provider.Config) *provider.Provider {
	cfg.ID = v.ID
	if cfg.Name == "" {
		cfg.Name = v.DisplayName
	}
	cfg.API = v.API
	if cfg.BaseURL == "" {
		cfg.BaseURL = v.BaseURL
	}
	if cfg.Headers == nil {
		cfg.Headers = v.Headers
	}
	// Without this a live-listed model would arrive carrying nothing but its ID,
	// leaving the caller to resolve it through the catalog a second time.
	if cfg.Resolve == nil {
		cfg.Resolve = v.Resolve
	}
	return provider.New(cfg)
}
