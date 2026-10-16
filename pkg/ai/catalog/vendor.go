package catalog

import (
	"maps"
	"slices"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"
)

// Vendor is one endpoint of models.
type Vendor struct {
	// ID is the short lowercase key used in a "vendor/model" reference.
	ID string
	// DisplayName is the vendor's name as it should be shown.
	DisplayName string

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

	// VertexEnv names the variables that say where a Vertex AI deployment
	// runs. They name a place, not a credential: Vertex authenticates with
	// Google Application Default Credentials. Zero for a vendor not served
	// through Vertex.
	VertexEnv VertexEnv

	// Compat is the protocol behavior copied onto every model that does not
	// declare its own — one of ai.AnthropicCompat, ai.OpenAIChatCompat,
	// ai.OpenAIResponsesCompat or ai.GoogleCompat, by value.
	Compat any

	// Headers are sent with every request to this vendor.
	Headers map[string]string

	// Note records anything a caller has to know before choosing this vendor,
	// such as a credential that only an interactive login can produce, or a
	// figure that could not be verified.
	Note string
}

func (v Vendor) clone() Vendor {
	out := v
	out.KeyEnv = slices.Clone(v.KeyEnv)
	out.Headers = maps.Clone(v.Headers)
	// Compat needs no clone: every compat is a struct held in an interface, so
	// the field copy above is already a copy of the value.
	return out
}

// Model decorates a model ID with this vendor's protocol facts. It knows
// nothing about the model itself: limits, prices, modalities and reasoning
// efforts are the caller's to set.
func (v Vendor) Model(id string) ai.Model { return v.Resolve(ai.Model{ID: id}) }

// Resolve stamps a model with this vendor's protocol facts, overwriting
// nothing the model already states. A live listing is what needs it: a host
// reports an ID and a name and never a protocol quirk.
func (v Vendor) Resolve(m ai.Model) ai.Model {
	m = m.Clone()
	m.Vendor = v.ID
	if m.API == "" {
		m.API = v.API
	}
	if m.BaseURL == "" {
		m.BaseURL = v.BaseURL
	}
	if m.Name == "" {
		m.Name = m.ID
	}
	if m.Compat == nil && m.API == v.API {
		m.Compat = v.Compat
	}
	if m.Headers == nil {
		m.Headers = maps.Clone(v.Headers)
	}
	if m.Pricing.Known() && m.Pricing.Currency == "" {
		m.Pricing.Currency = ai.USD
	}
	return m
}

// KeyFromEnv returns the first non-empty value among KeyEnv, read through
// getenv — os.Getenv, or a lookup that also consults the caller's own secret
// store. It is empty, with no error, for a vendor that takes no key.
func (v Vendor) KeyFromEnv(getenv func(string) string) (string, error) {
	for _, name := range v.KeyEnv {
		if value := strings.TrimSpace(getenv(name)); value != "" {
			return value, nil
		}
	}
	if len(v.KeyEnv) > 0 {
		return "", &MissingEnvError{Vendor: v.ID, EnvVars: v.KeyEnv, Note: v.Note}
	}
	return "", nil
}

// BaseURLFromEnv returns BaseURL, or the override BaseURLEnv names, read
// through getenv, with BaseURLSuffix added where it is missing. It fails for a
// vendor that RequiresBaseURL when the override is unset.
func (v Vendor) BaseURLFromEnv(getenv func(string) string) (string, error) {
	override := ""
	if v.BaseURLEnv != "" {
		override = strings.TrimRight(strings.TrimSpace(getenv(v.BaseURLEnv)), "/")
	}
	if override == "" {
		if v.RequiresBaseURL {
			return "", &MissingEnvError{Vendor: v.ID, EnvVars: []string{v.BaseURLEnv}, Note: v.Note}
		}
		return v.BaseURL, nil
	}
	if v.BaseURLSuffix != "" && !strings.HasSuffix(override, v.BaseURLSuffix) {
		override += v.BaseURLSuffix
	}
	return override, nil
}

// VertexFromEnv reads the deployment VertexEnv names, through getenv, into
// the value ai.Config.Vertex takes. It is nil for a vendor not served through
// Vertex, and fails when the project is unset: it has no default worth
// guessing.
func (v Vendor) VertexFromEnv(getenv func(string) string) (*ai.VertexConfig, error) {
	if v.VertexEnv == (VertexEnv{}) {
		return nil, nil
	}
	project := strings.TrimSpace(getenv(v.VertexEnv.Project))
	if project == "" {
		return nil, &MissingEnvError{Vendor: v.ID, EnvVars: []string{v.VertexEnv.Project}, Note: v.Note}
	}
	region := ""
	if v.VertexEnv.Region != "" {
		region = strings.TrimSpace(getenv(v.VertexEnv.Region))
	}
	return &ai.VertexConfig{Project: project, Region: region}, nil
}

// VertexEnv names the variables holding a Vertex AI deployment.
type VertexEnv struct {
	// Project names the variable holding the Google Cloud project. Required.
	Project string
	// Region names the variable holding the region. Unset means the driver's
	// default.
	Region string
}

// Endpoint builds a live endpoint for this vendor. Its models come from
// cfg.Models and the host's own listing.
func (v Vendor) Endpoint(cfg endpoint.Config) *endpoint.Endpoint {
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
	return endpoint.New(cfg)
}
