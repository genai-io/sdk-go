// Package catalog is the vendor and model directory, as data.
//
// A vendor here is a row, not a package. What distinguishes DeepSeek from
// Moonshot from Ollama is a base URL, an environment variable, a reasoning
// dialect and a list of models — never Go code — because all three serve the
// OpenAI Chat Completions protocol that one driver already speaks. Adding an
// OpenAI-compatible endpoint is an entry in vendors.go.
//
// The entries are a starting point, not an authority. Vendors ship models
// faster than any vendored table is refreshed, so an unknown model ID still
// resolves: it inherits its vendor's protocol, endpoint and dialect, and the
// driver's live listing fills in what the endpoint publishes about it.
//
// Every vendor records the date its figures were last checked against that
// vendor's documentation, because the failure mode here is silent — a
// two-year-old context window reads exactly like a fresh one. Stale reports
// entries that have aged out. Where a vendor publishes no per-model limits at
// all, the entry says so in its Note and reports zero rather than carrying a
// number nobody checked.
//
//	model, err := catalog.Model("deepseek/deepseek-v4-pro")
//	model, err := catalog.Model("claude-opus-4-6")  // unambiguous, vendor inferred
//
// Nothing in this package reads the environment. KeyEnv and BaseURLEnv name
// the variables a vendor conventionally uses; package llm/auth is what reads
// them, and only if you import it.
package catalog

import (
	"fmt"
	"maps"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Vendor is one provider of models.
type Vendor struct {
	// ID is the short lowercase key used in a "vendor/model" reference.
	ID string
	// DisplayName is the vendor's name as it should be shown.
	DisplayName string
	// Order sorts vendors for display; lower comes first.
	Order int

	// API is the wire protocol this vendor's endpoint speaks — the field that
	// decides which driver handles its models.
	API llm.API

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
	// KeyEnv names the environment variables that conventionally hold the
	// credential, most preferred first. Empty for endpoints needing none.
	KeyEnv []string

	// DeploymentEnv names the environment variables that carry a
	// deployment-scoped setting rather than a credential — a Vertex project
	// and region, for instance. auth reads them into Config.Native.
	DeploymentEnv map[string]string

	// Input lists the content kinds this vendor's models accept, for models
	// that do not declare their own. Empty means text only.
	Input []llm.Modality

	// Reasoning is the ladder for models that do not declare their own,
	// ordered least to most effort.
	Reasoning []llm.ReasoningLevel

	// Compat is the protocol behavior copied onto every model that does not
	// declare its own — one of llm.AnthropicCompat, llm.OpenAIChatCompat,
	// llm.OpenAIResponsesCompat or llm.GoogleCompat, by value.
	Compat any

	// SamplingParams are default sampling parameters for this vendor's models.
	SamplingParams map[string]any

	// Headers are sent with every request to this vendor.
	Headers map[string]string

	// Models is the known catalog. It is not exhaustive: Model resolves an
	// unlisted ID against the vendor's defaults.
	Models []llm.Model

	// Infer fills in what the Models table does not state for an ID — usually
	// limits, sometimes reasoning support. Several vendors encode the context
	// window in the model ID itself ("kimi-...-128k", "glm-5.2-...") and
	// publish nothing through their API, which no static table keeps up with.
	//
	// It runs on every resolved model, after vendor defaults, and by
	// convention only fills fields that are still zero.
	Infer func(llm.Model) llm.Model

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
var NoReasoning = []llm.ReasoningLevel{}

// NeedsDeployment reports whether this vendor requires deployment-scoped
// configuration — a cloud project, a region — beyond a credential.
func (v Vendor) NeedsDeployment() bool { return len(v.DeploymentEnv) > 0 }

// Model resolves a model ID against this vendor, whether or not it is listed.
func (v Vendor) Model(id string) llm.Model {
	for _, m := range v.Models {
		if strings.EqualFold(m.ID, id) {
			return v.decorate(m)
		}
	}
	return v.decorate(llm.Model{ID: id})
}

// ModelList returns the vendor's known models, fully decorated.
func (v Vendor) ModelList() []llm.Model {
	out := make([]llm.Model, len(v.Models))
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
func (v Vendor) decorate(m llm.Model) llm.Model {
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
		m.Pricing.Currency = llm.USD
	}
	// A retired model is a signpost, not an offer: leaving its limits at zero
	// keeps it from looking usable in a picker that only reads the numbers.
	if v.Infer != nil && m.Stage.Available() {
		m = v.Infer(m)
	}
	m.Input = slices.Clone(m.Input)
	m.Reasoning = slices.Clone(m.Reasoning)
	m.Pricing.Tiers = slices.Clone(m.Pricing.Tiers)
	m.SamplingParams = maps.Clone(m.SamplingParams)
	m.Headers = maps.Clone(m.Headers)
	return m
}

// Stale returns the vendors whose entries have not been verified against
// their vendor's documentation within the given age, oldest first. Pass the
// current time explicitly so a caller decides what "now" means — a build
// checking freshness in CI and a runtime warning want different clocks.
//
// A vendor with no Verified date, or an unparseable one, is always reported.
func Stale(now time.Time, age time.Duration) []Vendor {
	var out []Vendor
	for _, v := range All() {
		checked, err := time.Parse("2006-01-02", v.Verified)
		if err != nil || now.Sub(checked) > age {
			out = append(out, v)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Verified < out[j].Verified })
	return out
}

// All returns every vendor, in display order.
func All() []Vendor {
	out := make([]Vendor, len(vendors))
	copy(out, vendors)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Order < out[j].Order })
	return out
}

// Find returns the vendor with the given ID.
func Find(id string) (Vendor, bool) {
	for _, v := range vendors {
		if strings.EqualFold(v.ID, id) {
			return v, true
		}
	}
	return Vendor{}, false
}

// Model resolves a model reference.
//
// The canonical form is "vendor/id" — note that a model ID may itself contain
// a slash ("mimo/xiaomi/mimo-v2.5-pro"), so only the first segment is the
// vendor. A bare ID is resolved by searching every vendor's catalog; that
// succeeds only when exactly one vendor lists it. An ID served by several
// vendors (deepseek-v4-flash, direct and via SenseNova) is reported as
// ambiguous rather than resolved to whichever happened to be listed first.
func Model(ref string) (llm.Model, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return llm.Model{}, fmt.Errorf("catalog: empty model reference")
	}

	if vendorID, id, ok := strings.Cut(ref, "/"); ok {
		if v, found := Find(vendorID); found {
			if id == "" {
				return llm.Model{}, fmt.Errorf("catalog: reference %q names vendor %q with no model", ref, vendorID)
			}
			return v.Model(id), nil
		}
	}

	var matches []llm.Model
	var direct []llm.Model
	for _, v := range All() {
		for _, m := range v.Models {
			if !strings.EqualFold(m.ID, ref) {
				continue
			}
			matches = append(matches, v.decorate(m))
			if !v.NeedsDeployment() {
				direct = append(direct, v.decorate(m))
			}
		}
	}
	// A vendor that needs deployment configuration — a cloud project, a region
	// — is never what a bare model name means. Someone typing
	// "claude-opus-5" wants the first-party API; reaching it through Vertex or
	// a private cloud deployment is a deliberate choice, and naming the vendor
	// is how that choice is made. Without this rule, adding one alternate
	// hosting vendor would make every model it serves ambiguous.
	if len(direct) > 0 {
		matches = direct
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return llm.Model{}, &UnknownModelError{Ref: ref}
	default:
		refs := make([]string, len(matches))
		for i, m := range matches {
			refs[i] = m.String()
		}
		return llm.Model{}, &AmbiguousModelError{Ref: ref, Candidates: refs}
	}
}

// Models returns every known model across all vendors, in vendor display
// order, retired ones included. Filter with llm.Available for a picker.
func Models() []llm.Model {
	var out []llm.Model
	for _, v := range All() {
		out = append(out, v.ModelList()...)
	}
	return out
}

// UnknownModelError reports a bare model ID no vendor lists. Qualify it with a
// vendor ("deepseek/some-new-model") to use a model the catalog has not caught
// up with.
type UnknownModelError struct{ Ref string }

func (e *UnknownModelError) Error() string {
	return fmt.Sprintf("catalog: no vendor lists model %q; qualify it as \"vendor/%s\"", e.Ref, e.Ref)
}

// AmbiguousModelError reports a bare model ID served by more than one vendor.
type AmbiguousModelError struct {
	Ref        string
	Candidates []string
}

func (e *AmbiguousModelError) Error() string {
	return fmt.Sprintf("catalog: model %q is served by several vendors (%s); qualify it",
		e.Ref, strings.Join(e.Candidates, ", "))
}
