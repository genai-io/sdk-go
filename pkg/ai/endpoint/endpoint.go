package endpoint

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Endpoint is a configured endpoint: the models it serves, the credential to
// reach it with, and the protocol that talks to it.
//
// Reading its model list is synchronous and cannot fail — it returns what is
// known now, which before the first Refresh is the static baseline it was
// built with. Fetching is a separate, explicit verb. That split is what lets a
// model picker render immediately and refresh behind the user, instead of
// blocking on a round trip that a dead endpoint can hang.
//
// A Endpoint is safe for concurrent use.
type Endpoint struct {
	cfg Config

	mu      sync.RWMutex
	overlay []ai.Model
}

// Config describes one endpoint.
type Config struct {
	// ID is the short key this endpoint is known by, e.g. "deepseek".
	ID string
	// Name is how it should be shown. Defaults to ID.
	Name string

	// BaseURL, APIKey, HTTPClient and Headers are applied to every model this
	// endpoint opens. A model's own BaseURL and Headers layer underneath.
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	Headers    map[string]string

	// Native is passed through to every Config this endpoint builds — see
	// Config.Native.
	Native any

	// Models is the static baseline: what is known without asking the
	// endpoint. It may be empty for an endpoint that only has a live listing.
	Models []ai.Model

	// API is the protocol this endpoint speaks. It is only needed when Models
	// is empty, so that Refresh knows which driver to ask; otherwise it is
	// taken from the baseline.
	API ai.API

	// Fetch retrieves the live model list. Nil means the default: open a
	// driver for this endpoint's protocol and call its Models. Set it for an
	// endpoint whose listing lives somewhere other than the protocol's own
	// models call.
	Fetch func(ctx context.Context, e *Endpoint) ([]ai.Model, error)
}

// New builds an endpoint from its parts.
func New(cfg Config) *Endpoint {
	cfg.Headers = maps.Clone(cfg.Headers)
	cfg.Models = cloneAll(cfg.Models)
	if cfg.Name == "" {
		cfg.Name = cfg.ID
	}
	if cfg.API == "" && len(cfg.Models) > 0 {
		cfg.API = cfg.Models[0].API
	}
	return &Endpoint{cfg: cfg}
}

// ID returns the endpoint's key.
func (e *Endpoint) ID() string { return e.cfg.ID }

// Name returns the endpoint's display name.
func (e *Endpoint) Name() string { return e.cfg.Name }

// API returns the protocol this endpoint speaks.
func (e *Endpoint) API() ai.API { return e.cfg.API }

// Models returns what is known now: the static baseline with the last fetched
// listing merged over it. It never blocks and never fails — before the first
// successful Refresh it is the baseline alone.
//
// The merge is field by field, not entry by entry. A listing carries what the
// endpoint publishes, which for most OpenAI-compatible vendors is an ID and
// nothing else; replacing a baseline entry wholesale would discard its
// pricing, its reasoning ladder and its protocol quirks, and a model stripped
// of its quirks stops working. So the endpoint wins on every field it stated,
// and the baseline fills the rest.
func (e *Endpoint) Models() []ai.Model {
	e.mu.RLock()
	overlay := cloneAll(e.overlay)
	e.mu.RUnlock()

	// Decorate the baseline as well as the listing. Every model an endpoint
	// hands out carries that endpoint's identity and protocol, whether it came
	// from the baseline, from a refresh, or from an ID nobody has listed —
	// otherwise Open would work for a model the endpoint has never heard of
	// and fail for one sitting in its own table.
	merged := make([]ai.Model, 0, len(e.cfg.Models))
	for _, m := range e.cfg.Models {
		merged = append(merged, e.decorate(m.Clone()))
	}
	for _, live := range overlay {
		i := slices.IndexFunc(merged, func(m ai.Model) bool { return strings.EqualFold(m.ID, live.ID) })
		if i < 0 {
			merged = append(merged, e.decorate(live))
			continue
		}
		merged[i] = MergeListing(merged[i], live)
	}
	return cloneAll(merged)
}

// Model looks one model up by ID. Unlike Models it also answers for an ID the
// endpoint has never heard of, by decorating it with the endpoint's protocol
// and endpoint — an unlisted model is nearly always one newer than the
// catalog, not one that does not exist.
func (e *Endpoint) Model(id string) (ai.Model, bool) {
	for _, m := range e.Models() {
		if strings.EqualFold(m.ID, id) {
			return m, true
		}
	}
	return e.decorate(ai.Model{ID: id}), false
}

// Refresh fetches the live model list and merges it in. A failure leaves the
// previous list untouched, so an endpoint that went down keeps serving what it
// last knew.
//
// A endpoint with no way to list — no Fetch and no protocol — is a no-op.
func (e *Endpoint) Refresh(ctx context.Context) error {
	fetch := e.cfg.Fetch
	if fetch == nil {
		if e.cfg.API == "" {
			return nil
		}
		fetch = defaultFetch
	}
	models, err := fetch(ctx, e)
	if err != nil {
		return err
	}
	e.mu.Lock()
	e.overlay = cloneAll(models)
	e.mu.Unlock()
	return nil
}

// defaultFetch opens a client for the endpoint's protocol and asks it.
//
// The driver needs a model to be constructed with, and listing does not depend
// on which one, so a placeholder stands in when the endpoint has no baseline
// to borrow from. Client.Models is what answers, rather than the driver
// directly, so a protocol with no listing endpoint reports it the same way
// here as it does to any other caller.
func defaultFetch(ctx context.Context, e *Endpoint) ([]ai.Model, error) {
	probe := ai.Model{ID: "-", API: e.cfg.API}
	if len(e.cfg.Models) > 0 {
		probe = e.cfg.Models[0]
	}
	client, err := ai.Open(e.ConfigFor(probe))
	if err != nil {
		return nil, err
	}
	return client.Models(ctx)
}

// ConfigFor builds the ai.Config for opening one of this endpoint's models.
func (e *Endpoint) ConfigFor(m ai.Model) ai.Config {
	return ai.Config{
		Model:      m.Clone(),
		APIKey:     e.cfg.APIKey,
		BaseURL:    e.cfg.BaseURL,
		HTTPClient: e.cfg.HTTPClient,
		Headers:    maps.Clone(e.cfg.Headers),
		Native:     e.cfg.Native,
	}
}

// Open returns a client for one of this endpoint's models. An ID the endpoint
// does not list still opens, carrying the endpoint's protocol and endpoint.
func (e *Endpoint) Open(modelID string, opts ...ai.Option) (*ai.Client, error) {
	// The bool is discarded on purpose: an unlisted ID is fine — Model
	// decorates it with this endpoint's protocol. What is not fine is having
	// no protocol to decorate it with.
	m, _ := e.Model(modelID)
	if m.API == "" {
		return nil, &ai.Error{Kind: ai.KindInvalidRequest, Message: fmt.Sprintf(
			"ai: endpoint %q states no protocol, so it cannot open model %q", e.cfg.ID, modelID)}
	}
	return ai.Open(e.ConfigFor(m), opts...)
}

// decorate stamps a model with the endpoint's identity and protocol. It only
// fills an API the model does not state, so a set of mixed-protocol models
// keeps its own.
func (e *Endpoint) decorate(m ai.Model) ai.Model {
	m.Vendor = e.cfg.ID
	if m.API == "" {
		m.API = e.cfg.API
	}
	return m
}

// MergeListing layers a live model listing over a known baseline entry.
//
// The listing wins on every field it stated; everything it left zero comes
// from the baseline. That asymmetry is the whole rule: an endpoint is
// authoritative about which models exist and about any figure it reported,
// while a baseline knows the pricing, the reasoning ladder and the protocol
// quirks that no listing publishes — and a model stripped of its quirks stops
// working.
//
// Endpoint uses it to merge a refresh over its static models; catalog.Enrich
// uses it to merge the vendored table under a listing. Both directions are the
// same operation, so they are the same function: a new Model field is handled
// once or not at all.
func MergeListing(base, live ai.Model) ai.Model {
	out := base.Clone()
	live = live.Clone()
	if live.Name != "" && live.Name != live.ID {
		out.Name = live.Name
	}
	if live.ContextWindow > 0 {
		out.ContextWindow = live.ContextWindow
	}
	if live.MaxOutput > 0 {
		out.MaxOutput = live.MaxOutput
	}
	if len(live.Input) > 0 {
		out.Input = live.Input
	}
	if len(live.Reasoning) > 0 {
		out.Reasoning = live.Reasoning
	}
	if live.Pricing.Known() {
		out.Pricing = live.Pricing
	}
	return out
}
