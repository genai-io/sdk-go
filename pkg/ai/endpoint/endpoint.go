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

// Endpoint is one configured host: the models it serves, the credential to
// reach it with, and the protocol that talks to it.
type Endpoint struct {
	cfg Config

	mu sync.RWMutex
	// listing is what the endpoint reported the last time Refresh asked. It is
	// merged over cfg.Models on the way out, and is empty until the first
	// successful fetch.
	listing []ai.Model
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

	// Vertex is passed through to every Config this endpoint builds — see
	// ai.Config.Vertex.
	Vertex *ai.VertexConfig

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
	//
	// It is handed the *Endpoint it belongs to, so it can read the credential
	// and host already configured here.
	Fetch func(ctx context.Context, e *Endpoint) ([]ai.Model, error)

	// Resolve fills in what is known about a model from its ID alone — its
	// protocol quirks, and whatever the caller keeps about its models — which
	// a host's listing rarely carries; catalog.Vendor.Endpoint installs the
	// vendor's here. It must fill rather than replace: the host wins on any
	// figure it gave.
	Resolve func(ai.Model) ai.Model
}

// New builds an endpoint from its parts.
func New(cfg Config) *Endpoint {
	cfg.Headers = maps.Clone(cfg.Headers)
	// A list is snapshotted here, in Refresh and in Models — wherever it
	// crosses in or out — so a caller may keep mutating its own builders and a
	// refresh in another goroutine cannot rewrite a list someone is reading.
	cfg.Models = ai.CloneModels(cfg.Models)
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
func (e *Endpoint) Models() []ai.Model {
	e.mu.RLock()
	listing := ai.CloneModels(e.listing)
	e.mu.RUnlock()

	// Decorate the baseline as well as the listing. Every model an endpoint
	// hands out carries that endpoint's identity and protocol, whether it came
	// from the baseline, from a refresh, or from an ID nobody has listed —
	// otherwise Client would work for a model the endpoint has never heard of
	// and fail for one sitting in its own table.
	merged := make([]ai.Model, 0, len(e.cfg.Models))
	for _, m := range e.cfg.Models {
		merged = append(merged, e.decorate(m.Clone()))
	}
	for _, live := range listing {
		i := slices.IndexFunc(merged, func(m ai.Model) bool { return strings.EqualFold(m.ID, live.ID) })
		if i < 0 {
			live.API = listedAPI(e.cfg.API, live.API)
			merged = append(merged, e.decorate(live))
			continue
		}
		merged[i] = MergeListing(merged[i], live)
	}
	// No clone on the way out: every entry above is already a fresh value —
	// the baseline through Clone, the listing through CloneModels, a merged pair
	// through MergeListing, which clones both sides.
	return merged
}

// Model looks one model up by ID. Unlike Models it also answers for an ID the
// endpoint has never heard of, by decorating it with the endpoint's identity
// and protocol and asking Resolve what else the ID implies — an unlisted model
// is nearly always one newer than the catalog, not one that does not exist.
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
	e.listing = ai.CloneModels(models)
	e.mu.Unlock()
	return nil
}

// defaultFetch opens a client for the endpoint's protocol and asks it. The
// probe is decorated like any other model, so a driver that lists by a quirk
// of the host sees the host's compat.
func defaultFetch(ctx context.Context, e *Endpoint) ([]ai.Model, error) {
	probe := ai.Model{ID: "-"}
	if len(e.cfg.Models) > 0 {
		probe = e.cfg.Models[0].Clone()
	}
	client, err := ai.New(e.ConfigFor(e.decorate(probe)))
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
		Vertex:     e.cfg.Vertex,
	}
}

// Client returns a client for one of this endpoint's models. An ID this
// endpoint does not list still resolves, carrying its protocol and base URL.
func (e *Endpoint) Client(modelID string, opts ...ai.Option) (*ai.Client, error) {
	// The bool is discarded on purpose: an unlisted ID is fine — Model
	// decorates it with this endpoint's protocol. What is not fine is having
	// no protocol to decorate it with.
	m, _ := e.Model(modelID)
	if m.API == "" {
		return nil, &ai.Error{Kind: ai.KindInvalidRequest, Message: fmt.Sprintf(
			"endpoint: %q states no protocol, so it cannot open model %q", e.cfg.ID, modelID)}
	}
	return ai.New(e.ConfigFor(m), opts...)
}

// decorate stamps a model with the endpoint's identity and protocol, then
// gives the resolver what it can add. It only fills an API the model does not
// state, so a set of mixed-protocol models keeps its own.
func (e *Endpoint) decorate(m ai.Model) ai.Model {
	m.Vendor = e.cfg.ID
	if m.API == "" {
		m.API = e.cfg.API
	}
	if e.cfg.Resolve != nil {
		m = e.cfg.Resolve(m)
	}
	return m
}

// MergeListing layers a live model listing over a known baseline entry.
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
	if api := listedAPI(out.API, live.API); api != out.API {
		out.API, out.Compat = api, live.Compat
	}
	return out
}

// listedAPI is the protocol a listed model speaks: the endpoint's own, unless
// a Chat endpoint lists the model as Responses-only, as GitHub Copilot does
// its newest GPT models. Any other protocol a listing names is the driver's
// default, not the host's word, and the endpoint's wins. An endpoint stating
// no protocol leaves the listing's.
func listedAPI(endpoint, listed ai.API) ai.API {
	if endpoint == "" || (endpoint == ai.APIOpenAIChat && listed == ai.APIOpenAIResponses) {
		return listed
	}
	return endpoint
}
