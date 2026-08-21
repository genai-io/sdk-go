package llm

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
)

// Provider is a configured endpoint: the models it serves, the credential to
// reach it with, and the protocol that talks to it.
//
// Reading its model list is synchronous and cannot fail — it returns what is
// known now, which before the first Refresh is the static baseline it was
// built with. Fetching is a separate, explicit verb. That split is what lets a
// model picker render immediately and refresh behind the user, instead of
// blocking on a round trip that a dead endpoint can hang.
//
// A Provider is safe for concurrent use.
type Provider struct {
	cfg ProviderConfig

	mu      sync.RWMutex
	overlay []Model
}

// ProviderConfig describes one endpoint.
type ProviderConfig struct {
	// ID is the short key this provider is known by, e.g. "deepseek".
	ID string
	// Name is how it should be shown. Defaults to ID.
	Name string

	// BaseURL, APIKey, HTTPClient and Headers are applied to every model this
	// provider opens. A model's own BaseURL and Headers layer underneath.
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	Headers    map[string]string

	// Native is passed through to every Config this provider builds — see
	// Config.Native.
	Native any

	// Models is the static baseline: what is known without asking the
	// endpoint. It may be empty for a provider that only has a live listing.
	Models []Model

	// API is the protocol this provider speaks. It is only needed when Models
	// is empty, so that Refresh knows which driver to ask; otherwise it is
	// taken from the baseline.
	API API

	// Fetch retrieves the live model list. Nil means the default: open a
	// driver for this provider's protocol and call its Models. Set it for an
	// endpoint whose listing lives somewhere other than the protocol's own
	// models call.
	Fetch func(ctx context.Context, p *Provider) ([]Model, error)
}

// NewProvider builds a provider from its parts.
func NewProvider(cfg ProviderConfig) *Provider {
	if cfg.Name == "" {
		cfg.Name = cfg.ID
	}
	if cfg.API == "" && len(cfg.Models) > 0 {
		cfg.API = cfg.Models[0].API
	}
	return &Provider{cfg: cfg}
}

// ID returns the provider's key.
func (p *Provider) ID() string { return p.cfg.ID }

// Name returns the provider's display name.
func (p *Provider) Name() string { return p.cfg.Name }

// API returns the protocol this provider speaks.
func (p *Provider) API() API { return p.cfg.API }

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
func (p *Provider) Models() []Model {
	p.mu.RLock()
	overlay := p.overlay
	p.mu.RUnlock()

	merged := slices.Clone(p.cfg.Models)
	for _, live := range overlay {
		i := slices.IndexFunc(merged, func(m Model) bool { return strings.EqualFold(m.ID, live.ID) })
		if i < 0 {
			merged = append(merged, p.decorate(live))
			continue
		}
		merged[i] = mergeModel(merged[i], live)
	}
	return merged
}

// Model looks one model up by ID. Unlike Models it also answers for an ID the
// provider has never heard of, by decorating it with the provider's protocol
// and endpoint — an unlisted model is nearly always one newer than the
// catalog, not one that does not exist.
func (p *Provider) Model(id string) (Model, bool) {
	for _, m := range p.Models() {
		if strings.EqualFold(m.ID, id) {
			return m, true
		}
	}
	return p.decorate(Model{ID: id}), false
}

// Refresh fetches the live model list and merges it in. A failure leaves the
// previous list untouched, so a provider that went down keeps serving what it
// last knew.
//
// A provider with no way to list — no Fetch and no protocol — is a no-op.
func (p *Provider) Refresh(ctx context.Context) error {
	fetch := p.cfg.Fetch
	if fetch == nil {
		if p.cfg.API == "" {
			return nil
		}
		fetch = defaultFetch
	}
	models, err := fetch(ctx, p)
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.overlay = models
	p.mu.Unlock()
	return nil
}

// defaultFetch opens a driver for the provider's protocol and asks it.
//
// The driver needs a model to be constructed with, and listing does not depend
// on which one, so a placeholder stands in when the provider has no baseline
// to borrow from.
func defaultFetch(ctx context.Context, p *Provider) ([]Model, error) {
	probe := Model{ID: "-", API: p.cfg.API}
	if len(p.cfg.Models) > 0 {
		probe = p.cfg.Models[0]
	}
	d, err := NewDriver(p.Config(probe))
	if err != nil {
		return nil, err
	}
	return d.Models(ctx)
}

// Config builds the Config for opening one of this provider's models.
func (p *Provider) Config(m Model) Config {
	return Config{
		Model:      m,
		APIKey:     p.cfg.APIKey,
		BaseURL:    p.cfg.BaseURL,
		HTTPClient: p.cfg.HTTPClient,
		Headers:    p.cfg.Headers,
		Native:     p.cfg.Native,
	}
}

// Open returns a client for one of this provider's models. An ID the provider
// does not list still opens, carrying the provider's protocol and endpoint.
func (p *Provider) Open(modelID string, opts ...Option) (*Client, error) {
	m, _ := p.Model(modelID)
	if m.API == "" {
		return nil, fmt.Errorf("llm: provider %q has no protocol for model %q", p.cfg.ID, modelID)
	}
	return Open(p.Config(m), opts...)
}

// decorate stamps a bare model with the provider's identity and protocol.
func (p *Provider) decorate(m Model) Model {
	m.Vendor = p.cfg.ID
	if m.API == "" {
		m.API = p.cfg.API
	}
	return m
}

// mergeModel layers a live listing over a baseline entry. The listing wins on
// anything it stated; everything it left zero comes from the baseline.
func mergeModel(base, live Model) Model {
	out := base
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

// Providers is a set of providers, keyed by ID.
//
// It is a plain collection: reads are synchronous, Refresh fans out and
// reports per-provider failures rather than returning one error, because a
// single dead endpoint must not empty the list.
type Providers struct {
	mu sync.RWMutex
	m  map[string]*Provider
}

// NewProviders builds a set from the given providers.
func NewProviders(list ...*Provider) *Providers {
	ps := &Providers{m: make(map[string]*Provider, len(list))}
	for _, p := range list {
		ps.Set(p)
	}
	return ps
}

// Set adds or replaces a provider by ID.
func (ps *Providers) Set(p *Provider) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.m == nil {
		ps.m = make(map[string]*Provider)
	}
	ps.m[p.ID()] = p
}

// Delete removes a provider.
func (ps *Providers) Delete(id string) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	delete(ps.m, id)
}

// Get returns one provider.
func (ps *Providers) Get(id string) (*Provider, bool) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	p, ok := ps.m[id]
	return p, ok
}

// All returns every provider, sorted by ID for a stable listing.
func (ps *Providers) All() []*Provider {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	out := make([]*Provider, 0, len(ps.m))
	for _, p := range ps.m {
		out = append(out, p)
	}
	slices.SortFunc(out, func(a, b *Provider) int { return strings.Compare(a.ID(), b.ID()) })
	return out
}

// Models returns every provider's known models, in provider order. It never
// blocks and never fails.
func (ps *Providers) Models() []Model {
	var out []Model
	for _, p := range ps.All() {
		out = append(out, p.Models()...)
	}
	return out
}

// Model resolves a "vendor/id" reference against the set. A model ID may
// itself contain a slash, so only the first segment names the provider.
func (ps *Providers) Model(ref string) (Model, bool) {
	id, rest, ok := strings.Cut(ref, "/")
	if !ok {
		return Model{}, false
	}
	p, found := ps.Get(id)
	if !found {
		return Model{}, false
	}
	return p.Model(rest)
}

// Open returns a client for a "vendor/id" reference.
func (ps *Providers) Open(ref string, opts ...Option) (*Client, error) {
	id, rest, ok := strings.Cut(ref, "/")
	if !ok {
		return nil, fmt.Errorf("llm: %q is not a vendor/model reference", ref)
	}
	p, found := ps.Get(id)
	if !found {
		return nil, fmt.Errorf("llm: no provider %q", id)
	}
	return p.Open(rest, opts...)
}

// RefreshResult reports what one fan-out refresh managed.
type RefreshResult struct {
	// Errors holds the failure for each provider that could not be refreshed.
	// A provider absent from the map succeeded.
	Errors map[string]error
}

// OK reports whether every provider refreshed.
func (r RefreshResult) OK() bool { return len(r.Errors) == 0 }

// Refresh fetches every provider's listing concurrently.
//
// It does not return an error: one endpoint being down must not fail the whole
// call, and each provider keeps whatever it last knew. Read RefreshResult to
// see which ones failed.
func (ps *Providers) Refresh(ctx context.Context) RefreshResult {
	providers := ps.All()
	errs := make([]error, len(providers))

	var wg sync.WaitGroup
	for i, p := range providers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = p.Refresh(ctx)
		}()
	}
	wg.Wait()

	result := RefreshResult{}
	for i, err := range errs {
		if err != nil {
			if result.Errors == nil {
				result.Errors = make(map[string]error)
			}
			result.Errors[providers[i].ID()] = err
		}
	}
	return result
}
