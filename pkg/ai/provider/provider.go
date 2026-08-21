package provider

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

// Provider is a configured provider: the models it serves, the credential to
// reach it with, and the protocol that talks to it.
//
// Reading its model list is synchronous and cannot fail — it returns what is
// known now, which before the first Refresh is the static baseline it was
// built with. Fetching is a separate, explicit verb. That split is what lets a
// model picker render immediately and refresh behind the user, instead of
// blocking on a round trip that a dead provider can hang.
//
// An Provider is safe for concurrent use.
type Provider struct {
	cfg Config

	mu      sync.RWMutex
	overlay []ai.Model
}

// Config describes one provider.
type Config struct {
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
	// provider. It may be empty for an provider that only has a live listing.
	Models []ai.Model

	// API is the protocol this provider speaks. It is only needed when Models
	// is empty, so that Refresh knows which driver to ask; otherwise it is
	// taken from the baseline.
	API ai.API

	// Fetch retrieves the live model list. Nil means the default: open a
	// driver for this provider's protocol and call its Models. Set it for an
	// provider whose listing lives somewhere other than the protocol's own
	// models call.
	Fetch func(ctx context.Context, e *Provider) ([]ai.Model, error)
}

// New builds an provider from its parts.
func New(cfg Config) *Provider {
	cfg.Headers = maps.Clone(cfg.Headers)
	cfg.Models = cloneAll(cfg.Models)
	if cfg.Name == "" {
		cfg.Name = cfg.ID
	}
	if cfg.API == "" && len(cfg.Models) > 0 {
		cfg.API = cfg.Models[0].API
	}
	return &Provider{cfg: cfg}
}

// ID returns the provider's key.
func (e *Provider) ID() string { return e.cfg.ID }

// Name returns the provider's display name.
func (e *Provider) Name() string { return e.cfg.Name }

// API returns the protocol this provider speaks.
func (e *Provider) API() ai.API { return e.cfg.API }

// Models returns what is known now: the static baseline with the last fetched
// listing merged over it. It never blocks and never fails — before the first
// successful Refresh it is the baseline alone.
//
// The merge is field by field, not entry by entry. A listing carries what the
// provider publishes, which for most OpenAI-compatible vendors is an ID and
// nothing else; replacing a baseline entry wholesale would discard its
// pricing, its reasoning ladder and its protocol quirks, and a model stripped
// of its quirks stops working. So the provider wins on every field it stated,
// and the baseline fills the rest.
func (e *Provider) Models() []ai.Model {
	e.mu.RLock()
	overlay := cloneAll(e.overlay)
	e.mu.RUnlock()

	// Decorate the baseline as well as the listing. Every model an provider
	// hands out carries that provider's identity and protocol, whether it came
	// from the baseline, from a refresh, or from an ID nobody has listed —
	// otherwise Open would work for a model the provider has never heard of
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
	// No clone on the way out: every entry above is already a fresh value —
	// the baseline through Clone, the overlay through cloneAll, a merged pair
	// through MergeListing, which clones both sides.
	return merged
}

// Model looks one model up by ID. Unlike Models it also answers for an ID the
// provider has never heard of, by decorating it with the provider's protocol
// and provider — an unlisted model is nearly always one newer than the
// catalog, not one that does not exist.
func (e *Provider) Model(id string) (ai.Model, bool) {
	for _, m := range e.Models() {
		if strings.EqualFold(m.ID, id) {
			return m, true
		}
	}
	return e.decorate(ai.Model{ID: id}), false
}

// Refresh fetches the live model list and merges it in. A failure leaves the
// previous list untouched, so an provider that went down keeps serving what it
// last knew.
//
// An provider with no way to list — no Fetch and no protocol — is a no-op.
func (e *Provider) Refresh(ctx context.Context) error {
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

// defaultFetch opens a client for the provider's protocol and asks it.
//
// The driver needs a model to be constructed with, and listing does not depend
// on which one, so a placeholder stands in when the provider has no baseline
// to borrow from. Client.Models is what answers, rather than the driver
// directly, so a protocol with no listing provider reports it the same way
// here as it does to any other caller.
func defaultFetch(ctx context.Context, e *Provider) ([]ai.Model, error) {
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

// ConfigFor builds the ai.Config for opening one of this provider's models.
func (e *Provider) ConfigFor(m ai.Model) ai.Config {
	return ai.Config{
		Model:      m.Clone(),
		APIKey:     e.cfg.APIKey,
		BaseURL:    e.cfg.BaseURL,
		HTTPClient: e.cfg.HTTPClient,
		Headers:    maps.Clone(e.cfg.Headers),
		Native:     e.cfg.Native,
	}
}

// Open returns a client for one of this provider's models. An ID the provider
// does not list still opens, carrying the provider's protocol and provider.
func (e *Provider) Open(modelID string, opts ...ai.Option) (*ai.Client, error) {
	// The bool is discarded on purpose: an unlisted ID is fine — Model
	// decorates it with this provider's protocol. What is not fine is having
	// no protocol to decorate it with.
	m, _ := e.Model(modelID)
	if m.API == "" {
		return nil, &ai.Error{Kind: ai.KindInvalidRequest, Message: fmt.Sprintf(
			"ai: provider %q states no protocol, so it cannot open model %q", e.cfg.ID, modelID)}
	}
	return ai.Open(e.ConfigFor(m), opts...)
}

// decorate stamps a model with the provider's identity and protocol. It only
// fills an API the model does not state, so a set of mixed-protocol models
// keeps its own.
func (e *Provider) decorate(m ai.Model) ai.Model {
	m.Vendor = e.cfg.ID
	if m.API == "" {
		m.API = e.cfg.API
	}
	return m
}

// MergeListing layers a live model listing over a known baseline entry.
//
// The listing wins on every field it stated; everything it left zero comes
// from the baseline. That asymmetry is the whole rule: an provider is
// authoritative about which models exist and about any figure it reported,
// while a baseline knows the pricing, the reasoning ladder and the protocol
// quirks that no listing publishes — and a model stripped of its quirks stops
// working.
//
// Provider uses it to merge a refresh over its static models. It is exported
// because a caller reconciling a listing against the catalog itself needs the
// same rule, and two copies of it would let one path keep a field the other
// dropped.
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

// cloneAll snapshots a model list wherever one crosses in or out of an
// Provider, so a caller may keep mutating its own builders and a refresh
// running in another goroutine cannot rewrite a list someone is reading.
func cloneAll(models []ai.Model) []ai.Model {
	out := make([]ai.Model, len(models))
	for i, m := range models {
		out[i] = m.Clone()
	}
	return out
}
