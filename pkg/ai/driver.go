package ai

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"reflect"
	"slices"
	"sort"
	"sync"
)

// The protocol seam: what a driver is.
//
// A driver is one wire protocol and nothing else. Everything a caller also
// needs — aggregating deltas into a Response, applying defaults, repairing
// history, validating, retrying, discovering credentials — belongs to Client
// and is written once for every protocol instead of once per driver. Keeping
// that line is what makes a new protocol a small, self-contained package.
//
// Implementing one is three decisions:
//
//	Driver        required — translate a Request, send it, stream Deltas back
//	ModelLister   optional — the endpoint can list the models it serves
//	TokenCounter  optional — the endpoint can size a prompt without running it
//
// The optional two are found by type assertion, so adding one later is a
// source-compatible change and leaving one out is never an error: Client.Models
// reports KindUnsupported, and Client.CountTokens falls back to EstimateTokens.

// Driver is one wire protocol, as a black box.
type Driver interface {
	// Name identifies this implementation in errors and diagnostics. It names
	// the protocol, not the vendor: one driver serves every endpoint that
	// speaks it.
	Name() string

	// Stream performs one inference call.
	//
	// Streaming is the only shape a driver implements. Client.Complete is
	// built by draining this, not by a second code path, so there is one way
	// a request reaches an endpoint and one way its output comes back.
	//
	// It shares a name with Client.Stream because it is the same operation one
	// layer down; what differs is granularity. A driver yields raw Deltas as
	// its protocol produces them, and the Client assembles those into the
	// ordered block lifecycle its own Stream yields.
	//
	// It does nothing until the returned iterator is consumed, and stops when
	// the iterator is abandoned or ctx is canceled.
	//
	// req is borrowed, not owned: the Client has already resolved its options,
	// repaired its history and validated the result, and it reads req again
	// after the call returns. A driver must not modify or retain it.
	//
	// An error ends the iterator and must be its last element. Make it an
	// *Error so IsAuth, IsRetryable and the rest can classify what happened.
	Stream(ctx context.Context, req *Request) iter.Seq2[Delta, error]
}

// ModelLister is the optional capability of an endpoint that publishes the
// models it serves. Client.Models reports KindUnsupported without it.
type ModelLister interface {
	Models(ctx context.Context) ([]Model, error)
}

// TokenCounter is the optional capability of an endpoint that will size a
// prompt without generating from it. Anthropic and Google publish one; the
// OpenAI-family protocols do not, so their drivers omit this and callers get
// EstimateTokens instead.
type TokenCounter interface {
	// CountTokens reports the request's exact size, as the provider's own
	// tokenizer sees it. req is borrowed on the same terms as Stream's.
	CountTokens(ctx context.Context, req *Request) (int, error)
}

// Delta is the protocol-driver output. Block is one ordered content update:
// text and thinking blocks carry incremental Text/Signature fragments; image,
// tool and opaque reasoning blocks are complete values. EndBlock closes the
// current textual block after applying Block.
//
// Usage and StopReason are replacing rather than additive: the last non-zero
// value wins.
type Delta struct {
	Block    Block
	EndBlock bool

	Usage      *Usage
	StopReason StopReason
	Model      string
	ID         string
}

// Config is everything a driver needs to reach an endpoint.
//
// Nothing here is discovered: no environment variable is read, no credential
// file is opened. A caller supplies the key it wants used, which is what makes
// the SDK safe in a server handling several tenants. Package ai/auth is the
// opt-in helper that fills a Config from the environment.
type Config struct {
	// Model is the model to talk to; its API selects the driver.
	Model Model

	// APIKey is the credential. Some endpoints (a local Ollama) need none.
	APIKey string

	// BaseURL overrides Model.BaseURL, which in turn overrides the driver's
	// own default host.
	BaseURL string

	// HTTPClient is used for every request when set. Supply one to control
	// timeouts, proxies, or transport-level instrumentation.
	HTTPClient *http.Client

	// Headers are added to every request — a gateway token, a tenant tag.
	Headers map[string]string

	// ProtocolConfig carries construction settings only one protocol needs, as
	// that driver's value — VertexConfig for a model served through Vertex AI. It
	// is the Config-level counterpart to Request.ProtocolOptions: the latter varies per
	// request, this one is fixed for the endpoint.
	//
	// Read it with ProtocolConfigAs. A wrong concrete type is rejected rather than
	// silently losing construction settings.
	ProtocolConfig ProtocolConfig
}

// MergedHeaders returns the headers to send: the model's, then the Config's
// over them, so a Config header of the same name wins.
//
// Every driver needs this same precedence, and four copies of it is four
// chances for one protocol to resolve a header differently from the rest.
func (c Config) MergedHeaders() map[string]string {
	if len(c.Model.Headers) == 0 && len(c.Headers) == 0 {
		return nil
	}
	out := make(map[string]string, len(c.Model.Headers)+len(c.Headers))
	maps.Copy(out, c.Model.Headers)
	maps.Copy(out, c.Headers)
	return out
}

// URL returns the base URL to use: the Config override if set, otherwise the
// model's, otherwise "" for the driver's default.
//
// It is not called Endpoint because that word already names a whole configured
// host in package ai/endpoint, and one word cannot mean both.
func (c Config) URL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return c.Model.BaseURL
}

// The driver registry: how a Model's protocol is turned into a Driver.
//
// It is the database/sql pattern. A driver package registers itself from init,
// so a blank import is what makes a protocol reachable, and a program that
// talks to one provider does not link the other vendors' SDKs.

// driverPath is where the bundled drivers live. It appears only in the error a
// caller gets when none is linked in, which is the one moment they need to know
// it.
const driverPath = "github.com/genai-io/sdk-go/pkg/ai/driver"

// Factory builds a Driver for one protocol.
type Factory func(Config) (Driver, error)

var registry struct {
	mu sync.RWMutex
	m  map[API]Factory
}

// RegisterAPI registers the driver factory for a wire protocol. Driver
// packages call it from init, so a blank import is enough to make a protocol
// reachable through Open. Registering the same API twice panics: two factories
// for one protocol means one of them is silently dead.
func RegisterAPI(api API, f Factory) {
	if f == nil {
		panic("ai: RegisterAPI with nil factory for " + string(api))
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.m == nil {
		registry.m = make(map[API]Factory)
	}
	if _, dup := registry.m[api]; dup {
		panic("ai: RegisterAPI called twice for " + string(api))
	}
	registry.m[api] = f
}

// RegisteredAPIs lists the protocols with a registered driver, sorted.
func RegisteredAPIs() []API {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	out := make([]API, 0, len(registry.m))
	for api := range registry.m {
		out = append(out, api)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// NewDriver builds the driver for cfg.Model's protocol.
func NewDriver(cfg Config) (Driver, error) {
	if cfg.Model.API == "" {
		return nil, fmt.Errorf("ai: model %q has no API set", cfg.Model.ID)
	}
	registry.mu.RLock()
	f, ok := registry.m[cfg.Model.API]
	registry.mu.RUnlock()
	if !ok {
		return nil, &UnregisteredAPIError{API: cfg.Model.API, Registered: RegisteredAPIs()}
	}
	cfg.Model = cloneModel(cfg.Model)
	cfg.Headers = maps.Clone(cfg.Headers)
	return f(cfg)
}

// New builds the driver for cfg.Model's protocol and wraps it in a Client. It
// is NewDriver plus NewWithDriver, and it is the constructor most callers
// want: reach for NewDriver and NewWithDriver only when the driver has to pass
// through your hands on the way, which is what wrapping it in Middleware
// needs.
func New(cfg Config, opts ...Option) (*Client, error) {
	d, err := NewDriver(cfg)
	if err != nil {
		return nil, err
	}
	return NewWithDriver(d, cfg.Model, opts...), nil
}

// UnregisteredAPIError reports a model whose protocol has no driver linked
// into the binary — nearly always a missing blank import.
type UnregisteredAPIError struct {
	API        API
	Registered []API
}

func (e *UnregisteredAPIError) Error() string {
	if len(e.Registered) == 0 {
		return fmt.Sprintf("ai: no driver registered for API %q; blank-import the package "+
			"that implements it from %s — or %s/all for every protocol, which costs "+
			"every protocol's dependencies", e.API, driverPath, driverPath)
	}
	names := make([]string, len(e.Registered))
	for i, a := range e.Registered {
		names[i] = string(a)
	}
	slices.Sort(names)
	return fmt.Sprintf("ai: no driver registered for API %q (registered: %v)", e.API, names)
}

// ProtocolConfig is one driver's protocol-specific construction settings — the
// Config-level counterpart to ProtocolOptions.
//
// ai.VertexConfig is the only one this module defines, and it lives in package
// ai rather than beside its driver so a caller can fill it in without pulling
// in Google Cloud auth.
type ProtocolConfig interface {
	// ProtocolConfig marks this type as one driver's construction settings. A
	// no-op, as ProtocolOptions.ProtocolOptions is.
	ProtocolConfig()
}

// ProtocolConfigAs reads a driver's protocol-specific construction settings out
// of a Config.
//
//	vertex, err := ai.ProtocolConfigAs[ai.VertexConfig](cfg)
func ProtocolConfigAs[T ProtocolConfig](config Config) (T, error) {
	var zero T
	if config.ProtocolConfig == nil {
		return zero, nil
	}
	native, ok := config.ProtocolConfig.(T)
	if ok {
		return native, nil
	}
	return zero, &Error{
		Kind:    KindInvalidRequest,
		Message: fmt.Sprintf("ai: native config has type %T; driver expects %s", config.ProtocolConfig, reflect.TypeFor[T]()),
	}
}

// RejectProtocolConfig returns an invalid-request error for a protocol with no
// native construction options.
func RejectProtocolConfig(config Config, driver string) error {
	if config.ProtocolConfig == nil {
		return nil
	}
	return &Error{
		Driver: driver, Kind: KindInvalidRequest,
		Message: "ai: driver " + driver + " does not define native configuration",
	}
}
