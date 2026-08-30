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
//	Driver        required — translate a Request, send it, stream Deltas back
//	ModelLister   optional — the endpoint can list the models it serves
//	TokenCounter  optional — the endpoint can size a prompt without running it

// Driver is one wire protocol, as a black box.
type Driver interface {
	// Name identifies this implementation in errors and diagnostics. It names
	// the protocol, not the vendor: one driver serves every endpoint that
	// speaks it.
	Name() string

	// Stream performs one inference call.
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
// What a driver must guarantee, and what it need not:
//
//   - A Delta may carry no Block at all. One with only Usage or StopReason set
//     is how metadata arrives out of band.
//   - Text and thinking fragments accumulate into one block until it closes.
//     Switching from one type to the other closes the open one first, so a
//     driver need not send EndBlock to change what it is producing.
//   - The last block need not be closed. The stream ending closes it.
//   - Usage accumulates across deltas rather than replacing, so a protocol that
//     reports input and output separately may send them separately.
//   - StopReason, Model and ID are last-write-wins, and an empty one is
//     ignored — a driver repeating them costs nothing.
//   - Yielding an error ends the stream. Everything already yielded stays on
//     the Response the caller receives, so a failure that arrives mid-answer
//     keeps both the text and the tokens it cost.
type Delta struct {
	Block    Block
	EndBlock bool

	Usage      *Usage
	StopReason StopReason
	Model      string
	ID         string
}

// Config is everything a driver needs to reach an endpoint.
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
	ProtocolConfig ProtocolConfig
}

// MergedHeaders returns the headers to send: the model's, then the Config's
// over them, so a Config header of the same name wins.
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
func (c Config) URL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return c.Model.BaseURL
}

// The driver registry: how a Model's protocol is turned into a Driver.

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

// NewClient builds the driver for cfg.Model's protocol and wraps it in a Client. It
// is NewDriver plus NewClientWithDriver, and it is the constructor most callers
// want: reach for NewDriver and NewClientWithDriver only when the driver has to pass
// through your hands on the way, which is what wrapping it in Middleware
// needs.
func NewClient(cfg Config, opts ...Option) (*Client, error) {
	d, err := NewDriver(cfg)
	if err != nil {
		return nil, err
	}
	return NewClientWithDriver(d, cfg.Model, opts...), nil
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
