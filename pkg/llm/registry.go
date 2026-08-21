package llm

import (
	"fmt"
	"net/http"
	"slices"
	"sort"
	"sync"
)

// Config is everything a driver needs to reach an endpoint.
//
// Nothing here is discovered: no environment variable is read, no credential
// file is opened. A caller supplies the key it wants used, which is what makes
// the SDK safe in a server handling several tenants. Package llm/auth is the
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

	// Native carries construction settings only one protocol needs, as that
	// driver's value — VertexConfig for a model served through Vertex AI. It
	// is the Config-level counterpart to Options.Native: the latter varies per
	// request, this one is fixed for the endpoint.
	//
	// Read it with ConfigNativeOf, which yields the zero value when a Config
	// carries none or carries another protocol's.
	Native any
}

// ConfigNativeOf reads a driver's protocol-specific construction settings out
// of a Config.
//
//	vertex := llm.ConfigNativeOf[llm.VertexConfig](cfg)
func ConfigNativeOf[T any](c Config) T {
	n, _ := c.Native.(T)
	return n
}

// Endpoint returns the base URL to use: the Config override if set, otherwise
// the model's, otherwise "" for the driver's default.
func (c Config) Endpoint() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return c.Model.BaseURL
}

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
		panic("llm: RegisterAPI with nil factory for " + string(api))
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.m == nil {
		registry.m = make(map[API]Factory)
	}
	if _, dup := registry.m[api]; dup {
		panic("llm: RegisterAPI called twice for " + string(api))
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
		return nil, fmt.Errorf("llm: model %q has no API set", cfg.Model.ID)
	}
	registry.mu.RLock()
	f, ok := registry.m[cfg.Model.API]
	registry.mu.RUnlock()
	if !ok {
		return nil, &UnregisteredAPIError{API: cfg.Model.API, Registered: RegisteredAPIs()}
	}
	return f(cfg)
}

// Open builds the driver for cfg.Model's protocol and wraps it in a Client.
func Open(cfg Config, opts ...Option) (*Client, error) {
	d, err := NewDriver(cfg)
	if err != nil {
		return nil, err
	}
	return New(d, cfg.Model, opts...), nil
}

// UnregisteredAPIError reports a model whose protocol has no driver linked
// into the binary — nearly always a missing blank import.
type UnregisteredAPIError struct {
	API        API
	Registered []API
}

func (e *UnregisteredAPIError) Error() string {
	if len(e.Registered) == 0 {
		return fmt.Sprintf("llm: no driver registered for API %q; import a driver package "+
			"(e.g. _ %q)", e.API, "github.com/genai-io/sdk-go/pkg/llm/driver/all")
	}
	names := make([]string, len(e.Registered))
	for i, a := range e.Registered {
		names[i] = string(a)
	}
	slices.Sort(names)
	return fmt.Sprintf("llm: no driver registered for API %q (registered: %v)", e.API, names)
}
