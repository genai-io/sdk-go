package ai

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"slices"
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
//   - Usage is merged field by field, last non-zero wins, so a protocol that
//     reports input at the start and output at the end may send them
//     separately. Each field carries the running total, not an increment.
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
	// Model is the model to talk to. Its API selects the driver; its BaseURL
	// and Headers are where requests go and what they carry.
	Model Model

	// APIKey is the credential. Some endpoints (a local Ollama) need none.
	APIKey string

	// HTTPClient is used for every request when set. Supply one to control
	// timeouts, proxies, or transport-level instrumentation.
	HTTPClient *http.Client

	// Vertex is the deployment of a model served through Vertex AI. The
	// anthropic/vertex and google/vertex drivers require it; the others ignore
	// it.
	Vertex *VertexConfig
}

// VertexConfig is the deployment a Vertex-served model lives in.
type VertexConfig struct {
	// Project is the GCP project ID serving the model.
	Project string
	// Region is the serving region. "global" is the usual choice and is what
	// an empty value resolves to.
	Region string
}

// VertexDefaultRegion is where a model is served when the deployment names no
// region. Google recommends the global endpoint for availability; a specific
// region is for data residency.
const VertexDefaultRegion = "global"

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
// reachable through New. Registering the same API twice panics:
// two factories for one protocol means one of them is silently dead.
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
	return slices.Sorted(maps.Keys(registry.m))
}

// newDriver builds the driver for cfg.Model's protocol.
func newDriver(cfg Config) (Driver, error) {
	if cfg.Model.ID == "" {
		return nil, fmt.Errorf("ai: model ID is required")
	}
	if cfg.Model.API == "" {
		return nil, fmt.Errorf("ai: model %q has no API set", cfg.Model.ID)
	}
	registry.mu.RLock()
	f, ok := registry.m[cfg.Model.API]
	registry.mu.RUnlock()
	if !ok {
		return nil, unregistered(cfg.Model.API, RegisteredAPIs())
	}
	cfg.Model = cfg.Model.Clone()
	return f(cfg)
}

// New builds the driver for cfg.Model's protocol and wraps it in a Client.
func New(cfg Config, opts ...Option) (*Client, error) {
	d, err := newDriver(cfg)
	if err != nil {
		return nil, err
	}
	return NewClientWithDriver(d, cfg.Model, opts...), nil
}

// unregistered reports a model whose protocol has no driver linked into the
// binary — nearly always a missing blank import.
func unregistered(api API, registered []API) error {
	if len(registered) == 0 {
		return fmt.Errorf("ai: no driver registered for API %q; blank-import the package "+
			"that implements it from %s — or %s/all for every protocol, which costs "+
			"every protocol's dependencies", api, driverPath, driverPath)
	}
	return fmt.Errorf("ai: no driver registered for API %q (registered: %v)", api, registered)
}
