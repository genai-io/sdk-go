# Constructing a client

There is one chain from a model reference to a client. You join it wherever
you already are, and each step adds one nameable thing:

```
   "openai/gpt-4.1"          a string, and nothing else
          │
          │  catalog.Model      the model's own facts
          ▼
       ai.Model               which protocol it speaks, its context window and
          │                   max output, the modalities it accepts, its
          │                   reasoning ladder, its prices, what it cannot do
          │
          │  auth.Config       the credential, and where to send it
          ▼
      ai.Config               APIKey from the vendor's own environment
          │                   variable, BaseURL, any deployment settings —
          │                   and a failure here if a key is required and
          │                   missing, rather than a 401 three calls later
          │
          │  ai.NewDriver      the protocol implementation
          ▼
      ai.Driver               found in the registry by Model.API, which is
          │                   what the blank import fills in; holds the HTTP
          │                   transport and the vendor's auth headers
          │
          │  ai.NewClientWithDriver
          ▼
     *ai.Client               the defaults every call inherits, and a private
                              copy of the Model so a later edit on your side
                              cannot reach a request already in flight
```

Nothing branches, so there is no choice to get wrong — only how far along you
need to stop.

The two names you will actually type are shortcuts along it, and each is
literally one line:

```go
// auth.Client = auth.Config + ai.NewClient
func Client(ref string, opts ...ai.Option) (*ai.Client, error) {
	cfg, err := Config(ref)
	if err != nil {
		return nil, err
	}
	return ai.NewClient(cfg, opts...)
}

// ai.NewClient = ai.NewDriver + ai.NewClientWithDriver
func NewClient(cfg Config, opts ...Option) (*Client, error) {
	d, err := NewDriver(cfg)
	if err != nil {
		return nil, err
	}
	return NewClientWithDriver(d, cfg.Model, opts...), nil
}
```

All three produce the same client:

```go
client, err := auth.Client("openai/gpt-4.1")

cfg, err := auth.Config("openai/gpt-4.1")
client, err := ai.NewClient(cfg)

driver, err := ai.NewDriver(cfg)
client := ai.NewClientWithDriver(driver, cfg.Model)
```

So the question is not which constructor to use. It is **how far along the
chain you need to stop**:

| Stop at | When | What you supply |
| --- | --- | --- |
| `auth.Client(ref)` | A command-line tool | The reference. The credential comes from the environment. |
| `auth.Config(ref)` | Same, but the endpoint or the `http.Client` has to change | The reference, then your edits to the `Config` |
| `ai.NewClient(cfg)` | A server. Nothing ambient may be read. | The `Model` and the credential |
| `ai.NewDriver(cfg)` | The driver has to pass through your hands — middleware | The same, and the last step yourself |
| `ai.NewClientWithDriver(driver, model)` | You already hold a driver, including a stub in tests | The driver and the `Model` |

Every path needs the driver for its protocol registered, which a blank import
does. Use `all` when the model is chosen at run time; use one when you know the
protocol at compile time and would rather not link the other vendors' SDKs.

```go
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
```

## Stopping at `auth.Client`

```go
client, err := auth.Client("anthropic/claude-opus-5", ai.WithEffort(ai.EffortHigh))
```

It looks the reference up in the catalog, reads the credential from the
environment variable that vendor documents, and returns a client. If the vendor
needs a key and none of its variables are set, this fails here rather than
sending an empty credential and surfacing it later as a confusing 401.

For the vendors that authenticate a person rather than a service — GitHub
Copilot, ChatGPT/Codex — it runs the browser sign-in and stores the result
`0600` under the user's config directory. That store is the only file this
library writes.

## Stopping at `auth.Config`

```go
cfg, err := auth.Config("openai/gpt-4.1")
cfg.BaseURL = "https://gateway.internal/v1"
cfg.HTTPClient = instrumented

client, err := ai.NewClient(cfg)
```

`auth.Config` fills the credential and endpoint and stops, so what it returns
is an ordinary `ai.Config` you can edit.

## Stopping at `ai.NewClient`

`pkg/ai` reads no environment variable and no file. That is what makes it safe
in a server holding several tenants' keys: nothing it does depends on ambient
state, so two requests cannot pick up each other's credentials.

```go
model, err := catalog.Model("anthropic/claude-opus-5")

client, err := ai.NewClient(ai.Config{
	Model:      model,
	APIKey:     tenantKey,
	BaseURL:    "https://gateway.internal/v1",
	HTTPClient: httpClient,
	Headers:    map[string]string{"X-Tenant": tenant},
})
```

`catalog.Model` is the lookup on its own — endpoint, limits, pricing and the
protocol to speak, with no network call and no credential. Fill in an
`ai.Model` yourself for a model the catalog has never heard of.

## Stopping at `ai.NewDriver`

Execution policy — retries, cost metering, caching, logging — is `Middleware`
decorating a driver, so it needs the driver as a value:

```go
driver, err := ai.NewDriver(cfg)
client := ai.NewClientWithDriver(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), cfg.Model)
```

`Retry` is the one policy shipped here, and every driver disables its vendor
SDK's own, so without it you get none. Order is outermost first: the first
middleware sees the request first and the response last. A `Middleware` is a
`Handler` wrapping a `Handler`, the same shape as `Driver.Stream`, so anything
you write composes with anything shipped here.

## Stopping at `ai.NewClientWithDriver`

`ai.NewClientWithDriver` does no I/O, no lookup and no credential validation. It is where every
path above ends, and it is what tests want:

```go
client := ai.NewClientWithDriver(scripted{deltas: deltas}, ai.Model{
	ID: "test", API: ai.APIAnthropicMessages, MaxOutput: 1024,
})
```

Because `Driver` is two methods, the stub for a given case is a few lines. See
[CONTRIBUTING.md](../CONTRIBUTING.md#testing-your-own-code) for one.

## A provider is not on this chain

Everything above builds a client for **one model**. A `Provider` is the layer
beside it: one configured host and the **list** of models on it, which is what
a model picker needs.

```go
p, err := auth.Provider("ollama")

models := p.Models()          // synchronous, never blocks, never fails
err = p.Refresh(ctx)          // the only call that reaches the network

client, err := p.Client("llama4")
```

Reading the list and fetching it are separate verbs on purpose: a picker
renders immediately from the catalog, and a host that is down cannot hang it.
For a host the catalog does not know about, use `provider.New(provider.Config{…})`.

## Defaults and overrides

Wherever you stopped, the same `Option` is a default at construction and an
override at the call:

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh))

response, err := client.Complete(ctx, messages,
	ai.WithEffort(ai.EffortLow)) // this turn only
```

Resolution runs model defaults, then client defaults, then call overrides.
Passing an option is what marks a setting explicit, so `ai.WithTemperature(0)`
is deterministic sampling and omitting it inherits.
