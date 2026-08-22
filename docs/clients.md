# Constructing a client

There is one `ai.Client` and several ways to reach it. They form a ladder: each
rung hands you more control and asks you for more, and every rung produces the
same `*ai.Client`, so nothing downstream depends on which one you used.

| You want | Use |
| --- | --- |
| One model, credential from the environment | [`auth.Client`](#1-authclient) |
| A model picker, or a host whose models are discovered at run time | [`auth.Provider`](#2-authprovider) |
| The environment's credential, with something overridden | [`auth.Config`](#3-authconfig-then-ainewclient) |
| A server holding several tenants' keys | [`ai.NewClient`](#4-ainewclient) |
| Retries, cost metering, caching, logging | [`ai.Wrap`](#5-ainewdriver-and-aiwrap) |
| A test with no network | [`ai.New`](#6-ainew-with-your-own-driver) |

Every path needs the driver for its protocol registered, which a blank import
does:

```go
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"           // all five
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"   // or just one
```

Import `all` when the model is chosen at run time. Import one when you know the
protocol at compile time and would rather not link the other vendors' SDKs.

## 1. `auth.Client`

The short path, and the right one for a command-line tool.

```go
client, err := auth.Client("anthropic/claude-opus-5")
```

It looks the reference up in the catalog, reads the credential from the
environment variable that vendor documents, and returns a client. If the vendor
needs a key and none of its variables are set, this fails here rather than
sending an empty credential and surfacing it later as a confusing 401.

Defaults for every call go on the end:

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh))
```

For the vendors that authenticate a person rather than a service — GitHub
Copilot, ChatGPT/Codex — this runs the browser sign-in and stores the result
`0600` under the user's config directory. That store is the only file this
library writes.

## 2. `auth.Provider`

A `Provider` is one configured host and the list of models on it. Use it when
you need the list, not just one model.

```go
p, err := auth.Provider("ollama")
if err != nil {
	return err
}

models := p.Models()              // synchronous, never blocks, never fails
if err := p.Refresh(ctx); err != nil {
	log.Printf("listing %s: %v", p.Name(), err) // non-fatal
}

client, err := p.Client("llama4")
```

Reading the list and fetching it are separate verbs on purpose: a model picker
renders immediately from the catalog, and a host that is down cannot hang it.
`Refresh` is the only call that reaches the network, and after it `Models()`
returns the catalog rows merged with what the host actually serves.

To configure a host the catalog does not know about, build the provider
yourself:

```go
p := provider.New(provider.Config{
	ID:      "internal",
	Name:    "Internal gateway",
	API:     ai.APIOpenAIChat,
	BaseURL: "https://gateway.internal/v1",
	APIKey:  key,
})
```

## 3. `auth.Config`, then `ai.NewClient`

When the environment has the credential but something else needs overriding —
a proxy, a gateway, a custom `http.Client`:

```go
cfg, err := auth.Config("openai/gpt-4.1")
if err != nil {
	return err
}
cfg.BaseURL = "https://gateway.internal/v1"
cfg.HTTPClient = instrumented

client, err := ai.NewClient(cfg)
```

`auth.Config` fills the credential and endpoint and stops there, so what it
returns is an ordinary `ai.Config` you can edit before using it.

## 4. `ai.NewClient`

`pkg/ai` reads no environment variable and no file. That is what makes it safe
in a server holding several tenants' keys: nothing it does depends on ambient
state, so two requests cannot pick up each other's credentials.

```go
model, err := catalog.Model("anthropic/claude-opus-5")
if err != nil {
	return err
}

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

## 5. `ai.NewDriver` and `ai.Wrap`

Execution policy — retries, cost metering, caching, logging — is `Middleware`
decorating a driver, so it needs the driver as a value. `NewDriver` is
`NewClient` stopping one step earlier:

```go
driver, err := ai.NewDriver(cfg)
if err != nil {
	return err
}

client := ai.New(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), cfg.Model)
```

`Retry` is the one policy shipped here, and every driver disables its vendor
SDK's own, so without it you get none. Order is outermost first: the first
middleware in the list sees the request first and the response last.

A `Middleware` is a `Handler` wrapping a `Handler`, which is the same shape as
`Driver.Stream` — so anything you write composes with anything shipped here.

## 6. `ai.New` with your own driver

`ai.New` takes a driver and a model and does no I/O, no lookup and no
validation of credentials. It is what the paths above end at, and it is what
tests want:

```go
client := ai.New(scripted{deltas: deltas}, ai.Model{
	ID: "test", API: ai.APIAnthropicMessages, MaxOutput: 1024,
})
```

Because `Driver` is two methods, the stub for a given case is a few lines. See
[CONTRIBUTING.md](../CONTRIBUTING.md#testing-your-own-code) for one.

## Defaults and overrides

Whichever path you took, the same `Option` is a default at construction and an
override at the call:

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh))

response, err := client.Complete(ctx, messages,
	ai.WithEffort(ai.EffortLow)) // this turn only
```

Resolution runs model defaults, then client defaults, then call overrides.
Passing an option is what marks a setting explicit, so `ai.WithTemperature(0)`
is deterministic sampling and omitting it inherits.
