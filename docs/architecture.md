# Architecture of `pkg/ai`

Three documents, three jobs. The README says **how to use** the library.
`go doc` carries the reasoning for each decision **next to the code** it
governs. This one is the part no single file owns: **how the pieces fit** —
which package knows what, which direction the dependencies run, and what a
request passes through between your call and the bytes on the wire.

## The thesis

**A package is a unit of code, so there is one per wire format and none per
vendor.**

27 vendors in the catalog are served by 5 protocols:

| Protocol | Package | Vendors |
| --- | --- | --- |
| OpenAI Chat Completions | `driver/openai/chat` | 18 |
| OpenAI Responses | `driver/openai/responses` | 3 |
| Anthropic Messages | `driver/anthropic` | 4 |
| Anthropic on Vertex AI | `driver/anthropic/vertex` | 1 |
| Google Gemini | `driver/google` | 1 |

Most vendors ship an endpoint speaking somebody else's protocol. DeepSeek,
Moonshot and Ollama speak OpenAI Chat Completions; MiniMax, Xiaomi MiMo and
Volcengine speak Anthropic Messages. What actually distinguishes them is a base
URL, an environment variable, a reasoning dialect and a list of models — four
things that are all data.

So **a vendor is a row in `catalog/vendors.go`, not a package.** Adding an
OpenAI-compatible endpoint is an entry in a table. Only a new wire format needs
Go code.

The test of whether this holds: no driver contains a vendor name. Every request
builder branches on `Compat` fields and `ReasoningLevel` data instead.

## The layers

The chain a caller walks down:

```
catalog.Vendor      data - a row you can read without a network
      | .Provider(cfg)
      v
provider.Provider   that row configured and credentialed, with a live model list
      | .Client(modelID)
      v
ai.Client           one model
```

Each rung adds one kind of knowledge and delegates down. `catalog` needs no
network; `provider` reaches one; `Client` runs one call.

Every step down is named for what it hands back, so the chain reads the same
whichever rung you enter it at:

| From | Call | You get |
| --- | --- | --- |
| `catalog.Vendor` | `.Provider(cfg)` | `*provider.Provider` |
| `provider.Provider` | `.Client(modelID)` | `*ai.Client` |
| `ai.Config` | `ai.NewClient(cfg)` | `*ai.Client` |
| `ai.Config` | `ai.NewDriver(cfg)` | `ai.Driver` |
| a reference | `auth.Client(ref)` | `*ai.Client` |
| a vendor ID | `auth.Provider(id)` | `*provider.Provider` |

`ai.New(driver, model)` is the one that follows Go's own convention instead:
`New` returns the package's main type from parts you already hold, where
`NewClient` resolves them from a `Config`.

Two package-level ways in, and the difference is whether the environment is
allowed to answer:

```go
ai.NewClient(Config)              // you supply the model, the key and the host
auth.Client("vendor/model")  // the catalog and the environment supply them
```

`pkg/ai` reads no environment variable and no file. That is what makes it safe
in a server holding several tenants' keys. `pkg/ai/auth` is the opt-in that
does read them, which is what a command-line tool wants. The split is an import
boundary, not a convention.

**Protocol is a second axis, not the next rung.** A Provider *has* a protocol;
it is not one. Put the two axes on one picture and the driver's place is
obvious — it hangs off the model, not off the chain:

```
  auth.Client("deepseek/deepseek-v4-pro")
        |
        v
  catalog.Vendor ---> provider.Provider ---> ai.Client ---> ai.Driver ---> HTTPS
   a row of data       host + credential      one model     one protocol
   no network          + live model list          |             ^
                                                  |             |
                                                  +-------------+
                                              Model.API selects it
```

`Model.API` — never the vendor name — is what picks the driver. That is the
whole reason a vendor can be data: everything else about DeepSeek is a base
URL, an environment variable and a reasoning dialect.

The cardinality is what keeps the two words apart:

```
1 protocol  ->  many providers     18 vendors speak OpenAI Chat Completions
1 provider  ->  many models
1 model     ->  1 client
```

So **provider** names a configured host you can reach, and **protocol** names
the request shape it speaks. `ProtocolOptions` scopes to the second, which is
why `anthropic.Options` reaches MiniMax and Volcengine too: they speak that
protocol without being that vendor.

## 27 vendors, 5 protocols, 5 driver packages

This is the thesis as a picture. The left column is data; only the right
column is Go code.

```
  catalog rows                    Model.API                  driver package
  ----------------------------    -----------------------    --------------------------
  deepseek    moonshot    zai  |
  alibaba     bigmodel    xai  |
  ollama      groq      nvidia |
  cerebras    together  agnesai+--> openai-chat-completions --> driver/openai/chat
  fireworks   copilot  sensenova         18 vendors
  openrouter  huggingface      |
  bedrock-openai               |

  anthropic   minmax           |
  mimo        volcengine       +--> anthropic-messages      --> driver/anthropic
                                        4 vendors

  openai      azure-openai     |
  openai-codex                 +--> openai-responses        --> driver/openai/responses
                                        3 vendors

  anthropic-vertex              --> anthropic-vertex        --> driver/anthropic/vertex
  google                        --> google-genai            --> driver/google
```

Adding a nineteenth OpenAI-compatible vendor moves the left column only.

## The core primitive: ordered blocks

A message carries `Content []Block` — a tagged union in sequence — rather than
parallel `Text`, `Images`, `ToolCalls` fields.

```go
//	Type              carried in         produced by
//	BlockText         Text               either
//	BlockImage        Image              you
//	BlockThinking     Text, Signature    the model
//	BlockToolCall     ToolCall           the model
//	BlockToolResult   ToolResult         you
//	BlockReasoning    Reasoning          the model
```

The order is the point. A model that thought, called a tool, then explained
itself produced three blocks in that sequence, and the next request has to
carry them back the same way. Parallel fields lose it, and no protocol accepts
a conversation whose order was reconstructed by guesswork.

This is also why `Response.Message()` exists and why appending
`ai.AssistantMessage(resp.Text())` instead is a bug: the former carries the
thinking and the opaque reasoning state forward, and dropping those makes a
reasoning model start over every turn.

Measured: interleaved content survives `anthropic`, `openai/responses` and
`google` in order. `openai/chat` flattens it — text is joined into one string
and calls move to a parallel `tool_calls` array — because that protocol cannot
express the interleaving. That is a property of the wire format, not a bug in
the driver.

## One conversation, and options

```go
client.Complete(ctx, messages, ai.WithSystem(s), ai.WithEffort(ai.EffortHigh))
```

The conversation is an ordinary `[]ai.Message`. Everything else is an `Option`,
and the same `Option` is a default at `ai.New` and an override at the call — so
a setting is spelled one way wherever it is set.

**There is no presence wrapper on the option fields.** Applying an option *is*
the presence bit: `WithTemperature(0)` is deterministic sampling and omitting
it inherits. An earlier design used `Optional[T]` with `Some(...)` for exactly
this distinction; it existed solely to serve a two-struct call shape and went
away with it.

Resolution is three layers, applied in order: model defaults, client defaults,
call overrides. Later overwrites earlier. That is the whole rule.

## Normalized, or native

The dividing line for every setting:

| | Where it goes |
| --- | --- |
| Every protocol can express it, spelled differently | a `With*` option; the driver translates |
| Only one protocol has it | that driver's own `ProtocolOptions` value, passed through |

Reasoning effort is the clearest case of the first. "How hard should it think"
exists everywhere, and every vendor spells it differently — Anthropic wants
`thinking.budget_tokens` or `output_config.effort`, Gemini wants
`thinkingLevel`, DashScope wants `enable_thinking` plus `thinking_budget`. So
it is `ai.WithEffort(ai.EffortHigh)`, and each `Model` carries a
`[]ReasoningLevel` ladder mapping the rung onto what its endpoint wants.

The ladder is data, not code: no driver contains an effort table. A model may
also declare a rung this package has never heard of; asking for it by exact
name sends it. A name that is neither portable nor in that model's ladder is
refused, and the error names what the model does offer.

`thinking.display` is the second case. No other protocol has the concept, so it
is `anthropic.Options.ThinkingDisplay` and travels untranslated.

These are typed, not `any`: `ai.ProtocolOptions` and `ai.ProtocolConfig` are
marker interfaces, so a value that was never meant to go there is a compile
error. A value of the *wrong driver's* type is caught at the moment that driver
reads it — Go has no union type, so that half cannot be a compile error.

## Protocol dialects

Two endpoints can speak the same protocol and still disagree. `compat.go` is
where an endpoint states its differences, as a value:

```go
Compat: ai.OpenAIChatCompat{Thinking: ai.ThinkingEffortOrDisable}   // DeepSeek
Compat: ai.OpenAIChatCompat{Thinking: ai.ThinkingType,
                            ReasoningContent: true}                 // Moonshot
Compat: ai.AnthropicCompat{BearerAuth: true}                        // Volcengine Ark
```

This is what keeps one driver from growing into a tree of vendor special cases.
`applyReasoning` in `driver/openai/chat` serves 18 vendors as a pure switch on
`ThinkingFormat`, with no vendor literal in it.

Compat values are authored in `catalog` and read in the drivers, and `pkg/ai`
itself reads them during validation. That is why the types live in `pkg/ai`:
it is the only package all three can depend on. Moving them into the driver
packages would force `catalog` to import every driver, and with it every
vendor SDK — destroying the blank-import selectivity the registry exists for.

## What the SDK will not do

It does no retry. Retry, caching, logging and cost metering are `Middleware`
decorating a driver (`ai.Wrap(driver, retry, meter)`), because only the
application knows the budget for a turn, what may be cached and what must not
be logged. One rule is not the caller's to discover: a retry may only replay a
call that failed *before producing any delta*.

It does no compaction. Deciding a conversation is too long, summarizing it or
dropping the oldest turns are the application's calls, with knowledge this
package does not have. What it does do is *repair*: `RepairHistory` removes
only what the protocol itself would reject — an unanswered tool call left by a
Ctrl-C, invalid UTF-8 from a conversation that passed through a JavaScript
runtime. Repair is not policy.

It does not rewrite a request to make it work. A model with no system role, or
one that cannot constrain output to a schema, is an error naming the model.
Moving the instructions into a user turn, or asking for JSON in words, is a
decision about the product, not about the wire.

It does not guess. A model whose context window is unknown reports zero
headroom rather than a substituted number, because acting on a guessed limit
fails silently in both directions.

## What one call passes through

Between `client.Complete(ctx, messages, opts...)` and bytes on the wire:

```
  messages + options
        |
        |  +---------------------- Client.prepare ------------------------+
        +--| newRequest         model defaults -> client -> call, in order |
        |  | validateStructure  the tagged-union invariants a Block holds  |
        |  | RepairHistory      pair tool calls, replace invalid UTF-8     |
        |  | validate           settings, then what this model cannot do   |
        |  +--------------------------------------------------------------+
        v
    *ai.Request  ------------->  Driver.Stream
                                      |   translate to the wire format,
                                      |   send, and yield while reading
                                      v
                               iter.Seq2[Delta, error]      raw fragments
                                      |
                                      |   blockTracker assembles
                                      v
                               iter.Seq2[Event, error]      ordered blocks
                                      |
                                      v
                                 *ai.Response               Complete drains this
```

Everything in the box happens once, for every protocol. That is the line a
driver sits behind: it translates and streams, and does nothing else. It is
also why `Client.Complete` is not a second code path — it drains `Stream`, so
a request reaches an endpoint exactly one way.

The same `*ai.Request` is what `CountTokens` measures, so a prompt cannot be
sized differently from how it is sent.

## Failing before the network

Three layers of validation run before a request leaves, in this order:

1. **structure** — the tagged-union invariants a `Block` must satisfy, and the
   protocol invariants a block replay must satisfy;
2. **settings** — values malformed on their own terms, or contradicting the
   conversation they were sent with;
3. **capability** — what this particular model declares it cannot do.

The point is a sentence a caller can act on — "model deepseek-v4-pro does not
accept image input" — instead of an opaque provider rejection or, worse, a
silently degraded answer. Sending an image to a text-only endpoint used to mean
the image was dropped and the model answered about something it had never seen.

Failures that do reach the network come back classified, so the answer to "what
now" is in the type: `IsAuth`, `IsContextExceeded`, `IsRetryable`,
`IsUnsupported`. A failed turn returns both an error *and* a `*Response`
carrying what arrived first, so a partial answer and its cost survive.

## Boundaries the compiler enforces

Some rules here are conventions and some are checked. It is worth knowing which:

- **The dependency graph runs one way, and `pkg/ai` is the root.** It imports
  nothing else in this repository:

  ```
  indentation is "imports the line above"

  pkg/ai                  the core. No repo imports, no vendor SDK,
                          no environment variable, no file.
    |
    +-- pkg/ai/driver/*   one package per protocol. Each pulls in that
    |                     protocol's vendor SDK, and nothing else does.
    |
    +-- pkg/ai/provider   one configured host and its live model list
          |
          +-- pkg/ai/catalog    the vendor table
                |
                +-- pkg/ai/auth    env, files, OAuth
  ```

  `catalog` depends on **no** driver package — its whole tree is `pkg/ai` plus
  `pkg/ai/provider`. That is what lets a program talking to one vendor avoid
  linking every vendor's SDK, and it is why the `Compat` types live in `pkg/ai`
  rather than beside the drivers that read them.
- `driver/openai/internal/errs` is unreachable from `driver/anthropic`, by
  `internal/`.
- `ProtocolOptions` / `ProtocolConfig` reject a value that was never meant to be
  one, at compile time.
- `Driver` is two methods, so a stub in a test is two methods.

And some are only documented, which means only review catches them:

- Drivers must not mutate or retain the `*Request` they are given. The `Client`
  reads it again afterwards.
- Middleware must not replay a call that has already produced a delta.

## Known warts

Recorded because a design document that only lists wins is not one.

**`APIAnthropicVertex` is not a wire protocol.** Four of the five `API` values
are genuinely distinct request shapes. The fifth is the Anthropic Messages
format with different authentication (Google ADC), a different host and a
different model-ID form; the driver hands everything downstream of client
construction back to `driver/anthropic` unchanged. It is a separate `API` value
because the registry keys on `API` and its Google Cloud auth dependency is
heavy — 271 third-party packages against `anthropic`'s 43 — so it must land only in a
build that asks for it. Fixing it properly means a second dimension in the
registry, which costs more than the wart.

**OpenAI cache-write tokens are not counted.** The endpoint reports them in
`input_tokens_details.cache_write_tokens`, which the pinned `openai-go` release
does not expose, so on GPT-5.6 and later those tokens are priced as ordinary
input — about a quarter under the real figure. The vendor entry says so.

**`ResolveLevel` snaps silently.** Asking for `medium` on a model offering only
off and high returns high, and nothing tells the caller. The direction is
deliberate — quietly reasoning *less* than asked is the more surprising failure
— but the silence is not something a caller can currently observe.

## Testing

One black-box package, `test/`. It imports the SDK the way an application does
and asserts on two things only: **the bytes that reached the endpoint, and the
value that came back.** Every endpoint is a stub HTTP server, so the suite needs
no network and no credential.

That shape is deliberate. Tests that reach inside verify the implementation
they were written against; these verify the contract, so a driver rewritten
behind the same wire behaviour keeps passing.
