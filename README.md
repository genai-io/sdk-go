# sdk-go

Go SDK for [San](https://github.com/genai-io/san): talk to any supported model
through one interface, and build agents on top of it.

```go
import (
    "github.com/genai-io/sdk-go/pkg/llm"
    "github.com/genai-io/sdk-go/pkg/llm/auth"
    _ "github.com/genai-io/sdk-go/pkg/llm/driver/all"
)

client, err := auth.Open("anthropic/claude-opus-5")   // key from ANTHROPIC_API_KEY

resp, err := client.Complete(ctx, &llm.Prompt{
    System:   "You are concise.",
    Messages: []llm.Message{llm.User("Explain goroutine leaks.")},
}, nil)
fmt.Println(resp.Content)
```

Swap the reference for `openai/gpt-5.6-terra`, `google/gemini-3.7-flash`,
`deepseek/deepseek-v4-pro` or `ollama/llama4` and nothing else changes.

## Layout

| Package | What it is |
| --- | --- |
| `pkg/llm` | Canonical types, the `Driver` seam, `Client` (streaming, aggregation, token counting) and `Provider` (model lists). **No third-party dependencies.** |
| `pkg/llm/driver/*` | One package per **wire protocol**: `anthropic`, `anthropicvertex`, `openaichat`, `openairesp`, `google`. Each owns whatever it takes to speak that protocol, and nothing else does. |
| `pkg/llm/jsonschema` | JSON Schema from Go types, and a validator. A leaf: it imports nothing outside the standard library, not even the rest of this SDK. |
| `pkg/llm/catalog` | Vendors and models **as data**: protocol, base URL, credential variables, reasoning dialect, limits, pricing — each entry stamped with the date it was last checked against the vendor's docs. |
| `pkg/llm/auth` | Optional. Resolves credentials from environment variables, and signs in to the vendors that use a browser instead of a key. |
| `pkg/llm/llmtest` | A scripted driver for testing code that calls a model. |
| `pkg/san` | Agent loop over `pkg/llm`: tools, turns, an event stream. |

### You only pay for the protocol you use

Vendor SDKs live in drivers and nowhere else, so what a program links is
decided by which drivers it imports. Non-standard-library packages pulled in:

| Import | Packages |
| --- | --- |
| `pkg/llm` alone | 2 |
| `+ driver/google` | 2 — plain HTTP, no vendor SDK |
| `+ driver/anthropic` | 14 |
| `+ driver/openaichat` or `openairesp` | 16 |
| `+ driver/anthropicvertex` | 130 — Google's ADC credential stack, gRPC, protobuf |
| `+ driver/all` | 144 |

Vertex AI genuinely needs Google's credential stack, so that driver is kept
separate and a program that does not use Vertex never links it. `driver/all` is
a convenience for a program that wants every protocol; anything narrower should
import the drivers it uses.

### A vendor is a row, not a package

There are twenty-five vendors and five protocols, because most vendors ship
an endpoint that speaks somebody else's. DeepSeek, Moonshot, Z.ai, SenseNova,
Alibaba, Agnes-AI and a local Ollama all speak OpenAI Chat Completions; MiniMax,
Xiaomi MiMo and Volcengine Ark speak Anthropic Messages; so do OpenRouter,
xAI, Groq, Cerebras, Together, Fireworks, NVIDIA and Hugging Face on the
OpenAI side. So the code is organised by protocol, and the differences between
vendors — base URL, environment variable, how the reasoning switch is spelled —
live in `catalog/vendors.go`. The nine gateway vendors added most recently
needed **no code at all**.

Adding an OpenAI-compatible provider is a table entry:

```go
{
    ID: "acme", DisplayName: "Acme", Order: 150,
    API:        llm.APIOpenAIChat,
    BaseURL:    "https://api.acme.ai/v1",
    BaseURLEnv: "ACME_BASE_URL",
    KeyEnv:     []string{"ACME_API_KEY"},
    Compat:     llm.OpenAIChatCompat{Thinking: llm.ThinkingEffort},
    Reasoning: []llm.ReasoningLevel{
        {Effort: llm.EffortOff},
        {Effort: llm.EffortLow, Value: "low"},
        {Effort: llm.EffortHigh, Value: "high", Default: true},
    },
},
```

## Listing models

Reading a provider's list is **synchronous and cannot fail** — it returns what
is known now, which before the first refresh is the catalog baseline. Fetching
is a separate, explicit verb, so a model picker renders immediately and
refreshes behind the user instead of blocking on a round trip a dead endpoint
can hang.

```go
p, err := auth.Provider("ollama")     // credential + endpoint from the environment

p.Models()                            // sync, never fails, never blocks
p.Model("llama4")                     // an unlisted ID still resolves, carrying the protocol
err = p.Refresh(ctx)                  // explicit fetch; a failure keeps the previous list

client, err := p.Open("llama4")
```

Across providers, refresh fans out and reports per-provider failures rather
than returning one error — one dead endpoint must not empty the list:

```go
set := auth.Providers()               // every vendor with a usable credential
result := set.Refresh(ctx)
for id, err := range result.Errors {
    log.Printf("%s: %v", id, err)     // the rest still refreshed
}
```

The merge is **field by field, not entry by entry**. Most OpenAI-compatible
endpoints publish an ID and nothing else; replacing a baseline entry wholesale
would discard its pricing, reasoning ladder and protocol quirks, and a model
stripped of its quirks stops working. The endpoint wins on every field it
stated; the baseline fills the rest.

## Streaming

`Stream` is a range-over-func iterator. It ends after `EventDone`, or right
after an error.

Text arrives a **block** at a time — a start, some deltas, an end — so a
consumer can tell "the model started a second paragraph" from "it kept
typing", which is the difference between opening a new bubble and appending to
the last one. `Event.Index` keys the block; `EventTextEnd` carries the whole
of it.

```go
for event, err := range client.Stream(ctx, prompt, opts) {
    if err != nil {
        return err
    }
    switch event.Type {
    case llm.EventTextStart:
        ui.OpenBlock(event.Index)
    case llm.EventTextDelta:
        ui.Append(event.Index, event.Text)
    case llm.EventTextEnd:
        ui.RenderMarkdown(event.Index, event.Text)   // now it is safe to
    case llm.EventToolCall:
        go run(event.ToolCall)          // arrives before the turn ends
    case llm.EventDone:
        usage = event.Response.Usage
    }
}
```

Only some protocols report where a block ends; Anthropic does, and the rest
have boundaries inferred from a change of kind. A block is always closed —
including when the turn fails partway, so a consumer's render state is never
left half-open.

## Reasoning

Effort is normalized to `off`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max`. What each rung actually puts on the wire is **model data, not driver
code**: a model carries an ordered `[]ReasoningLevel`, and each rung holds both
the normalized effort and the literal its endpoint wants.

```go
// Claude 4.6+: output_config.effort takes a level string
{Effort: llm.EffortHigh, Value: "high", Default: true}

// Claude 4.5 and the Anthropic-compatible third parties: a token budget
{Effort: llm.EffortHigh, Budget: 128_000}
```

Both halves have to be there. A rung carrying only the vendor's literal would
force callers to learn each vendor's vocabulary; a rung carrying only the
normalized effort pushes the mapping back into driver code. What stays in code
is only *which request field* the value goes in — `Compat.Thinking` — because
that is a different axis: DeepSeek's "on" is a `reasoning_effort` string and
its "off" is a `thinking` object, two fields no single value could express.

Asking for a rung a model does not offer **rounds up**, then falls back down
only if nothing above exists — quietly reasoning less than asked is the more
surprising failure.

```go
resp, err := client.Complete(ctx, prompt, &llm.Options{Effort: llm.EffortHigh})
```

## Tools from Go types

A hand-written `map[string]any` schema and the struct the arguments decode into
are two descriptions of one shape, and nothing keeps them in step. Deriving one
from the other removes the second description:

```go
type SearchArgs struct {
    Query string `json:"query" jsonschema:"description=what to look for"`
    Limit int    `json:"limit,omitempty"`
    Sort  string `json:"sort,omitempty" jsonschema:"enum=relevance|date"`
}

tool := llm.ToolFor[SearchArgs]("search", "search the web")
args, err := llm.UnmarshalArgs[SearchArgs](call)   // unknown fields rejected
```

**Arguments are validated before the tool runs.** They are model output, and
model output is wrong sometimes — a missing required field, a string where a
number belongs, an invented property. Running the tool anyway turns a mistake
the model could have corrected into whatever the tool does with nonsense: a
deletion with an empty path, a query with a null filter. `Tool.ValidateArgs`
turns it back into a tool error the model sees and retries, and `pkg/san` calls
it for you.

`ValidateJSONSchema` is not a complete JSON Schema implementation and does not
claim to be: it checks what a tool call actually gets wrong — required,
type, enum, unknown properties, recursively — and ignores the rest.

## Structured output

Getting data rather than prose out of a model otherwise means asking for JSON
in the prompt and scraping the reply — which fails in a long tail of ways that
all look like the model misbehaving: a markdown fence, a "Sure, here you go!"
preamble, a trailing paragraph, a truncated object. Every protocol here can
constrain generation properly, so none of that is necessary.

```go
type Person struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

person, err := llm.Parse[Person](client.Complete(ctx, prompt, &llm.Options{
    Schema: &llm.Schema{
        Name:        "person",
        Description: "the person described in the text",
        Definition: map[string]any{
            "type":                 "object",
            "properties":           map[string]any{
                "name": map[string]any{"type": "string"},
                "age":  map[string]any{"type": "integer"},
            },
            "required":             []string{"name", "age"},
            "additionalProperties": false,
        },
        Strict: true,
    },
}))
```

`Schema` maps onto each protocol's own mechanism — Anthropic's
`output_config.format`, OpenAI's `response_format` and `text.format`, Gemini's
`responseJsonSchema` — so the constraint is enforced during decoding rather
than requested politely.

`Response.Unmarshal` is deliberately lenient about what it accepts, because a
caller should not have to know which path produced the response they are
holding: it takes a bare JSON value, one inside a markdown fence, or one buried
in prose, tracking string literals so a brace inside a quoted value does not
end the scan early.

A model that cannot constrain output is **refused**, with the remedy named:

```
unsupported: model … cannot constrain output to a schema;
add llm.SimulateSchema() to ask for the shape in words instead
```

That middleware is opt-in for the same reason `SimulateSystemPrompt` is —
instructing is not constraining, and a caller who did not ask for the weaker
guarantee should not silently get it.

## Knowing whether the prompt fits

Without this, a caller learns a prompt was too large only by sending it: the
request is spent, the latency is spent, and an error arrives instead of an
answer.

```go
count, err := client.CountTokens(ctx, prompt, nil)
count.Exact   // true when the provider counted, false when this package estimated

left, _, err := client.Headroom(ctx, prompt, nil)   // tokens remaining in the window
```

Anthropic and Gemini publish a counting endpoint and their drivers use it. The
OpenAI-family protocols do not, so `EstimateTokens` fills in — measuring text by
script (a CJK character costs about a token where four bytes of Latin text do)
and images by pixel count read from the file header, because compression ratios
vary by orders of magnitude and payload size says almost nothing. It counts the
system prompt and the tool schemas too, which a dozen tools can make outweigh
the conversation.

The estimate leans towards over-counting, and says it is an estimate. A caller
comparing it against a window should leave headroom: being wrong that way
wastes a little context, and being wrong the other way wastes a request. An
unknown window reports **no** headroom rather than infinite — acting on a size
nobody knows is exactly what proactive compaction must not do.

## Capabilities are checked before the request

A model declares what it *cannot* do, and the request is validated against that
locally — so an unsupported ask fails with a sentence you can act on, and
nothing is spent on a call that was never going to work.

```go
_, err := client.Complete(ctx, promptWithImage, nil)
// unsupported: model deepseek/deepseek-v4-pro does not accept image input
llm.IsUnsupported(err)  // true — not retryable, not a credential problem
```

Sending an image to a text-only endpoint used to drop it silently, so the model
answered about a picture it had never seen. That is the failure this replaces.

`Unsupported` is stated as absences (`Tools`, `ToolChoice`, `System`,
`Multiturn`, `Streaming`) so its zero value is a fully capable model — a model
that arrived from a live listing carrying nothing but an ID is assumed capable,
not assumed crippled. Images are the one exception: vision has to be declared,
because guessing it wrong wastes a request.

### Retired models stay listed

A model the vendor no longer serves keeps its catalog entry, marked
`StageRetired` with a `Replacement`. Deleting it would turn a clear message
into an opaque 404 from the provider:

```go
// unsupported: model anthropic/claude-3-7-sonnet-20250219 is retired
// and no longer serves requests; use claude-sonnet-5
```

Filter them out of a picker with `llm.Available(models)`; leave them in for
lookup, because answering "that one is gone, use this" needs the entry to
still be there.

## Middleware

`Client` takes middleware around the model call, which is where retry, caching,
request logging and cost metering belong — policy stays with the caller, who is
the only one who knows the budget for a turn and what may be logged.

```go
client, err := llm.Open(cfg, llm.WithMiddleware(
    llm.Retry(llm.RetryPolicy{Attempts: 3}),
    myCostMeter,
))
```

`llm.Retry` only replays a call that failed **before producing any output**.
Once a delta has reached you the answer has begun; replaying would duplicate
what was already shown and discarding it would lose it, and neither is a
middleware's decision. It honours the provider's `Retry-After` when that fits
inside `MaxDelay` — a provider asking for twenty minutes is telling you to come
back later, not to block a goroutine for twenty minutes.

`llm.SimulateSystemPrompt()` folds a system prompt into an opening exchange for
a model with no system role. It is opt-in because the substitution is lossy: a
folded prompt is ordinary conversation the model may argue with or forget.

## Errors

Failures are classified, so a caller can decide what to do without matching on
message strings.

```go
switch {
case llm.IsContextExceeded(err):  // compact the prompt and retry
case llm.IsRetryable(err):        // back off; llm.RetryAfter(err) may have a hint
case llm.IsAuth(err):             // ask for a credential
}
```

**A failed turn still hands back what it produced.** `Complete` returns a
non-nil response alongside the error, carrying the text that already streamed
and the tokens already billed — a turn that died after 3k tokens still cost
3k, and discarding that loses both the partial answer and the accounting:

```go
resp, err := client.Complete(ctx, prompt, opts)
if err != nil {
    if resp != nil {
        show(resp.Content)          // what arrived before the failure
        meter.Add(resp.Usage)       // the spend happened either way
    }
    return err
}
```

A cancelled turn reports `StopAborted` rather than `StopError`: it is what the
caller asked for, not a failure to investigate.

### Overflow that does not announce itself

Most providers say "prompt is too long" and `IsContextExceeded` catches it. Two
behaviours do not: some endpoints accept an oversized prompt and answer anyway,
and others truncate the input to fill the window exactly and return a length
stop with zero output. Both look like a normal answer, and a caller that only
checks the error keeps resending a prompt that will never fit.

```go
if llm.IsOverflow(resp, client.Model()) {
    // compact and retry — including the two silent cases
}
```

## Credentials

`pkg/llm` never reads the environment or the filesystem — a `Config` carries the
key it is given, which is what makes it safe in a server serving several
tenants. `pkg/llm/auth` is the opt-in convenience that resolves keys from the
variables each vendor documents:

```go
cfg, err := auth.Config("deepseek/deepseek-v4-pro")   // DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
client, err := llm.Open(cfg)
```

Or bypass it entirely:

```go
model, _ := catalog.Model("deepseek/deepseek-v4-pro")
client, err := llm.Open(llm.Config{Model: model, APIKey: keyFromVault, HTTPClient: instrumented})
```

### Vendors that authenticate a person, not a service

GitHub Copilot and a ChatGPT subscription have no API key to paste — they have a
subscription and a browser. `auth.Login` runs the grant each one uses and stores
the result, so it happens once rather than every run:

```go
_, err := auth.Login(ctx, "copilot", auth.LoginOptions{})       // device code
_, err := auth.Login(ctx, "openai-codex", auth.LoginOptions{})  // browser, PKCE

cfg, err := auth.Config("copilot/gpt-5.1")   // afterwards, indistinguishable from a key
```

`auth.Interactive(vendorID)` reports which vendors work this way, so a CLI can
offer "sign in" instead of prompting for a key it will never get.

Two things are handled underneath. Copilot's usable token lasts about half an
hour — well inside a long session — so the credential is presented through an
`http.RoundTripper` that renews it mid-session rather than resolving it once at
start-up; what persists is the durable GitHub token, not the half-hour one.
And Copilot only reveals which endpoint your account talks to *after* you
authenticate — an enterprise account's differs from an individual's — so the
sign-in records it and `Config` uses it.

Credentials go to `$XDG_CONFIG_HOME/genai-io/credentials.json` (0600, in a 0700
directory, written and renamed so an interrupted save cannot truncate working
credentials). `Store` is an interface and `auth.DefaultStore` selects the
implementation: a CLI wants the file, a server wants its own secret manager.

## Protocol quirks are typed

A model carries its endpoint's behavior as a typed struct rather than a bag of
strings, so a quirk is a field with a name and a doc comment:

```go
llm.AnthropicCompat{ForceAdaptiveThinking: true, NoTemperature: true}
llm.OpenAIChatCompat{Thinking: llm.ThinkingType, ReasoningContent: true}
llm.OpenAIResponsesCompat{Stateless: true}
llm.GoogleCompat{ThinkingLevel: true}
```

Read it with `llm.CompatOf[llm.AnthropicCompat](model)`, which yields the zero
value when a model states none — so "not stated" and "all defaults" are the
same thing. Every field's zero value is the first-party behavior, so a
third-party endpoint states only its differences.

`Model.SamplingParams` and `Options.SamplingParams` are merged into the request
body verbatim (caller over model), which is how a llama.cpp or vLLM server
receives `top_k`, `min_p` or `repetition_penalty` without this SDK modelling
them.

## Reaching past the neutral options

`Options` models what every protocol has. For what only one protocol has,
`Options.Native` takes that driver's `Native` value — so needing one setting
does not mean writing a driver:

```go
client.Complete(ctx, prompt, &llm.Options{
    Effort: llm.EffortHigh,
    Native: anthropic.Native{ThinkingDisplay: "omitted"},   // faster first token
})
```

A `Native` for a different protocol is ignored rather than failing the request,
so the same `Options` stay usable when you swap the model underneath them.
`Options.ForceTool` is in core rather than `Native` because every protocol
expresses it. So is `Options.CacheRetention` (`none` / `short` / `long`): an
Anthropic 1-hour cache write costs twice the input rate where a 5-minute one
costs 1.25x, so it pays off from the second read rather than the first — that
is a real cost lever, and `Usage.CacheWrite1h` carries the slice back so the
bill can be computed correctly.

## Agents

```go
agent, err := san.New(
    san.WithModel(client),
    san.WithSystem("You are a build assistant."),
    san.WithTools(tools),
    san.WithMaxSteps(20),
)
result, err := agent.ThinkAct(ctx)
```

See `examples/` for runnable programs.

## Status

Early development, and the API will change. See
[ADR-0003](docs/design/decisions/0003-llm-package-layout.md) for why the
packages are shaped this way, and [issue #1](https://github.com/genai-io/sdk-go/issues/1)
for the roadmap.

Catalog entries were verified against each vendor's published documentation on
2026-08-20; every vendor records that date in its `Verified` field, and
`catalog.Stale` reports entries that have drifted out of date. Where a vendor
publishes no per-model limits (Alibaba Model Studio, SenseNova, Volcengine Ark)
the entry says so in its `Note` rather than carrying a guess.

Known gaps: interactive logins (GitHub Copilot, ChatGPT subscription) are not
implemented — supply an already-exchanged token as `Config.APIKey`. Anthropic's
Vertex and Bedrock backends are not covered.

## License

[Apache 2.0](LICENSE)
