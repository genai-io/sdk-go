# ADR-0003: Organise the LLM SDK by wire protocol, not by vendor

- Status: accepted
- Date: 2026-08-20
- Supersedes the placeholder types in `pkg/llm` added in #3

## Context

`pkg/llm` was a copy of San's internal type definitions with no implementation
behind it: a `Provider` interface, a `Client`, and no way to reach a model
without writing a provider yourself. Everything that actually talks to a model
lives in `san/internal/llm` — roughly 15k lines covering fourteen vendors.

Reading that code, the fourteen vendor packages are not fourteen
implementations. Ten of them are ~150 lines of configuration over one shared
`openaicompat` layer: a base URL, an environment variable, a reasoning toggle,
and a table of context windows. Three more (MiniMax, Xiaomi MiMo, Volcengine)
are the same over the Anthropic SDK. Only four request/response shapes exist in
the whole tree.

The San packages also carry things a public SDK must not: a settings store
under the user's home directory, a package-level singleton connection, a global
provider registry keyed by `provider:auth_method`, TUI display ordering, and
OAuth login flows.

## Decision

Three layers, split along the seam that actually exists.

**`pkg/llm`** — canonical types, the `Driver` seam and `Client`. Zero
third-party dependencies, no environment access, no filesystem access.

**`pkg/llm/driver/{anthropic,openaichat,openairesp,google}`** — one package per
wire protocol. Each owns exactly one vendor SDK, translates a `Request` into
that protocol and streams `Delta` values back. Drivers register themselves by
`llm.API` from `init`, the `database/sql` pattern, so a blank import is what
makes a protocol reachable and a program that talks to one provider does not
link the other three SDKs.

**`pkg/llm/catalog`** — vendors and models as data. A vendor is a row: display
name, protocol, base URL, credential variables, reasoning dialect, models,
pricing. Adding an OpenAI-compatible endpoint is a table entry.

**`pkg/llm/auth`** — a separate, optional import that resolves credentials from
the environment.

### Consequences worth stating

*Aggregation moved into `Client`.* Drivers emit deltas and nothing else;
joining text, ordering tool calls and reconciling usage happens once for all
four protocols. In San each provider re-implemented this through a shared
`streamutil.State`, which is why the Anthropic path and the OpenAI path drifted
into two different tool-result sanitizers.

*Reasoning effort is normalized to `off | low | medium | high`.* Vendors spell
it as token budgets, level strings and boolean flags; the driver maps, and a
request runs unchanged across providers. `Reasoning.Resolve` snaps an
unsupported level onto the nearest one the model offers rather than silently
falling back to the default — asking a two-state endpoint for "medium" turns
reasoning on.

*Tool results are grouped: `Message.ToolResults` is a slice.* San stored one
result per message and re-merged them inside the Anthropic driver. The wire
format wants them together, so history holds them that way and no driver merges.

*Message content is an ordered `[]Part`.* This replaces San's `Content` +
`Images` + `DisplayContent` with `[Image #N]` placeholder parsing, which existed
only to recover the ordering the flat fields threw away.

*Usage always means the same thing.* `Input` is fresh prompt tokens, with the
cached prefix under `CacheRead`/`CacheWrite`. OpenAI-family protocols report one
combined figure; their drivers split it, so `TotalInput` equals what the API
reported and per-turn sums never double-count a re-read cache.

*Unknown limits stay zero.* A model whose context window cannot be determined
reports 0, and callers must treat that as "cannot size" rather than substituting
a guess. Guessing low burns context on every compaction; guessing high never
fires at all.

## Alternatives considered

**Port the fourteen vendor packages as they are.** Fastest, easiest to review,
and it freezes San's internal structure into a public API — including the
per-vendor boilerplate that a data table replaces.

**Make San depend on this SDK immediately.** The better end state, and the plan;
doing it in the same change would have coupled a two-repo migration to an API
that has not been used by anything yet. This round copies. See "Follow-ups".

**One Go module per driver**, so a consumer's module graph carries only the SDK
it uses. Real benefit, real cost in release machinery. Revisit if the dependency
weight becomes a complaint.

## Follow-ups

- Point San's `internal/llm` at this package and delete the duplicate.
- Interactive logins (GitHub Copilot device code, ChatGPT subscription) under
  `pkg/llm/auth`. The Copilot catalog entry documents that its token must be
  supplied by the caller until then.
- Anthropic Vertex/Bedrock backends, which San supports and this does not yet.
- ~~Refresh the catalog against each vendor's published specs.~~ Done
  2026-08-20; see the addendum below.

## Addendum, 2026-08-20: catalog verification

Every entry was checked against the vendor's own documentation and each vendor
now carries a `Verified` date, with `catalog.Stale` to report drift. Two
findings were not data at all:

*A protocol has more than one reasoning dialect, and which one a model speaks
is model data.* The Anthropic driver sent `thinking: {type: "enabled",
budget_tokens: N}` to every model. Claude 4.6 and later want `{type:
"adaptive"}` with the level in `output_config.effort`, and Opus 5 / Opus 4.7 /
4.8 / Sonnet 5 / Fable 5 reject a budget outright — so every current Claude
model would have failed with a 400 while the third-party Anthropic-compatible
endpoints (MiniMax, MiMo, Ark), which only implement the older shape, kept
working. Gemini 3 has the same split against 2.5 (`thinkingLevel` versus
`thinkingBudget`). Both are now `llm.OptReasoning` values on the model, which
is what a per-model dialect field was for.

*Advertising a capability the driver cannot deliver is worse than not having
it.* The first cut dropped `EffortOff` from the OpenAI ladder on the belief
that the Responses API takes only a level. It accepts `reasoning.effort:
"none"`, so "off" was being silently upgraded into reasoning the caller had
asked not to pay for.

Where a vendor publishes no per-model limits — Alibaba Model Studio, SenseNova,
Volcengine Ark — the entry now says so in its `Note` and reports zero rather
than carrying a figure nobody checked.


## Addendum, 2026-08-21: aligning with pi-ai

A field-by-field read of `@earendil-works/pi-ai` found the two SDKs already
agreed on the load-bearing decisions — protocol-not-vendor routing, adaptive
thinking as per-model data, static baseline plus dynamic overlay, a
credential-agnostic core. Everything below is where pi was further along and
the shape it uses was better than ours.

### The reasoning ladder became data

`Model.Reasoning` is now an ordered `[]ReasoningLevel`, each rung carrying both
the normalized `Effort` and what its endpoint wants — `Value` for a
level-taking field, `Budget` for a token-taking one. The per-dialect `switch`
in each driver is gone; adding a vendor's spelling is adding rungs.

Two halves are required, and it is worth writing down why, because "just use
the vendor's own strings" is the obvious simplification and it does not work.
A rung holding only the vendor literal makes the caller learn that Anthropic
says `think+` and OpenAI says `high`, and the same request stops running across
providers — which is exactly the state San is in. A rung holding only the
normalized effort pushes the mapping back into code. The normalized ladder is
unavoidable; pi defines one too. What we avoided was defining a *third* one, so
the rungs match pi's: `off | minimal | low | medium | high | xhigh | max`.

A slice rather than pi's map, because order is meaningful (clamping walks it),
a rung needs more than one value, and it serializes deterministically.

*Which request field* the value goes in stays in code, as
`OpenAIChatCompat.Thinking`. That is a genuinely different axis: DeepSeek's
"on" is a `reasoning_effort` string and its "off" is a `thinking` object — two
fields, which no single value could carry.

### Clamping rounds up

`Model.ResolveLevel` searches upward from the requested rung and only falls
back downward when nothing above exists. The previous nearest-neighbour policy
resolved `low` to `off` on a two-state endpoint — switching reasoning off for a
caller who asked for a little of it. pi's `clampThinkingLevel` had it right.

### Quirks are typed

`Options map[string]string` and its `Opt*` string constants are replaced by a
typed `Model.Compat any` holding one of `AnthropicCompat`, `OpenAIChatCompat`,
`OpenAIResponsesCompat` or `GoogleCompat`, read through
`CompatOf[T](model)`. Every field's zero value is the first-party behavior, so
"not stated" and "all defaults" are the same value and a third-party endpoint
states only its differences.

`any` rather than a type parameter: `[]Model` has to hold models of different
protocols, which `Model[T]` could not. The structs live in core rather than
beside their drivers so the catalog can build them without importing a vendor
SDK.

### Request split into Prompt and Options

`Prompt` is the conversation; `Options` is how to run it. A nil `*Options` is a
meaningful "use the defaults". The split is what makes a native-options tier
possible later — same Prompt, a different options type — and it is why pi can
ship `stream()` and `streamSimple()` over one conversation type.

Named `Prompt`, not `Context`: Go already has one of those in every signature
here, and the collision would be a daily cost.

### Smaller alignments

- `Input []Modality` replaces `SupportsImages bool`. Providers keep adding
  kinds; a caller asking `Accepts(ModalityAudio)` of a model that predates
  audio gets a correct answer without the model needing a field for it.
- `Options.ToolChoice` — "you must call a tool" was previously unexpressible.
- `Model.SamplingParams` and `Options.SamplingParams`, merged caller-over-model
  into the request body verbatim, so a llama.cpp or vLLM server can receive
  parameters this SDK does not model.
- `Model.Headers` — per-model headers, under Config headers of the same name.
- `Pricing.Tiers` — request-wide rate switches above a prompt-size threshold.
  MiniMax bills M3 at double above 512k input tokens, which the flat card
  documented as inexpressible.
- `Usage.Reasoning` — thinking tokens as a subset of Output, where reported.
- `ClassifyMessage` now checks a non-overflow exclusion list first. Bedrock
  formats throttling as "ThrottlingException: Too many tokens…", which matched
  the overflow signature and turned a retryable 429 into "compact your prompt".

### Listing models split into a read and a fetch

`llm.Provider` holds a static baseline and a fetched overlay. `Models()` is
synchronous and cannot fail; `Refresh(ctx)` is the explicit fetch, and a
failure leaves the previous list intact. `llm.Providers.Refresh` fans out and
returns a per-provider error map rather than one error, because one dead
endpoint must not empty the list. `catalog.Vendor.Provider` seeds a provider
from the catalog; `auth.Provider` adds the credential.

The merge is **field by field**, which is where this deliberately departs from
pi. pi's `createProvider` replaces a matched baseline entry wholesale, and can,
because its dynamic providers construct complete `Model` objects. Ours get raw
listings from `Driver.Models` carrying an ID, a name and sometimes a window —
so a wholesale replace would strip a model of its `Compat`, and a model without
its protocol quirks silently stops working. The endpoint wins on every field it
stated; the baseline fills the rest.

`Provider.Model` also answers for an ID it has never seen, decorated with the
provider's protocol and endpoint: an unlisted model is nearly always one newer
than the catalog, not one that does not exist.

### A native escape hatch, without a second entry point

pi ships every protocol twice — `stream()` with typed per-API options and
`streamSimple()` with neutral ones. Go has no conditional types to make the
first one type-safe through a shared `Client`, and a parallel entry point would
lose Client's aggregation.

Instead `Options.Native any` carries that driver's `Native` value, read with
`NativeOf[T]`. One entry point, aggregation intact, and a `Native` meant for
another protocol is ignored rather than failing — so the same `Options` survive
swapping the model underneath them.

`Options.ForceTool` went into core rather than `Native`: all four protocols
express "call this specific tool", and it was previously unreachable.

### Nine vendors, no code

OpenRouter, xAI, Z.ai (international), Groq, Cerebras, Together, Fireworks,
NVIDIA and Hugging Face were added as catalog rows alone — twenty-three vendors
now against the same four protocols. The one thing that needed code was
`ThinkingReasoningObject`, because OpenRouter nests the level under
`reasoning: {effort}` rather than sending `reasoning_effort` flat, and that is
a request-field question the ladder cannot answer.

The aggregators state no vendor-wide reasoning ladder. They serve models from
many upstreams with different controls, so one ladder would be wrong for most
of what they host; a caller who knows their model states it on the Model. This
is the same rule as unknown context windows — say nothing rather than say
something unchecked. OpenRouter is the exception: it normalizes reasoning
itself, so one ladder is right across everything it serves.

### Vertex, without forking the driver

`APIAnthropicVertex` routes to its own driver, but that driver is sixty lines:
Vertex differs from first-party Anthropic in how the client authenticates and
where it points, and in nothing downstream of that. So package `anthropic`
gained `ClientOptions` and `NewWithClient`, and `anthropicvertex` builds a
client with Google credentials and hands it over. This is pi's answer too — its
`AnthropicOptions.client` exists to let an `AnthropicVertex` client be injected.

It is a separate package rather than a second factory in `anthropic` so its
Google Cloud auth dependency lands only in a build that asks for it.

The project and region arrive as `Config.Native`, a `VertexConfig` that lives
in core so `auth` can fill it from the environment without importing the driver.

Adding Vertex also made every Claude model ambiguous by bare name, which was a
real regression — `catalog.Model("claude-opus-5")` started failing for
everyone. The rule now is that **a vendor needing deployment configuration is
never what a bare name means**: nobody typing a model ID means "the Vertex
deployment", and naming the vendor is how that choice is made. Genuine
ambiguity between two direct vendors still reports rather than picking one.

### A failed turn hands back what it produced

`Complete` returns a non-nil response alongside the error, and the streaming
path yields both in one event. A turn that died after 3k tokens still cost 3k;
returning only an error discarded the partial answer and the accounting for
spend that had already happened. `StopAborted` separates a cancellation — what
the caller asked for — from a failure to investigate. `pkg/san` now counts a
failed step's tokens for the same reason.

This is pi's contract restated for Go. pi encodes failures in the stream as an
`AssistantMessage` and never throws; returning `(*Response, error)` keeps the
Go convention while carrying the same information.

### Overflow that does not announce itself

`IsOverflow(resp, model)` adds the two structural cases pi documents and we had
no answer for: an endpoint that accepts an oversized prompt and answers anyway
(detectable only because it billed for more prompt than its window holds), and
one that truncates the input to fill the window and returns a length stop with
zero output. Both look like a normal answer; a caller checking only the error
keeps resending a prompt that will never fit.

### Smaller absorptions from pi

- `PrepareMessages` cleans invalid UTF-8 on the way out. A lone UTF-16
  surrogate — half a pair, three bytes that are not valid UTF-8 — arrives from
  anything that passed through a JavaScript runtime, and providers disagree on
  whether to reject it or return mojibake. pi sanitizes for the same reason.
- `Options.CacheRetention` (`none`/`short`/`long`) and `Usage.CacheWrite1h`.
  Anthropic bills a 1-hour cache write at twice the input rate where a
  five-minute one costs 1.25x, so the choice is a real cost lever and the
  split has to travel back for the bill to be right.
- `Response.ID` — the provider's own identifier, which is what a provider-side
  trace or a support ticket is keyed by.
- `Response.Model` now records what a gateway actually served, not what was
  asked for.

## Addendum, 2026-08-21: what Genkit Go was right about

Genkit is a framework where this is an SDK, so most of it is deliberately not
taken: flows, the plugin registry, the dev UI, OpenTelemetry tracing,
retrievers, embedders and evaluators all belong to a layer above this one, and
its `Config any` per-request escape is strictly worse than the typed
`Options.Native` already here. Four things it gets right that we did not.

**Capabilities are checked before the request.** Genkit's `validateSupport`
middleware refuses a request the model cannot serve rather than sending it.
Ours did the opposite and worse: a driver silently dropped images for a
text-only model, so the model answered about a picture it had never seen.
`Model.Unsupported` plus `Model.Validate` now refuses locally, with a sentence
naming the problem, before anything is spent.

The capability struct is stated as absences where Genkit's `ModelSupports`
states capabilities. Genkit can use positive naming because every plugin
declares support explicitly; our models also arrive from live listings carrying
nothing but an ID, and under positive naming those would validate as incapable
of everything. Absences keep the zero value "fully capable", which is both the
common case and the safe reading of an unknown model. Vision is the one
exception — it has to be declared, because guessing it wrong wastes a request.

**A lifecycle stage beats deleting the entry.** Genkit's `ModelStage` carries
featured/stable/unstable/legacy/deprecated. When the catalog was refreshed, the
retired Claude models were deleted, which turns "retired on 2026-02-19, use
claude-sonnet-5" into an opaque 404 for anyone still pointing at one. They are
back, marked `StageRetired` with a `Replacement`, and `llm.Available` filters
them out of a picker. Google's preview Gemini models are marked `StagePreview`
for the same honesty.

**A middleware seam.** Genkit wraps model calls, tool calls and the whole
generate loop; we own none of the tool loop in this package, so only the model
level applies. `Middleware` + `WithMiddleware` is where retry, caching, logging
and metering go — keeping policy with the caller, who alone knows the budget
for a turn and what must not be logged.

`llm.Retry` ships as the proof, and its restriction is the interesting part: it
replays only a call that failed *before producing output*. Once a delta has
reached the caller the answer has begun, and replaying would duplicate what was
shown while discarding would lose it — neither is a middleware's decision. That
is a narrower guarantee than a naive retry and the only correct one for a
streaming call.

**Simulating a missing system role.** Genkit's `simulateSystemPrompt` folds the
system prompt into an opening exchange for models without a system role.
`llm.SimulateSystemPrompt()` does the same, as opt-in middleware rather than a
default, because the substitution is lossy: a folded prompt is ordinary
conversation the model may argue with or forget, where a real system prompt is
weighted and cached differently. A caller who wants that trade makes it; one
who does not gets an error rather than a quietly weaker prompt.

Considered and not taken: Genkit's `GenerationUsage` counts characters, images,
videos and audio files alongside tokens, for providers that bill per character.
None of ours do, and the fields would be permanently zero — an invitation to
write cost code against numbers nobody fills in.

## Addendum, 2026-08-21: a first-principles audit

Setting the two comparisons aside and asking instead what an SDK must do for a
program to hold a conversation with a model, then walking the lifecycle of a
real session — pick a model, get credentials, build a prompt, check it fits,
send, stream, run tools, persist, resume, account for cost — turned up two
load-bearing gaps that neither comparison had surfaced. One of them was a bug
introduced by this very design.

**A Model did not survive being written to a session file.** `Compat` is an
`any`, and encoding/json turns it back into a `map[string]any`, so `CompatOf`
returned the zero value. A DeepSeek model reloaded from disk had lost its
thinking dialect: "reasoning off" stopped sending the field that switches it
off, and reasoning stayed on with nothing reporting it. This is precisely the
failure the provider merge is written to avoid — "a model stripped of its
quirks stops working" — and the same hole was left open in serialization.

`Model` now marshals and unmarshals explicitly, using `API` as the
discriminator for the compat type, and `RegisterCompat` lets a custom protocol
add its own. A protocol with no registered decoder is an **error**, not a
silent downgrade: a model whose quirks were dropped looks fine and misbehaves
later, which is the worse of the two outcomes.

**There was no way to ask whether a prompt fits.** Usage came back only after a
call, so a caller learned a prompt was too large by spending a request on it.
`IsOverflow` detects the failure after the fact; nothing prevented it.

`Client.CountTokens` uses the provider's own tokenizer where the protocol
publishes one — Anthropic and Gemini both do — and falls back to
`EstimateTokens` where it does not. `TokenCount.Exact` says which happened,
because the two are not interchangeable: an exact count can be compared against
the window directly, an estimate needs headroom.

The estimator is a heuristic and says so. It measures text by script rather
than by the four-bytes-per-token rule of thumb, which under-counts CJK about
fourfold — the direction that costs a request. It sizes images from the
dimensions in the file header rather than the payload size, because compression
ratios vary by orders of magnitude. It counts the system prompt and the tool
schemas, which a dozen tools can make outweigh the conversation. And
`Headroom` reports zero for a model whose window is unknown, rather than
infinite: acting on a size nobody knows is what proactive compaction must not
do.

Everything else in the lifecycle held up: messages round-trip losslessly,
cancellation propagates, abandoning a stream releases the connection, and
`Client` and `Provider` are safe for concurrent use.

## Addendum, 2026-08-21: structured output

The one gap all three passes — pi, Genkit, and the first-principles audit —
pointed at. Without it, a whole class of use (extraction, classification,
form-filling, routing) means asking for JSON in the prompt and scraping the
reply, and that fails in a long tail of ways that all present as the model
misbehaving.

All four protocols turned out to have native support, so no workaround was
needed: `output_config.format` on Anthropic, `response_format` on OpenAI Chat,
`text.format` on Responses, `responseJsonSchema` plus a JSON mime type on
Gemini. `Options.Schema` maps onto each. Gemini needs both halves — the mime
type alone promises valid JSON, not the shape.

Two decisions worth recording.

**`Response.Unmarshal` is lenient on purpose.** A natively constrained answer
is bare JSON and needs no leniency; an answer produced by `SimulateSchema`
needs a lot. Making the caller know which path produced the response they are
holding would push the whole distinction back onto them, so extraction accepts
a bare value, a fenced one, or one buried in prose — scanning for a balanced
object while tracking string literals, so a brace inside a quoted value does
not end it early.

**The fallback is opt-in, and refusal is the default.** A model that cannot
constrain output gets an error naming `SimulateSchema` as the remedy, rather
than being handed instructions and a hope. This is the same rule as
`SimulateSystemPrompt`: instructing is not constraining, and a caller who did
not ask for the weaker guarantee should not silently receive it.
`Unsupported.SchemaWithTools` covers the endpoints that can do either but not
both in one request.

## Addendum, 2026-08-21: typed tool arguments and content blocks

**Schemas are derived from Go types.** `JSONSchemaOf[T]`, `SchemaOf[T]` and
`ToolFor[T]` build a JSON Schema by reflection, reading `json` tags for names
and optionality and a small `jsonschema` tag for description, enum and forced
required. The reason is drift: a hand-written map and the struct the arguments
decode into are two descriptions of one shape, and nothing keeps them in step.

A type it cannot describe precisely is described *loosely* rather than wrongly
— an interface field becomes an unconstrained value — because claiming a shape
the type does not have is the worse failure. A self-referential type stops at
the cycle rather than recursing.

**Tool arguments are validated before execution.** This closes a real hole:
`pkg/san` previously decoded a tool call's arguments and ran the tool with
whatever came out, so a model that omitted a required path or invented a
property got the tool executed on nonsense. `Tool.ValidateArgs` turns that into
a tool error the model sees and corrects, which is what it does with one.

`ValidateJSONSchema` deliberately implements a subset: required, type, enum,
`additionalProperties: false`, recursing through objects and arrays. A full
draft-2020 validator is a library of its own, and the keywords beyond this set
are not the ones tool calls get wrong. Saying which subset is implemented is
better than either extreme.

**Content blocks have boundaries.** The event stream gained
`EventTextStart`/`EventTextEnd` and their thinking equivalents, with an
`Index`. Without them a consumer cannot tell a second paragraph from a
continuation, and cannot know when a block is finished enough to render as
markdown rather than as a growing string.

Only some protocols mark block boundaries. Rather than leave consumers to guess
per provider, `Client` synthesises them from a change of kind, and a driver
whose protocol does mark them refines that through `Delta.EndBlock` — Anthropic
sets it on `content_block_stop`, which is what lets two adjacent blocks of the
same kind be told apart at all. A block is always closed, including when the
turn fails partway, so a consumer's render state is never left half-open.

## Addendum: interactive sign-in

A few vendors authenticate a *person*, not a service. GitHub Copilot and a
ChatGPT subscription have no key to paste; they have a subscription and a
browser. Their catalog entries existed but said "exchange the token yourself",
which is not an SDK feature so much as an admission.

**The grants are provider-agnostic; the providers are data.** `auth/oauth`
implements RFC 8628 (device authorization) and RFC 7636 (PKCE over a loopback
redirect) knowing nothing about Copilot or OpenAI. What is provider-specific —
client identifiers, endpoints, and in Copilot's case a second exchange — is a
`flow` value in `auth`. This is the same shape as the rest of the SDK: a
protocol is code, a vendor is a row. Adding Claude Pro or Gemini's subscription
later is a `flow`, not a package.

**What persists is the durable credential, not the usable one.** Copilot's
sign-in yields a GitHub token that does not expire, which is exchanged for a
Copilot token that lasts about half an hour. Storing the latter would mean
signing in again every half hour. Storing the former means something has to
mint the short-lived one — and half an hour is well inside a single long
session, so it cannot happen once at start-up.

That is why the credential is presented through an `http.RoundTripper` rather
than as `Config.APIKey`. The transport holds a `tokenSource` that renews on
demand and overwrites whatever `Authorization` the driver set, which is how a
driver that insists on a placeholder key still sends a real one. It also keeps
`Config` free of network I/O: building a config for a signed-in vendor touches
nothing, and the first request pays for the first token.

**The endpoint is part of the credential.** Copilot reveals which host your
account talks to only after you authenticate, and an enterprise account's is
not an individual's. So `Credential` carries an `Endpoint`, recorded at
sign-in, and `Config` prefers it over the catalog's default. This is also why
Copilot's login performs one exchange immediately rather than deferring
everything: it is how the endpoint is discovered, and it is where an account
without a subscription is found out — at sign-in, rather than mid-turn later.

**A `Store` interface, not a file.** Where a secret belongs is the application's
decision. The bundled `FileStore` is what a CLI wants — 0600 in a 0700
directory, written and renamed so an interrupted save cannot leave a truncated
file where working credentials were. A server wants its own secret manager, and
a test wants neither. `auth.DefaultStore` selects one process-wide, because the
calls that need it (`Config`, `Provider`, `Available`) take no options.

This makes `auth` the first thing in the SDK that writes to disk. `pkg/llm`
still does not, and that boundary is the point: the core stays usable in a
multi-tenant server, and everything that reads an environment or a home
directory is opt-in.

## Addendum: an architecture pass

A review against Eskil Steenberg's maintainability criteria — replaceability,
cognitive load, risk isolation, team scaling — found the large things already
right and three smaller things wrong. Recording both halves, because "we
checked and it holds" is worth as much as a change.

**What held.** Vendor SDKs appear only in drivers; `pkg/llm` links two
non-standard-library packages. `Driver` is three methods, so any protocol can
be rewritten from its interface alone, and `RegisterAPI` means one can be
written outside this repository. The dependency graph is a DAG with every edge
pointing inward. None of that needed changing.

**Exported steps of an ordered algorithm became unexported.** `Classify`
documents that it applies its checks "in the order that keeps each from masking
the next", and then the package exported each step separately —
`ClassifyStatus`, `ClassifyTransport` — with no caller. `PrepareMessages` had
the same shape, exporting `SanitizeToolMessages`, `DropEmptyMessages` and
`SanitizeText` beneath it. A menu of entry points into an order-sensitive
algorithm is how two drivers end up classifying the same failure differently.
The orchestrators stay public; the steps are reachable from tests through
`export_test.go`, which is where a step that is only exercised through its
orchestrator ought to be pinned anyway.

**`RegisterCompat` takes a type instead of a decode function.** It used to take
`func([]byte) (any, error)`, which is enough to rebuild a compat value but not
enough to check one. A type is enough for both, and every caller wanted the
same decoding anyway. That closed the last hole in the `Compat any` escape
hatch: setting an `OpenAIChatCompat` on an Anthropic model left
`CompatOf[AnthropicCompat]` returning the zero value, so the model ran with
first-party defaults and nothing reported that its dialect had been ignored —
the same silent downgrade `UnmarshalJSON` already refuses. `Model.Validate` now
catches it before the request.

**Package jsonschema, a leaf.** Deriving a schema from a Go type and checking a
value against one are not language-model problems. Both halves moved to a
package that imports nothing outside the standard library — not even this SDK —
so they can be tested, replaced or reused on their own. `SchemaOf` and
`ToolFor` stayed behind, because they return `llm` types; they are now three
lines each. Direction is `llm` → `jsonschema`, one way.

**The Gemini driver dropped its SDK.** `google.golang.org/genai` also serves
Vertex AI, so it carries gRPC, protobuf, OpenTelemetry and Google's cloud
credential stack: 130 packages a program reaching Gemini with an API key never
uses. The driver now speaks the REST API directly and links two.

The risk in hand-rolling a wire format is that a misspelled field is not
rejected, it is ignored — reasoning or a schema would quietly stop being
applied. The public REST reference does not document every field this driver
sends (`parametersJsonSchema` and `responseJsonSchema` among them), so the
types in `wire.go` are transcribed from the SDK's own struct tags, which are
generated from the API definition. Transcription, not re-derivation. The
existing tests already asserted on the JSON going over the wire rather than on
SDK calls, so they carried over unchanged and served as the specification;
the paths the SDK had owned — listing, counting, error classification — gained
tests they never had.

One thing improved rather than merely moved: the SDK's error type exposed no
response headers, so a 429 from Gemini had no `Retry-After` to honour. The
driver keeps the response, so it does.

**What was proposed and rejected.** Moving `provider.go` into its own package
was recommended and then withdrawn. The count that motivated it was wrong —
103 package-level exports, not the 167 first measured, which had included
methods — and `net/http` is the same size. More to the point, `Provider.Open`
returns a `*Client` and `Provider.Config` returns a `Config`: it sits on the
core's principal types rather than beside them, so the move would have cost
four renames and some forty call sites to relocate six symbols. Cognitive load
is a reason to split a package when the parts are genuinely separate things,
not when the count looks large.

### Still open

- Five protocols behind pi: Bedrock Converse, Azure Responses, Codex Responses,
  Mistral Conversations, and pi's own `pi-messages`. Bedrock Converse is the
  most expensive: a distinct wire format and an AWS SDK dependency.
- Deferred responses — async or batch turns with a durable handle.
- Image generation, which is a different modality and probably a different
  package: it shares almost nothing with a language-model call beyond auth.
- Errors still terminate the stream rather than arriving as a response that
  carries the usage and partial content produced before the failure.
