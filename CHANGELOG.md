# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the API may change between minor releases.
Each such change is listed under **Changed** with what to write instead.

## [Unreleased]

An audit release: thirteen defects, each reproduced before it was fixed, and the
tests that would have caught them. They survived because `pkg/ai`,
`pkg/ai/jsonschema`, `pkg/ai/catalog`, `pkg/ai/provider`, `pkg/ai/auth`,
`pkg/ai/auth/oauth` and all five drivers had no test of their own — everything
was checked through one black-box suite that exercised the paths a request takes
when it works. Statement coverage across `pkg/` goes from 58% to 80%.

### Changed

Nine things break.

- **`ai.ToolResult.Content` is `ai.Content`, not a string**, so a tool that
  looked at something can answer with it. Write `Content: ai.TextContent(out)`
  where you wrote `Content: out`, and read it back with `result.Text()`. An
  image reaches the model on the Anthropic and OpenAI Responses protocols; on
  Chat Completions and Gemini, which carry only text there, `Model.Validate`
  refuses the request rather than dropping the picture on the way.
  `agent.ResultContent` is what the loop sends, and `agent.ResultText` stays for
  a log or a session record.
- **`ai.NewClient` is `ai.New`.** Every constructor here is named for what it
  returns, and this was the exception. The old name stays one release as a
  deprecated alias.
- **A hook's error ends the exchange**, with `StopError`. `PreTool` and
  `PostTool` used to turn it into a tool error and carry on, against what every
  document said. To tell the model something instead, return
  `Decision{Block: true, Reason: …}` — that is what the value is for.
- **`Run` reports a turn's failure once**, on `TurnEnd.Err`. It also yielded it
  on the iterator, so every caller saw it twice. The iterator's error is now
  only for what happens outside a turn, `ErrBusy` today.
- **`auth.Deployment` returns `(ai.ProtocolConfig, error)`** and no longer knows
  what a Vertex deployment is. The catalog row does, through a new
  `Vendor.Deployment`.
- **`catalog.NoReasoning` and `catalog.CopilotHeaders` are gone.** Both were
  exported values any caller could mutate for the whole process. Read the
  headers off the vendor row, which is returned by value.
- **`anthropic.NewWithClient` takes the protocol it speaks**, so a driver built
  on the Anthropic client can report a different `API`. `anthropic/vertex` now
  reports `anthropic-vertex` rather than the protocol underneath it.
- **The MiniMax vendor is `minimax`.** It was keyed `minmax`, a misspelling of
  the brand every other field spells correctly; the old key still resolves
  through an alias.
- **A schema derived from a type carrying its own `MarshalJSON` panics.** Such a
  type was described by its Go fields, so `big.Int` became an empty object and
  `json.RawMessage` an array of integers — schemas that reject the JSON their
  own type writes. `MarshalText` still derives, as a string.

Removed with no caller anywhere: `ai.Content.IsEmpty`, `ai.Content.Images`,
`ai.Client.ContextWindow`, `ai.IsOverflow`. `auth.Transport` and
`session.NewRecorder` are unexported — the recorder built by the latter
double-counted turns on a resumed session, and `session.Open` is the way in.
`openai/internal/errs` is `openai/internal/oai` and holds what the two OpenAI
drivers share; failure classification moved to `driver/internal/errs`, which all
five drivers reach and nothing outside `driver/` can. `MessageUpdate` carries
its `Turn`, which it alone did not.

### Fixed

Three that a caller could not work around:

- **A consumer that stopped reading mid-stream panicked the process.**
  `Client.Stream` yielded `EventDone` after a yield had already been refused.
- **`pkg/ai` read the environment.** Not itself, but both vendor SDKs prepend
  defaults that read `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and five more, so
  `ai.New` with an empty key silently used a process-wide credential and host —
  which made the promise this package is built on false. The drivers now build
  their client from the struct form and consult no variable.
- **A session created empty could not be reopened.** `SetMessages` announced a
  replacement of nothing, and the fold read the resulting empty snapshot as a
  corrupt file. Both shipped examples did exactly that.

`pkg/ai`:

- `Retry` counted a metadata-only first delta as output, so an Anthropic 529
  after `message_start` was never replayed.
- `RunTools` refused a tool call that left an optional argument out, on every
  provider whose models do not answer in OpenAI's strict shape.
- Three validation rules named Anthropic Messages and not Anthropic on Vertex,
  so thinking could not be replayed there.
- A pricing tier replaced all four rates, zeroing any it did not state.
- Message matching ran ahead of the status code, so a 401 mentioning tokens was
  classified as a context overflow, and as not retryable.
- `context.Canceled` came back unclassified from the client and from `Retry`.
- `Repair` dropped a whole turn carrying an orphaned tool result, text and all,
  and never sanitised the system prompt.
- `json:",omitempty"` derived a property named `""`; an empty struct derived
  `"required": null`; an `enum` on a bool derived string members.

Drivers:

- A truncated or failed OpenAI Responses turn arrived as an empty success:
  `response.incomplete` and `response.failed` are events of their own, not a
  status inside `response.completed`. Refusals were invisible too.
- Gemini's thinking tokens went uncounted — reported separately as
  `thoughtsTokenCount` — so every thinking turn under-reported its cost.
- Vertex discarded the caller's `BaseURL`, because the credential option was
  applied after it.
- Two parallel Gemini tool calls collapsed into one, both taking the same minted
  ID because Gemini omits `functionCall.id`.
- Chat Completions dropped reasoning sent under `reasoning`, and lost the tool
  calls it had accumulated when a stream was cut.
- An Anthropic error carried no code and a message built from the request line
  and raw body; `disable_parallel_tool_use: false` was sent on every constrained
  tool choice; redacted thinking could not be replayed.
- Gemini's off rung sent no thinking configuration, so the model kept reasoning,
  and `Models()` filtered by substring rather than by supported method.

`pkg/agent`:

- A session id was joined onto the store root unvalidated, so `Delete("")`
  removed the whole store and `"../x"` reached outside it.
- A torn final line swallowed the entry appended after it and restarted the
  sequence at one; a corrupt line in the middle truncated the read silently.
- A sequential batch ran its remaining tools after `Interrupt`.
- An interrupted call's tokens vanished from `TurnEnd`.
- `AddMessages` queued past the last step boundary entered behind the next
  exchange's own input.
- `SetTools` landed mid-batch rather than at the next inference.
- Deleting a session while another was appending broke `Meta` and `List` for
  every session in that store.
- `Entries` ignored its context, in both stores.

Catalog, provider and auth:

- `bedrock-openai` declared a reasoning ladder with no thinking dialect, so the
  driver never read the field.
- `provider.Provider` stripped an unlisted model of the window and ladder its
  vendor could infer from the ID.
- Two clients for one interactive vendor refreshed the same rotating token
  independently, and a failed save was ignored.

### Added

- **`ai.WithHeaders`**, the headers one call carries over the ones its Config
  and model already send. A header that depends on the turn rather than the
  endpoint — a tenant or trace tag, an opt-in a provider meters differently —
  could only be changed by building a second client, and a second client is a
  second connection pool. All four protocols apply them, `CountTokens`
  included.
- **`agent.Inference.Client` and `Agent.SetClient`/`Agent.Client`**, so an agent
  is no longer welded to the model it was built on. `SetClient` is a person
  switching model mid-session: the conversation, the prompt and the tools are
  what they were, and only where the next call goes is different.
  `Inference.Client` routes one call — a cheaper model for a step that only
  summarises, a fallback endpoint on a retry, since every attempt asks the
  hooks again. A turn may therefore hold calls to more than one model, which is
  what `TurnEnd.Usage` now says: it sums tokens, and a cost is folded from each
  `MessageEnd` against the model that answered it.
- **A unit-test layer in every package that lacked one**: the stream lifecycle
  and the consumer that breaks mid-stream, `Repair`, `Classify`, `Retry`, schema
  derivation and checking, the Gemini SSE parser, tool-call accumulation, error
  classification across all five drivers, the credential store and token
  refresh, the provider merge. `pkg/ai/auth/oauth` goes from nothing to 87%.
- **`TestCatalogInvariants` and a golden table.** Vendor rows are data and
  nothing checked them, which is how an inert reasoning ladder and a misspelled
  brand survived. A row that changes another row's meaning now shows as a diff.
- **`catalog.Vendor.Resolve` and `provider.Config.Resolve`**, so a host's live
  listing keeps what the catalog knows about a model it has never named.
- **`catalog.Vendor.Deployment` and `catalog.MissingDeploymentError`.**
- **`auth.Flow` and `auth.RegisterFlow`**, following `ai.RegisterAPI`'s shape.
  Interactive sign-in was a closed set of two hard-coded vendors.
- **`oauth.ExpiryMargin`**, the one expiry rule. There were three, disagreeing.
- **`golangci-lint` in CI**, with a `make golangci-lint` target that verifies the
  configuration before running it. There is no exclusion list: an error this
  repository means to drop is written `_ =` with the reason beside it.
- **Dependabot watches `gomod`.** Only GitHub Actions were watched, and the two
  vendor SDKs had drifted forty minor releases behind — which is how the
  Responses driver came to be reading events the endpoint had stopped sending
  that way.

## [0.2.0] - 2026-08-31

This release is `pkg/agent`: the loop around a model call, its events, its
hooks, its tools and its sessions. `pkg/ai` changes with it where the two meet,
and those changes are listed first because everything else is built on them.

Five things break. In the order you will meet them:

1. `ai.Tool` is a `Schema` and a `Run`, not four fields.
2. `agent.Tool` declares `Schema() ai.Schema`, not `Definition() ai.Tool`.
3. `PreInfer` is handed an `*agent.Inference`, not an `*ai.Request`.
4. `session` renames what was ambiguous, and `Entry` carries the turn.
5. `Recorder.Handle` takes a `context.Context`.


### Changed

- **`ai.Tool` is now a schema and a function.** `Name`, `Description` and
  `Parameters` are gone; the three of them were a `Schema` written out longhand,
  which is the type the package already had for exactly this — a named,
  described JSON shape. Write `ai.Tool{Schema: ai.Schema{Name: …, Description:
  …, Definition: …}, Run: …}`, or `ai.ToolSchema[T](name, description)` to
  derive it from a Go type. `ToolFunc` is unchanged. `Tool.ValidateArgs` and
  `Tool.ParameterSchema` moved onto the schema as `Schema.Validate` and
  `Schema.DefinitionMap`. A tool can now be `Strict`, which it could not be
  before.
- **`ai.Schema.DefinitionMap` and `WireName` take a value receiver**, matching
  `Validate`. They could not be called on a `Schema` returned by value, which
  is what `agent.Tool.Schema()` returns. Calls are unchanged, except where one
  relied on the nil receiver: a possibly-nil `*Schema`, such as
  `Request.Schema`, needs the nil check written out.
- **`ai.RepairHistory` is now `ai.Repair`.** The old name introduced a word the
  package does not otherwise use: there is no history type here, only
  `[]Message` and `Request.Messages`. `Repair` is unchanged in behaviour — it
  still pairs tool calls with their results and replaces invalid UTF-8, and
  still removes only what a protocol would reject. Write `ai.Repair(msgs)`.
- **`agent.Tool` declares `Schema() ai.Schema`** instead of `Definition()
  ai.Tool`, which returned a value whose `Run` field was always nil.

- **`PreInfer` is handed an `agent.Inference`, not an `*ai.Request`.** It has
  the three things an agent contributes — `System`, `Messages`, `Tools` — and
  an `Options []ai.Option` layered on last for everything else, so
  `ai.WithForceTool`, `ai.WithSchema`, `ai.WithMaxTokens` and the rest now
  reach the wire. They did not before: the agent projected two fields of the
  request it handed over and dropped the other nine in silence. The shape
  changed rather than the projection widening because a half-filled request
  cannot distinguish a field left alone from one set to zero, which is the
  ambiguity `ai.Request.Temperature` is a pointer to avoid. `MessageStart` and
  `MessageEnd` carry `Inference` in place of `Request`.
- **An agent's toolset is now authoritative.** `System` and `Tools` go out on
  every call, so an agent with no tools offers none even on a client built with
  some. Previously an empty toolset was projected as no instruction at all, and
  there was no way for an agent to say "none".
- **Agent events carry what a consumer used to rebuild.** Every event in a turn
  has its `Turn`; `MessageEnd` carries the `Request` and `Attempt` its
  `MessageStart` opened with, and `ToolEnd` the `Name` and `Args` of its
  `ToolStart`. `ToolUpdate` gained `Name`. The `Recorder` was the first
  consumer and needed a mutex, two maps and a counter to pair spans back up;
  none of that is left.

- **`WithID`, `WithName` and `Agent.Name` are gone.** `WithID` documented
  itself as identifying the agent in the events it emits, which no event ever
  carried; between them the two fields fed nothing but `String`. `String`
  remains, naming the agent by its model. Nothing replaces them: an agent's
  identity is the caller's, and it already holds the agent.
- **`SetTools` and `AddHooks` are variadic**, matching `WithTools` and
  `WithHooks`; `AddHook` is `AddHooks`. Messages keep their existing split — a
  conversation is handed over as a slice (`WithMessages`, `SetMessages`) and
  added to as items (`AddMessages`).
- **`agent.Told` is `agent.ResultText`**, a name that says what it returns.
- **The session package is renamed where a name was ambiguous.** `session.Tool`
  is `session.ToolRun` — a tool is a thing that can be run, and `ai` and `agent`
  both have that type, where this is the record of one having been.
  `session.Turn` is `session.Outcome`, holding how a turn ended rather than
  restating which turn it was; `EntryTool` is `EntryToolRun` and `EntryTurn` is
  `EntryOutcome`.
- **`Entry` carries the turn, and the payloads no longer do.** Which exchange
  an entry belongs to says where it sits, which is what `Seq` and `At` say too,
  so it is `Entry.Turn` rather than a field repeated inside three payload
  types. Message and snapshot entries gain it, having previously had no turn at
  all. Read `entry.Turn` where you read `entry.Inference.Turn`,
  `entry.ToolRun.Turn` or `entry.Turn.Turn`.
- **`Recorder.Handle` takes a context.** `Store` is context-aware in every
  method and the recorder passed `context.Background()` to all of them, so a
  store that was not the local filesystem could block the loop delivering
  events with no way to cancel it.
- **`Recorder.Snapshot` is gone**, and nothing replaces it. It existed so a
  caller could tell a session that compaction had replaced the conversation;
  the agent announces that itself now, as `MessagesReplaced`. Delete the call.

### Added

- **`session/memory`**, a second `session.Store` — sessions that live as long as
  the process, and the implementation that proves the interface describes more
  than the filesystem. The session package's own tests record into it now, so
  they exercise the contract rather than a directory; `TestStoreContract` runs
  the same assertions against both stores.
- **`MessageUpdate.Text` and `.Thinking`** give the fragment an update carries,
  empty when it carries something else. Every example in this repo held the
  same three-level reach through `Delta.Type` and `Delta.Block.Type` to print a
  token; they print `v.Text()` now.
- **`agent.WithContinuation(attempts, prompt)`** takes another step in the same
  exchange when the output cap cut an answer off, instead of ending the turn
  with half of one. Off by default: the loop knows when it happened, but paying
  for more tokens and choosing the words are the application's.
- **`Agent.Interrupt` returns a channel** that closes once the exchange it
  ended is actually over and the agent is idle again. The goroutine that read
  the keystroke is not the one ranging over `Run`, so it could not see the
  range end and had no way to know when the agent stopped touching the
  conversation. Calling it as a statement is unchanged.
- **`agent.MessagesReplaced`**, the event `SetMessages` was missing: the
  conversation being thrown away and replaced, announced at the start of the
  next exchange so a session's fold knows to start over.
- **`ai.ToolCall.UnmarshalArgs`** decodes a call's arguments into a Go value —
  the function `ToolCall.Input` has always been documented as decoding with.
  An argument the schema does not have is an error rather than a silent drop.
- **`agent.WithRetry(attempts, backoff)`** replaces `WithMaxAttempts`, taking
  `ai.Retry`'s own arguments. It is off by default, because retry belongs on
  the client and two budgets multiply rather than add.
- **`agent.StopRefusal` and `agent.StopSequence`.**

### Fixed

- **A panicking tool no longer takes the process down.** Tools run on
  goroutines this package creates, which is the one place a panic cannot be
  recovered by whoever wrote the code — so one nil-map write in one tool killed
  the caller's program mid-conversation, with no way to prevent it. A panic is
  now the failure a tool already has a way to express: the model is told, the
  batch finishes, and the turn carries on. `agent.PanicError` carries the stack
  for whoever is watching `ToolEnd`, while its `Error` stays one line, because
  that line is what the model reads.
- **A compacted conversation survives a restore without the caller doing
  anything.** The session is the fold of what the agent announced, and
  `SetMessages` announced nothing — so a session restored the history that
  compaction had just thrown away, growing the context the caller was trying to
  shrink. It is now announced as `MessagesReplaced` at the start of the next
  exchange, which is when the agent next has anywhere to report it, and the
  recorder folds it as the point a fold starts from. `Recorder.Snapshot` was
  the workaround and is removed: a step a caller must remember is a step a
  caller forgets. Messages passed to `WithMessages` enter the fold the same
  way, which they previously never did.
- **Recording stops at the first failed write.** It used to carry on, and a
  fold with a hole in it is not a shorter conversation but a broken one — lose
  the message carrying a tool call and the results answering it are orphaned,
  which no provider accepts and `ai.Repair` silently deletes. A prefix still
  folds; a log with a gap does not.
- **Turns are numbered from the session's beginning.** The agent counts from
  one on every run, deliberately, and the session stored that number as its
  own — so a resumed session held two exchanges both called turn 1.
- **Resuming no longer records a copy of the conversation it just read.**
- **A malformed entry is an error, not a silent skip.** `Entry` is a tagged
  union that nothing validated, so one whose type and payload disagreed was
  dropped while folding, taking a message out of the conversation with it.
- **An agent no longer retries by default, and honours `Retry-After` when it
  does.** Retry belongs on the client, where `ai.Retry` implements it; a second
  budget on the agent multiplied rather than added, so the setup both READMEs
  teach — three attempts on a client wrapped in `ai.Retry(3, …)` — was nine
  model calls for one step. The agent's own loop also never waited at all,
  replaying a rate limit that named a delay within microseconds of receiving it.
  `WithRetry` replaces `WithMaxAttempts` and turns a second budget on
  deliberately, for what `ai.Retry` structurally cannot replay: a stream that
  already yielded output, and a stalled one.
- **A refused or filtered turn is no longer reported as `end_turn`.** The loop
  translated only `max_tokens`, so `ai.StopRefusal` and `ai.StopSequence` both
  read as a model that answered normally.
- **A stream that ends without a response is retryable again.** It was raised
  as a bare `error`, which `ai.IsRetryable` — asked five lines later — reads as
  permanent; `ai.Collect` calls the same failure retryable.
- **A finished turn no longer pins the caller's context.** The agent held the
  last turn's `CancelFunc`, and a `CancelFunc` closes over its context, so
  everything hanging off it stayed alive until the next turn replaced it.
- **`agent.ToolFunc` no longer silently drops arguments the model invented.**
  It decoded with a bare `json.Unmarshal` where `ai.ToolFunc` refuses unknown
  fields, so a tool could run on a request it never received.

### Performance

- **The jsonl session store appends ~25× faster** (204 µs → 8 µs, 42 → 4
  allocations). It rewrote `meta.json` atomically on every recorded event to
  maintain `Entries` and `UpdatedAt`, which are a cache of what the entries
  file already knows. Metadata is now written when it is read, and each
  session's sequence number is recovered from the entries file itself — which
  also fixes a second process carrying on from a stale count after a crash and
  writing duplicate sequence numbers. The store-wide lock is now per session.

## [0.1.2] - 2026-08-22

### Fixed

- **MiniMax, MiMo and OpenAI now size the models their catalogs do not list.**
  All three serve generations the table does not carry a row for — MiniMax's M2
  line, MiMo's v2 line, OpenAI's o-series and the GPT-4 generations — and none
  of the three publishes a window through its API. Those models resolved with
  no window at all, which is what "cannot size this conversation" means to a
  caller: every context measurement built on it goes quiet. Each vendor now
  reads the generation out of the model ID, as Moonshot and BigModel already
  did. A generation none of them names still reports zero rather than borrowing
  a neighbour's figure.

## [0.1.1] - 2026-08-22

### Fixed

- **DeepSeek and Alibaba (Model Studio) now replay their own reasoning.** Both
  endpoints accept `reasoning_content` on an assistant message, but neither
  catalog entry said so, so `Client` refused to send a thinking block back to
  them. That ended any conversation the moment it continued past the model's
  first thinking turn — the turn could be read but never replayed. Both entries
  now set `OpenAIChatCompat.ReasoningContent`, verified against both endpoints.

## [0.1.0] - 2026-08-22

First release.

### Added

- **One typed API over five wire protocols** — Anthropic Messages, OpenAI Chat
  Completions, OpenAI Responses, Google Gemini and Anthropic on Vertex AI. The
  same `Message`, `Response` and streaming events whichever serves the request.
- **A model catalog**: 27 vendors and 55 models, with endpoints, limits,
  pricing, reasoning ladders and per-endpoint quirks as data readable without a
  network call. A vendor is a row, not a package.
- **Streaming** as an iterator of events, where text, thinking, tool calls and
  images share one start/delta/end lifecycle.
- **Tool calling**: `ai.ToolFunc` builds a tool from a name, a description and
  a function; the schema is derived from the argument struct, arguments are
  checked against it before the function runs, and `Client.Run` holds the
  conversation to the end.
- **Structured outputs**: `ai.CompleteAs[T]` derives the schema, constrains
  generation to it and decodes the answer, so the type is named once.
- **`pkg/ai/jsonschema`**, a JSON Schema derivation and validation package with
  no module dependencies, targeting what providers accept rather than what the
  specification permits.
- **Typed errors** — auth, rate limit, context exceeded, unsupported — and a
  failed turn that still returns what arrived and what it cost.
- **Middleware** as a `Handler` wrapping a `Handler`, with `ai.Retry` shipped
  and caching, logging and cost metering left to the caller.
- **Credential separation**: `pkg/ai` reads no environment variable and no
  file; `pkg/ai/auth` is the opt-in that does, including the browser sign-in
  for vendors that authenticate a person rather than a service.

[0.1.2]: https://github.com/genai-io/sdk-go/releases/tag/v0.1.2
[0.1.1]: https://github.com/genai-io/sdk-go/releases/tag/v0.1.1
[0.1.0]: https://github.com/genai-io/sdk-go/releases/tag/v0.1.0
