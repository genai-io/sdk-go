# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the API may change between minor releases.
Each such change is listed under **Changed** with what to write instead.

## [Unreleased]

### Changed

- **The session package is renamed where a name was ambiguous.** `session.Tool`
  is `session.ToolRun` — a tool is a thing that can be run, and `ai` and `agent`
  both have that type, where this is the record of one having been. `session.Turn`
  is `session.Exchange`, because `Turn.Turn` is not a name. The entry constants
  follow: `EntryTool` is `EntryToolRun`, `EntryTurn` is `EntryExchange`, and
  `Entry.Tool`/`Entry.Turn` are `Entry.ToolRun`/`Entry.Exchange`.
- **`Recorder.Handle` takes a context.** `Store` is context-aware in every
  method and the recorder passed `context.Background()` to all of them, so a
  store that was not the local filesystem could block the loop delivering
  events with no way to cancel it.
- **`Recorder.Snapshot` is gone**, and nothing replaces it — see below.
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

- **`ai.Tool` is now a schema and a function.** `Name`, `Description` and
  `Parameters` are gone; the three of them were a `Schema` written out longhand,
  which is the type the package already had for exactly this — a named,
  described JSON shape. Write `ai.Tool{Schema: ai.Schema{Name: …, Description:
  …, Definition: …}, Run: …}`, or `ai.ToolSchema[T](name, description)` to
  derive it from a Go type. `ToolFunc` is unchanged. `Tool.ValidateArgs` and
  `Tool.ParameterSchema` moved onto the schema as `Schema.Validate` and
  `Schema.DefinitionMap`. A tool can now be `Strict`, which it could not be
  before.
- **`agent.Tool` declares `Schema() ai.Schema`** instead of `Definition()
  ai.Tool`, which returned a value whose `Run` field was always nil.

### Added

- **`agent.MessagesReplaced`**, the event `SetMessages` was missing. See below.
- **`ai.ToolCall.UnmarshalArgs`** decodes a call's arguments into a Go value —
  the function `ToolCall.Input` has always been documented as decoding with.
  An argument the schema does not have is an error rather than a silent drop.
- **`agent.WithRetry(attempts, backoff)`** — see below.
- **`agent.StopRefusal` and `agent.StopSequence`.**

### Fixed

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

- **`ai.RepairHistory` is now `ai.Repair`.** The old name introduced a word the
  package does not otherwise use: there is no history type here, only
  `[]Message` and `Request.Messages`. `Repair` is unchanged in behaviour — it
  still pairs tool calls with their results and replaces invalid UTF-8, and
  still removes only what a protocol would reject. Write `ai.Repair(msgs)`.

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
