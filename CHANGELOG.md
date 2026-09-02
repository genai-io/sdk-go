# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the API may change between minor releases.
Each such change is listed under **Changed** with what to write instead.

## [Unreleased]

An audit release. Nothing here is a new capability; it is the defects an audit
of every package turned up, and the tests that would have caught them. Thirteen
of the fixes below were reproduced before they were made — one of them a panic
that took the caller's process down.

The cause was uniform: `pkg/ai`, `pkg/ai/jsonschema`, `pkg/ai/catalog`,
`pkg/ai/provider`, `pkg/ai/auth`, `pkg/ai/auth/oauth` and all five drivers had
no test file of their own. Everything was verified through one black-box suite
that exercised the paths a request takes when it works. So the fixes come with
a unit-test layer per package, and statement coverage moves from 58% to 80%.

Eight things break. In the order you will meet them:

1. `ai.NewClient` is `ai.New`.
2. A hook returning an error ends the exchange, as the documentation always said.
3. `Run` no longer yields a turn's own failure on the iterator.
4. `auth.Deployment` returns a value and an error.
5. `catalog.NoReasoning` and `catalog.CopilotHeaders` are gone.
6. `anthropic.NewWithClient` takes the protocol it is speaking.
7. The MiniMax vendor is spelled `minimax`.
8. A schema derived from a type this package cannot describe now panics.

### Fixed

- **A consumer that stops reading mid-stream no longer panics the process.**
  On a driver error `Client.Stream` closed the open blocks and then yielded
  `EventDone` without asking whether the first yield had been refused — so a
  caller who broke out of the range on that closing block crashed with "range
  function continued iteration after function for loop body returned false".
  Every yield in the file now honours its answer. This is the one defect here
  that could not be worked around from outside.
- **A cut-off or failed OpenAI Responses turn is no longer silence.** The driver
  read only `response.completed` and switched on the status inside it, but the
  endpoint reports those outcomes as their own events — so a request that hit
  `max_output_tokens` arrived as an empty success with no usage, no
  `StopMaxTokens` and no error, and a server-side failure arrived as nothing at
  all. `response.incomplete`, `response.failed` and `response.refusal.delta` are
  each handled now.
- **`pkg/ai` no longer reads the environment.** It never did so itself, but both
  vendor SDKs prepend their own defaults, which read `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`,
  `OPENAI_BASE_URL`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID`. So
  `ai.New(ai.Config{Model: m})` with an empty key silently used a process-wide
  credential and host — which made the promise the package is built on, and the
  multi-tenant server it exists for, false. All three drivers now assemble their
  client from the struct form, and no environment variable is consulted.
- **Anthropic overloads are retried again.** `Retry` treats a call as unreplayable
  once output has reached the caller, but it counted any delta as output, and
  the Anthropic driver's first delta carries only a model ID and a usage figure.
  A 529 after `message_start` was therefore never retried. Only content counts
  now.
- **A model may omit an optional tool argument.** Every field goes into
  `required` because OpenAI's strict mode demands it, with optionality expressed
  as a null union instead — that is correct, and unchanged on the wire. But
  `RunTools` validated the model's arguments against that same schema, so a
  model on Anthropic, Gemini or Ollama that simply left `limit` out was told
  "missing required property" and made to try again. `Check` now reads an absent
  property as null when the schema admits null.
- **Thinking replays on Vertex.** Three validation rules named the Anthropic
  Messages protocol and not Anthropic on Vertex, so a signed thinking block was
  refused there — ending any conversation that continued past the model's first
  thinking turn — an unsigned one passed and was then dropped by the driver, and
  sampling parameters validated but were never sent.
- **Vertex honours the endpoint you gave it.** The Google credential option was
  applied after the caller's, and it sets a base URL and an HTTP client of its
  own, so `Config.BaseURL` was silently discarded. The order is reversed. A
  caller's `HTTPClient` is still dropped, because the credential *is* that
  client's transport and layering one over it would remove the authentication —
  that is now said out loud rather than claimed otherwise.
- **A Vertex driver says it is one.** It returned the Anthropic driver
  unchanged, so `Name()`, every `Error.Driver` and every listed model reported
  `anthropic-messages`.
- **Gemini's thinking tokens are counted.** The driver read only
  `candidatesTokenCount`; Gemini reports reasoning separately as
  `thoughtsTokenCount`, so every thinking turn under-reported its output and its
  cost. `Usage.Reasoning` is now filled here and by the Responses driver, and is
  a memo inside `Output` rather than a figure beside it, because `Pricing.Cost`
  prices `Output` alone.
- **Two parallel tool calls no longer collapse into one.** Gemini omits
  `functionCall.id`, and the Anthropic request builder minted the same
  compatibility ID for every call that lacked one, so a batch of two came out as
  a batch of one.
- **A session created empty can be reopened.** `SetMessages` announced a
  replacement unconditionally, so seeding a fresh agent with the empty history a
  new session hands back recorded a snapshot of nothing — which the fold then
  read as a corrupt file and refused. Both shipped examples did exactly that,
  and `examples/agent-session`'s promise that a second run continues the first
  was not true. A replacement is now announced only when something was actually
  replaced, and an empty conversation is a state a session can hold.
- **A session id cannot escape its store.** `jsonl` joined the id onto the root
  with no validation, so `Delete("")` removed the entire store and `"../x"`
  reached outside it — with ids that come from application input, the `-resume`
  flag in the example included. Empty, `.`, `..` and anything carrying a
  separator are refused by every method.
- **A session survives the process dying mid-write.** The reader gave up on a
  torn final line and reported no sequence number, and the writer appended
  straight onto those bytes with no newline between — so the next entry was
  swallowed into an unreadable line and numbering restarted at one. The tail is
  truncated when the file is opened, and the sequence is recovered from the last
  line that parses.
- **A corrupt line in the middle of a session is an error, not a shorter
  session.** Reading stopped silently at the first line that would not parse, so
  damage in the middle came back as a conversation with its end missing — the
  hole a fold cannot survive. Only an unparsable *final* line is still tolerated.
- **A hook that fails ends the exchange.** `PreTool` and `PostTool` turned an
  error into a tool error the model was shown and carried on, while every
  document said an error ends the exchange. It does now, with `StopError`, and
  the calls that will never run report as much. `Decision{Block: true}` remains
  the other answer: a refusal the model is told about and may work around. That
  is the whole difference between the two things a gate returns.
- **A cancelled batch stops between tools.** A sequential batch ran every
  remaining tool after `Interrupt`, because nothing checked the turn's context
  between them.
- **An interrupted call is still paid for.** The turn returned before adding the
  response's usage, so the tokens a cancelled stream had already spent vanished
  from `TurnEnd`.
- **A message added at the end of a turn arrives before the next turn's, not
  after it.** `AddMessages` queued past the last step boundary was drained one
  step into the following exchange, which put it after that exchange's own
  input — a conversation in an order nobody said things in.
- **`SetTools` takes effect where it says.** The toolset was read per call, so a
  change landed in the middle of a batch the model had already been offered; it
  is snapshotted when the batch begins.
- **Deleting a session no longer breaks the store.** A vanished `meta.json` was
  treated as fatal, so one session removed while another was appending made
  every later `Meta` and `List` on that store fail.
- **`bedrock-openai` can reason.** It declared a reasoning ladder with no
  thinking dialect, so the driver returned before ever reading the field — an
  inert row that looked configured. A catalog invariant now refuses that
  combination.
- **`catalog.Model("minimax/…")` resolves.** The vendor was keyed `minmax`, a
  misspelling of the brand every other field spells correctly. The row is
  `minimax`; the old key still resolves through an alias.
- **A live-listed model keeps what the catalog knows.** `provider.Provider`
  decorated an unlisted ID with its protocol and vendor only, dropping the
  window and reasoning ladder the vendor's own resolver would have inferred from
  the ID — so a model newer than the table came back stripped. `Config.Resolve`
  carries that resolver, and `Vendor.Provider` installs it.
- **Two clients no longer race to refresh one token.** Each call built its own
  token source, so two clients for the same interactive vendor renewed
  independently with the same rotating refresh token: the second renewal failed
  and the last write won. There is one source per vendor and store, the store is
  re-read immediately before renewing, and a failed save of a rotated token is
  now an error rather than a silently spent credential.
- **A tier no longer zeroes the rates it does not mention.** Pricing tiers
  replaced all four rates wholesale, so a tier that stated only an input rate
  priced everything else at nothing.
- **A 401 whose body mentions tokens is still a 401.** Message matching ran
  ahead of the status code, so any failure whose text contained a
  context-exceeded phrase was classified as one — and as not retryable. The
  message is consulted only when the status cannot answer.
- **A cancelled call is classified.** `context.Canceled` came back bare from the
  client and from `Retry`, so `IsKind(err, KindCanceled)` was false for the one
  cancel a caller causes on purpose, while the same cancel from a driver was
  classified.
- **A message carrying an orphaned tool result keeps its text.** `Repair`
  dropped the whole turn; it now removes the orphaned blocks and keeps the turn
  if anything wire-visible is left. Invalid UTF-8 in the system prompt, in a
  tool result's name and in a reasoning summary is now replaced too — previously
  only message content was.
- **`json:",omitempty"` names its field.** The deriver read the tag's empty name
  as the property name, so the field was called `""` and the model could never
  fill it. An empty struct also derived `"required": null`, which two drivers
  forwarded verbatim.
- **A field the deriver cannot describe is refused, not guessed.** A type with
  its own `MarshalJSON` was described by its Go fields, so `big.Int` became an
  empty object and `json.RawMessage` an array of integers — schemas that reject
  the JSON their own type writes. `MarshalText` is still derived, as a string,
  because that shape is knowable. An `enum` tag on a bool field is parsed as
  booleans rather than producing members it can never match.
- **Gemini can be told not to think.** The off rung sent no thinking
  configuration at all, so the model kept its default reasoning; an explicit
  zero budget is sent now.
- **Reasoning under the `reasoning` key is not dropped.** The Chat Completions
  driver read only `reasoning_content`, losing what OpenRouter and Ollama send,
  and read neither unless the model declared a ladder.
- **Tool calls survive a cut stream.** Chat Completions returned on the error
  before yielding the calls it had accumulated.
- **An Anthropic error carries its code.** The message was the SDK's rendering
  of the whole request line and raw body, and the code was always empty; both
  are read from the error payload now.
- **`disable_parallel_tool_use` is sent only when it is true**, rather than as
  `false` on every constrained tool choice — a field a third-party
  Anthropic-compatible host may not accept.
- **Redacted thinking survives a tool call.** The driver ignored it on the way in
  and had no way to send it back, though Anthropic requires it echoed.
- **Gemini lists the models it can generate with**, by asking whether
  `generateContent` is among a model's supported methods rather than by matching
  substrings in its name.
- **`Entries` honours its context** in both stores. It ignored cancellation.

### Changed

- **`ai.NewClient` is now `ai.New`.** Every constructor here is named for what it
  returns — that is the rule `doc.go` states and every driver package and
  `provider.New` already followed — and this one was the exception that made the
  rule read as a description of something else. `NewClient` remains for one
  release as a deprecated alias. Write `ai.New(cfg)`.
- **A hook's error ends the exchange.** See Fixed. Code that relied on `PreTool`
  or `PostTool` returning an error to tell the model something should return
  `Decision{Block: true, Reason: …}` instead, which is what that value is for.
- **`Run` reports a turn's failure once.** The iterator yielded `out.Err` after
  `TurnEnd` had already carried it, so every caller saw a failure twice — the
  shipped chat example printed a Ctrl-C as both "(canceled)" and "context
  canceled". The iterator's error is now reserved for what happens outside a
  turn, `ErrBusy` today. Read `TurnEnd.Err`.
- **`auth.Deployment` returns `(ai.ProtocolConfig, error)`**, and no longer knows
  what a Vertex deployment is. Which environment variables a vendor needs, and
  what to build from them, is a `Deployment` function on the catalog row —
  `auth` reads the table rather than special-casing one protocol.
- **`catalog.NoReasoning` and `catalog.CopilotHeaders` are gone.** Both were
  exported package-level values any caller could mutate for the whole process.
  The headers are read off the vendor row, which is returned by value.
- **`auth.Transport` is unexported.** It had no constructor and no exported
  field, so nothing outside the package could make or read one.
- **`anthropic.NewWithClient` takes the protocol it speaks** as a third
  argument, so a driver built on the Anthropic client can report a different
  `API` — which is what `anthropic/vertex` needs.
- **`session.NewRecorder` is unexported**, and `Entry.Payload` with it. The
  recorder it built had neither the turn offset nor the restored history, so on
  a resumed session it double-counted turns and rewrote the conversation it had
  just read. `session.Open` is the way in.
- **`ai.Content.IsEmpty`, `ai.Content.Images`, `ai.Client.ContextWindow` and
  `ai.IsOverflow` are gone.** None had a caller; `IsEmpty` also disagreed with
  the package's own notion of an empty message about whether thinking counts.
- **`openai/internal/errs` is `openai/internal/oai`**, and now holds what the two
  OpenAI drivers share — the client constructor, the error reader, the inline
  image framing — instead of three would-be packages. Failure classification
  moved up to `driver/internal/errs`, where all five drivers can reach it and
  nothing outside `driver/` can.
- **`MessageUpdate` carries its `Turn`.** It was the one event that did not,
  against what `event.go` said about all of them.

### Added

- **A unit-test layer in every package that lacked one** — the stream lifecycle
  and the consumer that breaks mid-stream, `Repair`, `Classify`, `Retry`, schema
  derivation and checking, the Gemini SSE parser, tool-call fragment
  accumulation, error classification across all five drivers, the credential
  store and the token refresh, the provider merge, and the catalog's invariants.
  Statement coverage across `pkg/` goes from 58% to 80%; `pkg/ai/auth/oauth` from
  nothing at all to 87%.
- **`TestCatalogInvariants` and a golden table.** Vendor rows are data, and
  nothing checked them — which is how an inert reasoning ladder and a misspelled
  brand survived. The invariants pin unique ids and display order, a protocol
  this SDK serves, a `Compat` whose type matches it, a parseable `Verified`
  date, at most one default rung, and a thinking dialect wherever a ladder is
  declared. `golden_test.go` holds the whole resolved table, so a change to one
  row that moves another shows up as a diff.
- **`catalog.Vendor.Resolve` and `provider.Config.Resolve`**, so a host's live
  listing keeps what the catalog knows about a model the table has never named.
- **`catalog.Vendor.Deployment` and `catalog.MissingDeploymentError`.**
- **`auth.Flow` and `auth.RegisterFlow`**, following `ai.RegisterAPI`'s shape.
  Interactive sign-in was a closed set of two hard-coded vendors; a consumer can
  add a third now.
- **`oauth.ExpiryMargin`**, the one expiry rule. There were three, disagreeing.
- **`golangci-lint` in CI**, with a configuration for this repository and a
  `make golangci-lint` target. The findings it reports on files this release did
  not touch are excluded by name with a note saying they are outstanding work,
  not accepted.
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
