# sdk-go

[![Go Reference](https://pkg.go.dev/badge/github.com/genai-io/sdk-go.svg)](https://pkg.go.dev/github.com/genai-io/sdk-go)
[![CI](https://github.com/genai-io/sdk-go/actions/workflows/ci.yml/badge.svg)](https://github.com/genai-io/sdk-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/github/go-mod/go-version/genai-io/sdk-go)](go.mod)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

A Go SDK for large language models, in two packages: one typed API over the
Anthropic Messages, OpenAI Chat Completions, OpenAI Responses and Google Gemini
protocols, and an agent runtime that runs the loop around it.

**`pkg/ai` — one model call**

- **One API, five protocols** — the same types whichever provider serves the request.
- **Streaming** — text, thinking, tool calls and images on one start/delta/end lifecycle.
- **Tool calling** — schema derived from your argument struct, arguments checked before your code runs.
- **Structured outputs** — `CompleteAs[T]` derives the schema, constrains generation and decodes.
- **Typed errors** — auth, rate limit, context exceeded and unsupported, not substrings to match.
- **A model catalog** — 27 vendors, 55 models; endpoints, limits and pricing as data.
- **No ambient credentials** — `pkg/ai` reads no environment variable and no file.

**`pkg/agent` — the loop around it**

- **Reason and act** — call the model, run the tools it asks for, call it again.
- **Everything as events** — eleven types on one channel, and the conversation is the fold of one of them.
- **Four hooks** — refuse a tool call, rewrite what is sent, redact what came back.
- **Parallel tools** — a batch runs concurrently unless a tool says it cannot.
- **Sessions** — record what an agent did, restore the conversation from it.

[Installation](#installation) ·
[Quickstart](#quickstart) ·
[Streaming](#streaming) ·
[Tool use](#tool-use) ·
[Agents](#agents) ·
[Structured outputs](#structured-outputs) ·
[Request options](#request-options) ·
[Messages](#messages-and-content) ·
[Errors](#errors-and-execution-policy) ·
[Credentials](#credentials) ·
[Protocols](#supported-protocols) ·
[中文文档](README.zh-CN.md)

## Installation

```sh
go get github.com/genai-io/sdk-go
```

Requires Go 1.24 or later.

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)

func main() {
	client, err := auth.Client("openai/gpt-4.1")
	if err != nil {
		log.Fatal(err)
	}

	response, err := client.Complete(context.Background(),
		[]ai.Message{ai.UserMessage("Explain goroutine leaks.")},
		ai.WithSystem("You are concise."))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response.Text())
}
```

`auth.Client` reads the credential from the environment variable the vendor
documents — `OPENAI_API_KEY` here. The blank import registers one wire
protocol; import only the ones you use, or `pkg/ai/driver/all` when the model
is chosen at run time.

This is the short end of one chain: `auth.Client` is `auth.Config` plus
`ai.New`, which is `ai.NewDriver` plus `ai.NewClientWithDriver`. Stop earlier when
you need to — a server that must not read ambient credentials stops at
`ai.New`, middleware stops at `ai.NewDriver`. See
[Constructing a client](docs/clients.md).

### Switching providers

Change the reference. Nothing else changes.

```go
ref := "anthropic/claude-opus-5" // or google/gemini-3.5-flash,
                                 //    deepseek/deepseek-v4-pro, ollama/llama4
client, err := auth.Client(ref)
```

`catalog.Models()` lists every `vendor/model` reference, without a network call.

Runnable examples are in [`examples/`](examples) — one per vendor, plus
[`tools/`](examples/tools) and [`structured/`](examples/structured).

## Streaming

`Stream` returns an iterator of events. Every kind of content — text, thinking,
tool calls, images — uses the same start/delta/end lifecycle, so one loop
handles all of them.

```go
for event, err := range client.Stream(ctx, messages) {
	if err != nil {
		return err
	}
	switch event.Type {
	case ai.EventBlockStart:
		ui.Open(event.Index, event.Block.Type)

	case ai.EventBlockDelta:
		switch event.Block.Type {
		case ai.BlockText:
			ui.AppendText(event.Index, event.Block.Text)
		case ai.BlockThinking:
			ui.AppendThinking(event.Index, event.Block.Text)
		}

	case ai.EventBlockEnd:
		if event.Block.Type == ai.BlockToolCall {
			go runTool(*event.Block.ToolCall)
		}

	case ai.EventDone:
		response = event.Response
	}
}
```

`EventBlockDelta` carries a fragment, `EventBlockEnd` the complete block, and
`EventDone` the aggregated `Response` — the same value `Complete` returns.
Every block that opens is closed, including on failure. Abandoning the iterator
cancels the request.

## Tool use

A tool is a name, a description, and a function. The struct is exactly what the
model may send.

```go
type SearchArgs struct {
	Query string `json:"query" description:"what to look for, in plain words"`
	Limit int    `json:"limit,omitempty" description:"how many passages to return" maximum:"10"`
}

search := ai.ToolFunc("search", "Search the documentation and return matching passages.",
	func(ctx context.Context, a SearchArgs) (string, error) {
		return docs.Search(ctx, a.Query, a.Limit) // dependencies are closed over
	})

response, history, err := client.Run(ctx,
	[]ai.Message{ai.UserMessage(question)}, []ai.Tool{search, fetch})
```

<details>
<summary><code>ToolFunc</code> is shorthand. This is what it folds down to.</summary>

```go
search := ai.Tool{
	Name:        "search",
	Description: "Search the documentation and return matching passages.",
	Parameters:  jsonschema.For[SearchArgs](),
	Run: func(ctx context.Context, arguments string) (string, error) {
		var a SearchArgs
		if raw := bytes.TrimSpace([]byte(arguments)); len(raw) > 0 {
			decoder := json.NewDecoder(bytes.NewReader(raw))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&a); err != nil {
				return "", fmt.Errorf("arguments for search: %w", err)
			}
		}
		return docs.Search(ctx, a.Query, a.Limit)
	},
}
```

`ai.Tool` is the whole of what a tool is: the three fields that go on the wire,
and one function that answers a call. `ToolFunc` derives the schema from
`SearchArgs` and decodes into it, and that is all it does — the two forms
produce byte-identical definitions and behave identically, errors included.

So the escape hatches are not features. A hand-written schema is an assignment,
`search.Parameters = handWritten`. A tool whose shape is not known until run
time is this form with `Parameters` from somewhere else. Neither needs anything
the common case does not already use.

The empty-argument branch is not decoration: every protocol here sends an empty
object for a call to a tool that takes none, and `json.Decoder` returns `EOF`
rather than a zero value for it.

</details>

`Run` is the loop the model needs: it asks, you answer, it continues.
`history` is the whole conversation, so a follow-up continues from it.

`SearchArgs` is named once, so the schema sent to the model and the struct its
arguments decode into cannot disagree:

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "what to look for, in plain words"},
    "limit": {"type": ["integer", "null"], "description": "how many passages to return", "maximum": 10}
  },
  "required": ["query", "limit"],
  "additionalProperties": false
}
```

Arguments are checked against it before your function runs, so a model's
mistake comes back as something it can correct. An unknown tool name, bad
arguments and a failing tool all reach the model as an `IsError` result rather
than ending the conversation.

Constraining the choice is one value with four states, the same four every
protocol expresses:

```go
ai.WithToolChoice(ai.ToolChoiceNone)            // no tool this turn
ai.WithToolChoice(ai.ToolChoiceRequired)        // some tool, the model's choice which
ai.WithToolChoice(ai.ToolChoiceNamed("search")) // this one
```

Write the turn loop yourself with `Complete` and `RunTools` when the turns are
your business — to stream as it arrives, to stop on a condition, to bill each
one.

## Agents

`pkg/ai` makes one model call. `pkg/agent` runs the loop around it: call the
model, run the tools it asks for, call it again, until the model answers
without asking for anything more.

`Run` advances the conversation one exchange and reports what it does as it
goes. The last event is `TurnEnd`, which carries how it went and the message
the model produced.

```go
a, err := agent.New(client,
    agent.WithSystem("You are a careful assistant."),
    agent.WithTools(readFile, listDir),
)
if err != nil {
    log.Fatal(err)
}

for e, err := range a.Run(ctx, ai.UserMessage("what does main.go do?")) {
    render(e)
}
```

Repeating it is a `for` loop, and the loop is yours — how messages are batched
into exchanges, what a failure means, and when to stop:

```go
for batch := range myMessages {
    for e, err := range a.Run(ctx, batch...) { render(e) }
}
```

Four ideas carry the design:

| | |
| --- | --- |
| **Everything is an event** | Nine types on one sequence. A message and a tool call each start, stream and end; the turn brackets them. The set is closed, so a consumer knows the list is all there is. |
| **The conversation is a fold** | Replay `MessageAdded` in order and you have exactly what the agent holds. That is all a session stores, and all a restore reads. |
| **Hooks are asked; events are told** | `PreInfer` and `PostInfer` sit either side of the model call, `PreTool` and `PostTool` either side of a tool. A permission system is a `PreTool` returning `Decision{Block: true}`. |
| **A tool answers two audiences** | `Content` goes to the model, `Details` to your interface — so formatting for a person is not paid for on every turn thereafter. |

A CLI reads stdin, an interface reads keys, a server reads requests, and none
of those is a shape a library should guess — which is why repeating an exchange
is your loop, not a method here. `AddMessages` puts something into the exchange
in flight; `Interrupt`, or simply breaking out of the range, ends it.

A batch of tool calls runs concurrently unless a tool declares it cannot. A
retryable stream failure is retried, and a stream that goes silent is bounded
and retried — each ending a turn with a stop reason that says which.

Sessions consume that stream rather than living inside the agent:
`rec.Handle(e)` in your own loop records what happened, and `session.Open`
folds it back into a conversation to resume from.

See [the Agent SDK](docs/agent.md) for the event contract, the hook composition
rules, tools and sessions.

## Structured outputs

```go
type Person struct {
	Name string `json:"name" description:"full legal name, family name last"`
	Age  int    `json:"age" description:"age in whole years" minimum:"0"`
}

person, err := ai.CompleteAs[Person](ctx, client, messages)
```

The type is named once: `CompleteAs` derives the schema from it, constrains
generation natively, and decodes the answer into it.

A tag key is the JSON Schema keyword it sets — eleven of them — and every word
is prompt text. Schemas are derived to be *accepted*, not merely valid, which
is a stricter target than the specification. See
[`pkg/ai/jsonschema`](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/ai/jsonschema).

## Request options

The conversation is an ordinary `[]ai.Message`. Everything else is an `Option`,
and the same option is a default at construction and an override at the call:

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh)) // default

response, err := client.Complete(ctx, messages,
	ai.WithTemperature(0),
	ai.WithMaxTokens(4096),
	ai.WithEffort(ai.EffortLow), // overrides the default
	ai.WithCacheRetention(ai.CacheLong))
```

Passing an option is what marks a setting explicit, so `WithTemperature(0)` is
deterministic sampling and omitting it inherits. Resolution runs model
defaults, then client defaults, then call overrides.

`WithEffort` is a normalized rung. Each model carries its own ladder onto
whatever its endpoint wants — a token budget, a level string, an enable flag —
and snaps to the nearest rung it offers.

## Messages and content

A message carries an ordered sequence of typed blocks rather than parallel
fields, because that order is what the next request has to replay.

```go
messages := []ai.Message{
	ai.UserMessage("What is in this image?", image),
	ai.AssistantMessage("A gopher."),
	ai.ToolResultsMessage(result),
}

text := response.Text()
history = append(history, response.Message()) // keeps every block, in order
```

Append `response.Message()`, not `ai.AssistantMessage(response.Text())` — the
first carries thinking and reasoning state forward, which is what lets a
reasoning model resume instead of starting over.

## Errors and execution policy

Failures are classified, so the answer to "what now" is in the type rather than
in the message text:

```go
switch {
case ai.IsAuth(err):            // the credential is wrong or missing
case ai.IsContextExceeded(err): // compact the conversation and retry
case ai.IsRetryable(err):       time.Sleep(ai.RetryAfter(err))
case ai.IsUnsupported(err):     // this model cannot do what was asked
}
```

A failed turn returns a non-nil error *and* a non-nil `*Response` with what
arrived, so a partial answer and its cost are not lost. Requests are validated
before they leave, and never rewritten to make them work: a model with no
system role is an error naming the model, because moving those instructions
into a user turn is a decision about your product.

Execution policy is `Middleware` decorating a driver:

```go
client := ai.NewClientWithDriver(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), model)
```

`Retry` is the only policy shipped, and every driver disables its vendor SDK's
own — so without it you get none. Caching, logging and metering stay yours.

## Credentials

`pkg/ai` never reads an environment variable or a file, which is what makes it
safe in a server holding several tenants' keys:

```go
client, err := ai.NewClient(ai.Config{
	Model: model, APIKey: key, BaseURL: "https://gateway.internal/v1",
	HTTPClient: httpClient, Headers: map[string]string{"X-Tenant": tenant},
})
```

`pkg/ai/auth` is the opt-in that does read it. `pkg/ai/provider` sits between a
catalog row and a client: one configured host and the models on it, where
reading the list and fetching it are separate verbs.

## Supported protocols

| Protocol | Package | Vendors in the catalog |
| --- | --- | --- |
| OpenAI Chat Completions | `pkg/ai/driver/openai/chat` | 18 |
| OpenAI Responses | `pkg/ai/driver/openai/responses` | 3 |
| Anthropic Messages | `pkg/ai/driver/anthropic` | 4 |
| Anthropic on Vertex AI | `pkg/ai/driver/anthropic/vertex` | 1 |
| Google Gemini | `pkg/ai/driver/google` | 1 |

A vendor is a catalog row, not a package: most ship an endpoint speaking
somebody else's protocol, so adding one is a data change.

## Documentation

**The client** — `pkg/ai`, one model call

| | |
| --- | --- |
| [API reference](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/ai) | Every type and function, with the rationale beside the code it governs |
| [Constructing a client](docs/clients.md) | One chain from a model reference to an `ai.Client`, and where to stop on it ([中文](docs/clients.zh-CN.md)) |
| [Architecture](docs/architecture.md) | How the pieces fit and what a request passes through ([中文](docs/architecture.zh-CN.md)) |

**The agent** — `pkg/agent`, the loop around it

| | |
| --- | --- |
| [API reference](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/agent) | The agent, its events, its hooks and its tools |
| [The Agent SDK](docs/agent.md) | Structure, the event contract, hooks, tools and sessions ([中文](docs/agent.zh-CN.md)) |

**The project**

| | |
| --- | --- |
| [Contributing](CONTRIBUTING.md) | Development setup, implementing a protocol, and the test suite |
| [Changelog](CHANGELOG.md) | What changed in each release |
| [`examples/`](examples) | Runnable programs, one per vendor plus tools, structured output and an agent |

## Versioning

Released under [semantic versioning](https://semver.org). While the major
version is `0` the API may still change between minor releases; the
[changelog](CHANGELOG.md) says what moved and what to write instead.

```sh
go get github.com/genai-io/sdk-go@v0.1.0
```

## License

[Apache 2.0](LICENSE)
