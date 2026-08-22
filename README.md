# sdk-go

[![Go Reference](https://pkg.go.dev/badge/github.com/genai-io/sdk-go.svg)](https://pkg.go.dev/github.com/genai-io/sdk-go)
[![CI](https://github.com/genai-io/sdk-go/actions/workflows/ci.yml/badge.svg)](https://github.com/genai-io/sdk-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/github/go-mod/go-version/genai-io/sdk-go)](go.mod)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

A Go client library for large language model inference, providing one typed API
over the Anthropic Messages, OpenAI Chat Completions, OpenAI Responses and
Google Gemini protocols.

You get the same `Message`, `Response` and streaming event types whichever
provider serves the request. The library ships a catalog of 27 vendors and 55
models — endpoints, limits, pricing and per-endpoint quirks as data — and reads
no credentials unless you opt in.

> 中文文档：[README.zh-CN.md](README.zh-CN.md)

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

### Switching providers

Change the model reference. Nothing else changes.

```go
client, err := auth.Client("anthropic/claude-opus-5")
client, err := auth.Client("google/gemini-3.5-flash")
client, err := auth.Client("deepseek/deepseek-v4-pro")
client, err := auth.Client("ollama/llama4")
```

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

`EventBlockDelta` carries a fragment; `EventBlockEnd` carries the complete
block, and arrives as soon as an atomic value such as a tool call is assembled.
`EventDone` carries the aggregated `Response` — the same value `Complete`
returns. The client closes every block it started, including on failure, and
abandoning the iterator cancels the request.

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

The model does not run your tools: it asks you to, you answer, and it
continues. `Run` is that loop, and `history` is the whole conversation, so a
follow-up continues from it.

`SearchArgs` is named once, so the schema sent to the model and the struct its
arguments decode into cannot come to describe different things:

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

Arguments are checked against that schema before your function runs, so a
model's mistake comes back as something it can correct rather than as whatever
your tool does with a missing field. An unknown tool name, bad arguments and a
failing tool all return to the model as a result marked `IsError` rather than
ending the conversation.

Constraining the choice is one value with four states — the same four every
protocol here expresses:

```go
ai.WithToolChoice(ai.ToolChoiceNone)            // no tool this turn
ai.WithToolChoice(ai.ToolChoiceRequired)        // some tool, the model's choice which
ai.WithToolChoice(ai.ToolChoiceNamed("search")) // this one
```

`ToolFunc` returns an ordinary `ai.Tool` — three fields that go on the wire and
one function — so a hand-written schema is an assignment and a tool whose shape
is not known until run time is that value written directly. Write the turn loop
yourself with `Complete` and `RunTools` when the turns are your business: to
stream text as it arrives, to stop on a condition, to bill each one.

## Structured outputs

```go
type Person struct {
	Name string `json:"name" description:"full legal name, family name last"`
	Age  int    `json:"age" description:"age in whole years" minimum:"0"`
}

person, err := ai.CompleteAs[Person](ctx, client, messages)
```

`CompleteAs` derives the schema from `Person`, constrains generation to it and
decodes the answer, so the type is named once. Every supported protocol
constrains generation natively.

A tag key is the JSON Schema keyword it sets — `description`, `enum`,
`minimum`, `maximum` and eight more — and every word of it is prompt text.
Schemas are derived to be *accepted*, not merely valid: every field in
`required`, every object closed, optionality as `["T","null"]`, because that is
what a provider's strict mode demands. See
[`pkg/ai/jsonschema`](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/ai/jsonschema)
for the vocabulary and the rules.

## Request options

The conversation is an ordinary `[]ai.Message`. Everything else is an `Option`,
and the same option is a default at `ai.New` and an override at the call:

```go
client := ai.New(driver, model, ai.WithEffort(ai.EffortHigh)) // default

response, err := client.Complete(ctx, messages,
	ai.WithTemperature(0),
	ai.WithMaxTokens(4096),
	ai.WithEffort(ai.EffortLow), // overrides the default
	ai.WithCacheRetention(ai.CacheLong))
```

Passing an option is what marks a setting as explicit, so `WithTemperature(0)`
is deterministic sampling and omitting it inherits. Resolution runs model
defaults, then client defaults, then call overrides.

Reasoning is requested as a normalized rung. Each model carries its own ladder
mapping that rung onto whatever its endpoint wants — a token budget, a level
string, an enable flag — so the same request runs anywhere, and a model that
does not offer the rung asked for is snapped onto the nearest one it does.

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

Append `response.Message()` rather than `ai.AssistantMessage(response.Text())`:
the former carries the model's thinking and reasoning state forward, which is
what lets a reasoning model resume instead of starting over. For an order the
constructors do not produce, write the `ai.Message` out with its `Content`.

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

A failed turn returns both a non-nil error and a non-nil `*Response` carrying
what arrived first, so a partial answer and its cost are not lost with the
error. Requests are validated before they leave — an image sent to a text-only
model, a tool call with no matching result — and the library does not rewrite
your request to make it work: a model with no system role is reported as an
error naming the model, because moving those instructions into a user turn is a
decision about your product, not about the wire.

Execution policy is `Middleware` decorating a driver, because only your
application knows the budget for a turn, what may be cached and what must not
be logged:

```go
client := ai.New(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), model)
```

`Retry` is the one policy shipped here, and every driver disables its vendor
SDK's own, so without it you get none. Caching, logging and cost metering stay
yours.

## Credentials

`pkg/ai` never reads an environment variable or a file, which is what makes it
safe in a server holding several tenants' keys:

```go
client, err := ai.NewClient(ai.Config{
	Model: model, APIKey: key, BaseURL: "https://gateway.internal/v1",
	HTTPClient: httpClient, Headers: map[string]string{"X-Tenant": tenant},
})
```

`pkg/ai/auth` is the opt-in that does read the environment, which is what a
command-line tool wants. `pkg/ai/provider` is the layer between a catalog row
and a client: one configured host and the list of models on it, where reading
the list and fetching it are separate verbs so a model picker renders
immediately and a dead endpoint cannot hang it.

## Supported protocols

| Protocol | Package | Vendors in the catalog |
| --- | --- | --- |
| OpenAI Chat Completions | `pkg/ai/driver/openai/chat` | 18 |
| OpenAI Responses | `pkg/ai/driver/openai/responses` | 3 |
| Anthropic Messages | `pkg/ai/driver/anthropic` | 4 |
| Anthropic on Vertex AI | `pkg/ai/driver/anthropic/vertex` | 1 |
| Google Gemini | `pkg/ai/driver/google` | 1 |

A vendor is a catalog row, not a package. Most vendors ship an endpoint
speaking somebody else's protocol, so adding one is a data change in
`pkg/ai/catalog` — not another HTTP implementation.

## Documentation

| | |
| --- | --- |
| [API reference](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/ai) | Every type and function, with the rationale beside the code it governs |
| [Architecture](docs/architecture.md) | How the pieces fit and what a request passes through ([中文](docs/architecture.zh-CN.md)) |
| [Contributing](CONTRIBUTING.md) | Development setup, implementing a protocol, and the test suite |
| [`examples/`](examples) | Runnable programs, one per vendor plus tools and structured output |

## Versioning

The API is in early development and may change. Pin a commit until a tagged
release is published.

## License

[Apache 2.0](LICENSE)
