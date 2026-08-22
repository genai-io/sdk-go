# sdk-go

A Go client library for large language model inference, providing one typed API
over the Anthropic Messages, OpenAI Chat Completions, OpenAI Responses and
Google Gemini protocols.

You get the same `Message`, `Response` and streaming event types whichever
provider serves the request. The library ships a catalog of 27 vendors and 55
models — endpoints, limits, pricing and per-endpoint quirks as data — and reads
no credentials unless you opt in.

> 中文文档：[README.zh-CN.md](README.zh-CN.md)

## Requirements

Go 1.24 or later.

## Installation

```sh
go get github.com/genai-io/sdk-go
```

## Usage

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
documents — `OPENAI_API_KEY` here. To supply it yourself, see
[Credentials](#credentials).

The blank import registers one wire protocol. Import only the protocols you
use; each pulls in that protocol's vendor SDK.

### Switching providers

Change the model reference. Nothing else in the code above changes.

```go
client, err := auth.Client("anthropic/claude-opus-5")
client, err := auth.Client("google/gemini-3.5-flash")
client, err := auth.Client("deepseek/deepseek-v4-pro")
client, err := auth.Client("ollama/llama4")
```

Swap the blank import for that model's protocol, or use
`pkg/ai/driver/all` when the model is chosen at runtime.

Runnable examples for each vendor are in [`examples/`](examples).

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
returns.

The client closes every block it started, including on failure. Abandoning the
iterator cancels the request.

## Request options

The conversation is an ordinary `[]ai.Message`. Everything else is an `Option`,
and the same option is a default at `ai.New` and an override at the call:

```go
client := ai.New(driver, model, ai.WithEffort(ai.EffortHigh)) // default

response, err := client.Complete(ctx, messages,
	ai.WithTemperature(0),
	ai.WithMaxTokens(4096),
	ai.WithEffort(ai.EffortLow),                // overrides the default
	ai.WithStopSequences("\n\n"),
	ai.WithCacheRetention(ai.CacheLong))
```

Passing an option is what marks a setting as explicit, so `WithTemperature(0)`
is deterministic sampling and omitting it inherits. Resolution runs model
defaults, then client defaults, then call overrides.

`Model` and returned model lists cross their boundaries as deep snapshots, and
history repair never writes to the messages you passed, so later mutation on
your side cannot rewrite an in-flight request.

### Reasoning effort

Reasoning is requested as a normalized rung. Each model carries its own ladder
mapping that rung onto whatever its endpoint wants — a token budget, a level
string, an enable flag — so the same request runs anywhere:

```go
ai.WithEffort(ai.EffortHigh)
```

A model that does not offer the rung asked for is snapped onto the nearest one
it does. `Model.Efforts()` reports what a given model offers.

## Messages and content

A message carries an ordered sequence of typed blocks rather than parallel
fields, because that order is what the next request has to replay.

```go
messages := []ai.Message{
	ai.UserMessage("What is in this image?", image),
	ai.AssistantMessage("A gopher."),
	ai.ToolResultsMessage(result),
}
```

For an order the constructors do not produce, write the message out:

```go
ai.Message{Role: ai.RoleUser, Content: ai.Content{
	ai.TextBlock("compare "),
	ai.ImageBlock(imageA),
	ai.TextBlock(" with "),
	ai.ImageBlock(imageB),
}}
```

Accessors project one kind when order does not matter:

```go
text := response.Text()
thinking := response.Thinking()
calls := response.ToolCalls()

history = append(history, response.Message()) // keeps every block, in order
```

Append `response.Message()` rather than `ai.AssistantMessage(response.Text())`:
the former carries the model's thinking and reasoning state forward, which is
what lets a reasoning model resume instead of starting over.

Invalid tagged payloads and blocks on the wrong role fail before a driver is
called.

## Tool use

Tool parameters are derived from a Go type, so the schema the model sees and
the struct the arguments decode into cannot drift apart:

```go
type SearchArgs struct {
	Query string `json:"query" jsonschema:"what to look for"`
	Limit int    `json:"limit,omitempty"`
}

search := ai.ToolFor[SearchArgs]("search", "Search the knowledge base")

response, err := client.Complete(ctx, messages, ai.WithTools(search))

for _, call := range response.ToolCalls() {
	if err := search.ValidateArgs(call.Input); err != nil {
		// Hand this back as a tool result; the model can correct it.
	}
	args, err := ai.UnmarshalArgs[SearchArgs](call)
	// ...
}
```

`ValidateArgs` checks a call against the tool's own schema before you run it, so
a model's mistake comes back as something it can fix rather than as whatever
your tool does with a missing field.

Answer each call in the turn that follows:

```go
messages = append(messages, response.Message())
messages = append(messages, ai.ToolResultsMessage(ai.ToolResult{
	ToolCallID: call.ID,
	ToolName:   call.Name,
	Content:    output,
}))
```

## Describing fields

The `jsonschema` tag is a field's description, and the model reads it. Field
names are often ambiguous to a model in ways they are not to you — `name` could
be a display name or a legal one, `age` could be years or a date of birth — so
this is where you disambiguate.

Two spellings look right and are not. Both fail loudly at construction, naming
the fix, rather than shipping a schema the model cannot use:

```go
jsonschema:"what to look for"              // correct: the whole tag is the text
jsonschema:"description=what to look for"  // panics: another library's grammar
description:"what to look for"             // panics: Go would drop this silently
```

The key is `jsonschema` because that is what the ecosystem's other schema
libraries use, so it is the one people reach for. It is checked rather than
renamed precisely because Go ignores an unrecognised tag key without a word.

## Structured outputs

```go
type Person struct {
	Name string `json:"name" jsonschema:"full legal name, family name last"`
	Age  int    `json:"age" jsonschema:"age in whole years"`
}

person, err := ai.CompleteAs[Person](ctx, client, messages)
```

Field descriptions work exactly as they do for a tool, above.

`CompleteAs` derives the schema from `Person`, constrains generation to it and
decodes the answer — so the type is named once. Add `ai.WithSchema` to give the
model a description of the shape, or to constrain to something the Go type does
not capture exactly.

Every supported protocol constrains generation natively. For a model that
cannot, ask for the shape in the prompt and decode with `Response.Unmarshal`,
which tolerates a fenced or prose-prefaced answer.

## Token counting

```go
count, err := client.CountTokens(ctx, messages)
if count.Exact {
	// The provider's own tokenizer counted it.
}

left, count, err := client.Headroom(ctx, messages)
```

Anthropic and Gemini publish a counting endpoint and are exact; the
OpenAI-family protocols do not, and the result says which happened rather than
presenting an estimate as a measurement. A model whose context window is
unknown reports zero headroom rather than a guess.

After a call, `ai.IsOverflow(response, client.Model())` reports an overflow —
including on the providers that truncate silently instead of erroring.

## Handling errors

Failures are classified, so the answer to "what now" is in the type rather than
in the message text:

```go
switch {
case ai.IsAuth(err):
	// The credential is wrong or missing.
case ai.IsContextExceeded(err):
	// Compact the conversation and retry.
case ai.IsRetryable(err):
	time.Sleep(ai.RetryAfter(err))
case ai.IsUnsupported(err):
	// This model cannot do what was asked; caught before the network.
}
```

A failed turn returns both a non-nil error and a non-nil `*Response` carrying
what arrived first — the text already streamed and the tokens already billed —
so a partial answer and its cost are not lost with the error.

Requests are validated before they leave: an image sent to a text-only model, a
tool call with no matching result, a schema on a model that cannot constrain
output. Each fails with a sentence naming the model and the reason.

The library does not rewrite your request to make it work. A model with no
system role, or one that cannot constrain output to a schema, is reported as an
error naming the model — moving the instructions into a user turn, or asking
for JSON in words, is a decision about your product, not about the wire.

## Retries and other execution policy

Execution policy is `Middleware` decorating a driver, because only your
application knows the budget for a turn, what may be cached and what must not
be logged:

```go
client := ai.New(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), model)
```

`Retry` is the one policy this library ships. Every driver disables its vendor
SDK's own retry, so without it you get none — and the rule for writing one is
easy to get wrong in a way that shows up as duplicated or vanished output
rather than as an error. It replays only when the failure is classified
retryable, only when nothing has reached you yet, and only while the context is
live; a provider's own `Retry-After` beats the backoff.

Caching, logging and cost metering stay yours: a `Middleware` is a `Handler`
wrapping a `Handler`, which is the same shape as `Driver.Stream`.

## Credentials

`pkg/ai` never reads an environment variable or a file. That is what makes it
safe in a server holding several tenants' keys — supply a `Config` directly:

```go
client, err := ai.NewClient(ai.Config{
	Model:      model,
	APIKey:     key,
	BaseURL:    "https://gateway.internal/v1",
	HTTPClient: httpClient,
	Headers:    map[string]string{"X-Tenant": tenant},
})
```

`pkg/ai/auth` is the opt-in that does read the environment, which is what a
command-line tool wants. It resolves the variables each vendor documents, and
runs the browser sign-in for the vendors that authenticate a person rather than
a service (GitHub Copilot, ChatGPT/Codex). The only file this library writes is
that credential store, `0600` under the user's config directory.

## Providers and model listings

Three layers, named for what tells them apart:

| Type | What it is |
| --- | --- |
| `catalog.Vendor` | Data. A row you can read without a network. |
| `provider.Provider` | That row configured and credentialed, with a model list you can refresh. |
| `ai.Client` | One model on it. |

```go
p, err := auth.Provider("ollama")
models := p.Models()       // synchronous, never blocks, never fails
err = p.Refresh(ctx)       // the only call that reaches the network
client, err := p.Client("llama4")
```

Reading the list and fetching it are separate verbs so a model picker renders
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

## Implementing a protocol

The driver interface is two methods:

```go
type Driver interface {
	Name() string
	Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error]
}
```

Everything a caller also needs — aggregating deltas into a `Response`, applying
defaults, repairing history, validating, retrying — belongs to `Client` and is
written once for every protocol. Model listing and token counting are optional
interfaces, discovered by type assertion; a driver that omits one is never an
error.

Register from `init` so a blank import is enough to make the protocol
reachable. A setting only your protocol has goes in a typed `ProtocolOptions` value
rather than a new field on `Request`:

```go
response, err := client.Complete(ctx, messages,
	ai.WithProtocolOptions(anthropic.Options{ThinkingDisplay: "omitted"}))
```

Your type implements `ai.ProtocolOptions` — a one-line marker method — so the
field is not a bare `any` and a value that was never meant to go there is a
compile error. A value of the *wrong driver's* type, or one sent to a protocol
that defines none, is an invalid request caught when that driver reads it. It
is never ignored silently.

Construction settings work the same way through `ai.ProtocolConfig`.

## Testing

Because `Driver` is two methods, testing your own code against a model means
writing the stub your case needs:

```go
type scripted struct{ deltas []ai.Delta }

func (scripted) Name() string { return "scripted" }

func (s scripted) Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		for _, d := range s.deltas {
			if !yield(d, nil) {
				return
			}
		}
	}
}
```

This library's own suite is one black-box package under `test/`. It imports the
SDK the way an application does and asserts on two things: the bytes that
reached the endpoint, and the value that came back. Every endpoint is a stub
HTTP server, so it needs no network and no credential.

```sh
go test ./test/
```

## Design notes

The rationale lives with the code it governs:

```sh
go doc github.com/genai-io/sdk-go/pkg/ai         # the request, the result, and what each file owns
go doc github.com/genai-io/sdk-go/pkg/ai/driver  # why a driver is a package and a vendor is a table row
go doc github.com/genai-io/sdk-go/pkg/ai/auth    # why credential resolution is a separate import
```

Two rules explain most of the layout. A package is a unit of code, so there is
one per wire format and none per vendor. And a file is named for the subject it
owns and owns all of it — if two files hold part of one idea, one of them is
wrong.

[`docs/architecture.md`](docs/architecture.md) is the part no single file owns:
how the pieces fit, which direction the dependencies run, and what a request
passes through between your call and the bytes on the wire.
([中文](docs/architecture.zh-CN.md))

## Versioning

The API is in early development and may change. Pin a commit until a tagged
release is published.

## License

[Apache 2.0](LICENSE)
