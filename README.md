# sdk-go

Provider-neutral Go primitives for LLM inference.

```go
import (
    "fmt"

    "github.com/genai-io/sdk-go/pkg/ai"
    "github.com/genai-io/sdk-go/pkg/ai/auth"
    _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)

client, err := auth.Client("openai/gpt-4.1")
if err != nil {
    return err
}

response, err := client.Complete(ctx,
    []ai.Message{ai.UserMessage("Explain goroutine leaks.")},
    ai.WithSystem("You are concise."))
if err != nil {
    return err
}
fmt.Println(response.Text())
```

Change the model reference to a Claude, Gemini, DeepSeek, Qwen, Ollama or
other catalog model; the prompt and response code stays the same. Swap the blank
import for that model's protocol — or use `driver/all` if the model is chosen
at runtime and you need every one of them.

## Package boundaries

| Package | Owns |
| --- | --- |
| `pkg/ai` | Semantic types, request preparation, validation, streaming aggregation, driver interfaces and provider state. No provider SDKs or ambient credential lookup. |
| `pkg/ai/driver/*` | One package per wire protocol: `anthropic` (and `anthropic/vertex`), `google`, `openai/chat`, `openai/responses`. |
| `pkg/ai/catalog` | Vendors and models as data: API, endpoint, limits, pricing and protocol compatibility. |
| `pkg/ai/auth` | Opt-in environment and interactive credential resolution. |

A vendor is normally a catalog row, not a package. A new provider that speaks
an existing protocol needs data, not another HTTP implementation.

## The core primitive: ordered blocks

`Content` is an ordered `[]Block`. The same primitive flows through `Message`,
`Response` and `Event`:

| Block type | Payload | Role |
| --- | --- | --- |
| `BlockText` | `Block.Text` | user or assistant |
| `BlockImage` | `Block.Image` | user |
| `BlockThinking` | `Block.Text`, optional provider signature | assistant |
| `BlockReasoning` | opaque replay state | assistant |
| `BlockToolCall` | `Block.ToolCall` | assistant |
| `BlockToolResult` | `Block.ToolResult` | user |

There are no parallel text/tool/reasoning fields to drift out of order.
Convenience accessors project a kind when order is not needed:

```go
text := response.Text()
thinking := response.Thinking()
calls := response.ToolCalls()

history = append(history, response.Message()) // preserves every block in order
```

Multimodal input is explicit:

```go
message := ai.Message{Role: ai.RoleUser, Content: ai.Content{
    ai.TextBlock("compare "),
    ai.ImageBlock(imageA),
    ai.TextBlock(" with "),
    ai.ImageBlock(imageB),
}}
```

Invalid tagged payloads and blocks on the wrong role fail before a driver is
called.

## One option type, two positions

The conversation is an ordinary `[]Message`. Everything else is an `Option`,
and the same option is a default at `New` and an override at the call:

```go
client := ai.New(driver, model, ai.WithEffort(ai.EffortHigh)) // default
response, err := client.Complete(ctx, messages,
    ai.WithMaxTokens(0),      // explicitly omit the wire cap
    ai.WithTemperature(0),    // explicitly deterministic
    ai.WithStopSequences(),   // clear the default list
    ai.WithEffort(ai.EffortLow))
```

Passing an option is what makes a setting explicit, so an explicit zero and an
omission stay distinct without a presence wrapper around every field.
Resolution is deterministic: model defaults, then client defaults, then call
overrides. `Model` and returned model lists cross their boundaries as deep
snapshots, and history repair never writes to the messages you passed, so
later caller mutation cannot rewrite an in-flight or recorded request.

## Streaming

All content kinds use one lifecycle. `EventBlockDelta` carries a fragment;
`EventBlockEnd` carries the complete block and arrives as soon as an atomic
tool call is assembled.

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
        usage = event.Response.Usage
    }
}
```

The client closes every started block, including on failure. Protocols that
publish boundaries refine them with `Delta.EndBlock`; otherwise a change of
block type establishes the boundary.

`Complete` returns both a partial `Response` and an error when a stream fails
after producing output. The partial blocks and usage are real and should not be
discarded.

## Execution policy is the caller's

`Middleware` wraps a call. Retry, caching, request logging and cost metering go
here, and the policy is yours: only your application knows the budget for a
turn, what may be cached, and what must not be logged.

```go
client := ai.New(ai.Wrap(driver, myRetry, myCostMeter), model)
```

One rule is not yours to discover. A retry may only replay a call that failed
*before producing any delta* — once output has reached you the answer has
begun, and resending would either duplicate the text already shown or silently
discard it. `IsRetryable` and `RetryAfter` classify the failure; give up once
anything has streamed.

The SDK also does not rewrite your request. A model with no system role, or one
that cannot constrain output to a schema, is reported as an error naming the
model — putting the instructions in a user turn instead, or asking for JSON in
words, is a decision about your product, not about the wire.

## Tools and structured output

Derive a tool schema and its decoder from one Go type:

```go
type SearchArgs struct {
    Query string `json:"query" jsonschema:"description=what to look for"`
    Limit int    `json:"limit,omitempty"`
}

tool := ai.ToolFor[SearchArgs]("search", "search the web")
args, err := ai.UnmarshalArgs[SearchArgs](call)
```

`Tool.ValidateArgs` checks a call's arguments against the tool's own schema
before you run it, so a model's mistake comes back as something it can correct
rather than as whatever your tool does with nonsense.

Native structured output uses the same `Schema` across protocols, derived from
the same Go type the answer decodes into:

```go
response, err := client.Complete(ctx, messages,
    ai.WithSchema(ai.SchemaOf[Person]("person", "one person's details")))
person, err := ai.Parse[Person](response, err)
```

## Token sizing and overflow

```go
count, err := client.CountTokens(ctx, messages)
if count.Exact {
    // Anthropic or Gemini counted the prepared request.
}

left, count, err := client.Headroom(ctx, messages)
if ai.IsOverflow(response, client.Model()) {
    // Compact and retry, including providers that truncate silently.
}
```

Protocols without a counting endpoint use an explicit estimate. Unknown model
windows stay unknown rather than being replaced by a guess.

## Drivers and protocol extensions

The required driver boundary is deliberately small:

```go
type Driver interface {
    Name() string
    Generate(context.Context, *Request) iter.Seq2[Delta, error]
}
```

Model enumeration is the optional `ModelLister` capability. `Client.Models`
returns `KindUnsupported` for a generation-only driver.

Normalized settings get a `With…` option. A genuinely protocol-only setting
uses that driver's typed `Native` value:

```go
response, err := client.Complete(ctx, messages,
    ai.WithNative(anthropic.Native{ThinkingDisplay: "omitted"}))
```

Wrong native types and native values sent to a protocol with no native options
are invalid requests. They are never ignored silently. OpenAI-only sampling
extensions are likewise rejected by Anthropic and Google drivers.

## Catalog, endpoints and credentials

Three layers, named for what tells them apart. `catalog.Vendor` is data — a row
you can read without a network. `endpoint.Endpoint` is that row configured and
credentialed, with a model list you can refresh. `ai.Client` is one model on it.

```go
ep, err := auth.Endpoint("ollama")
models := ep.Models()             // synchronous, never blocks
err = ep.Refresh(ctx)             // explicit network refresh
client, err := ep.Open("llama4")
```

`pkg/ai` never discovers credentials. Supply `Config` directly in a server,
or opt into `pkg/ai/auth` for environment variables and supported browser
flows.

## Tests

One suite, in `test/`, black-box: it imports the SDK the way an application
does and asserts on two things only — the bytes that reached the endpoint, and
the value that came back.

```
go test ./test/
```

Every endpoint in it is a stub HTTP server, so it needs no network and no
credential. `Driver` is two methods, so testing your own code against a model
means writing the stub your case needs:

```go
type scripted struct{ deltas []ai.Delta }

func (scripted) Name() string { return "scripted" }

func (s scripted) Generate(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
    return func(yield func(ai.Delta, error) bool) {
        for _, d := range s.deltas {
            if !yield(d, nil) {
                return
            }
        }
    }
}
```

## Why it is shaped this way

The reasoning lives with the code it governs, not in a parallel set of
documents that has to be kept in step with it:

```
go doc ./pkg/ai/driver    why a driver is a package and a vendor is a table row
go doc ./pkg/ai           the request, the result, and where each file's subject lives
go doc ./pkg/ai/auth      why credential resolution is a separate import
```

Two rules explain most of the layout, and both are stated where they apply. A
package is a unit of code, so there is one per wire format and none per vendor:
seventeen of the vendors here speak OpenAI Chat Completions and differ only in
data. And a file is named for the subject it owns and owns all of it — if two
files hold part of one idea, one of them is wrong.

The API is still in early development and may change.

## License

[Apache 2.0](LICENSE)
