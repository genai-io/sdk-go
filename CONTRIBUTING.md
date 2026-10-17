# Contributing

## What belongs here

sdk-go is a set of building blocks for assembling agents — any agent, not one
application. [san](https://github.com/genai-io/san) is built on it; it does not
define it.

- **In:** what any agent builder would want — protocols, streaming, tool
  calling, structured output, token counting, typed errors, the agent loop and
  its hooks, sessions, compaction, MCP, credentials.
- **Out:** one application's policy and UI — prompt text, permission rules,
  rendering, product settings.
- **Once:** nothing is implemented both here and in an application. When an
  application grows a generic piece, it moves here and the application drops
  its copy.
- **Removing:** "no known caller" is not a reason to delete a capability.
  Remove only what duplicates another part of the SDK, has an equivalent, or is
  flexibility nobody asked for — and check san's callers before calling
  anything unused.

Generic pieces that still live in san move here one at a time, each in a
change of its own. Not here yet:

- [ ] Compaction: the summary flow, the budget trigger and context-exceeded
  recovery, as a package over `Hook.PreStep` and `Hook.OnInferError`.
- [ ] `PreStepContext.Tokens` anchored on the last call's usage, not a fresh
  estimate.
- [ ] Retry for a single `Complete`, sharing `agent.WithRetry`'s policy.
- [ ] Copilot's per-turn headers and DashScope's per-model limits.
- [ ] Dropping images from history for a model that cannot read them.

## Development

```sh
make build    # go build ./...
make test     # go test -race ./...
make lint     # go vet, plus the format check
make format   # gofmt and goimports, in place
make ci       # everything CI runs
```

CI runs against the Go version in `go.mod` and the current release. A library
has to build on the version it claims to support, so both have to pass.

## Commits

Every commit needs a `Signed-off-by` trailer matching its author — this is the
[Developer Certificate of Origin](https://developercertificate.org/), and a
workflow checks it:

```sh
git commit -s
```

Pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/),
which a workflow also checks. The accepted types are `feat`, `fix`, `docs`,
`style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore` and `revert`; a
scope is optional.

## Adding a vendor

Most vendors do not need code. If the endpoint speaks a protocol already
implemented here — OpenAI Chat Completions covers 18 of the 28 vendors in the
catalog — then adding it is a row in `pkg/ai/catalog`: a base URL, the
environment variable it documents, its reasoning dialect and its models.

## Implementing a protocol

A new protocol is a new package. The driver interface is two methods:

```go
type Driver interface {
	Name() string
	Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error]
}
```

Everything a caller also needs — aggregating deltas into a `Response`, applying
defaults, repairing history, validating — belongs to `Client` and is written
once for every protocol. Model listing and token counting are optional
interfaces, discovered by type assertion; a driver that omits one is never an
error.

Register from `init` so a blank import is enough to make the protocol
reachable.

A setting only your protocol has is not a new field on `Request`. If your
protocol is in the OpenAI family, `SamplingParams` already reaches it. A
construction setting the protocol cannot run without is a typed field on
`ai.Config`, the way `Config.Vertex` is: the drivers that need it require it,
and the others ignore it.

## Tests

Two layers, and a change usually needs one of each.

`integration/` is one black-box package. It imports the SDK the way an application
does and asserts on two things: the bytes that reached the endpoint, and the
value that came back. Every endpoint is a stub HTTP server, so it needs no
network and no credential.

Beside it, each package tests its own units — the stream lifecycle, history
repair, error classification, schema derivation, the SSE parser, tool-call
accumulation, the catalog's invariants, the credential store. Nothing here
needs a network either.

```sh
make test          # go test -race ./...
go test ./integration/  # the cross-protocol suite alone
```

A new protocol needs its own entries in the cross-protocol tables in `integration/` —
the ones that pin what each protocol sends for the same request — because that
is what keeps six drivers agreeing on one meaning. It also needs its own
package tests for the parts no other protocol exercises.

A new vendor needs no code, and `pkg/ai/catalog` will tell you if the row is
wrong: `TestCatalogInvariants` pins that a ladder on an OpenAI-Chat endpoint
states a `Thinking` dialect, that IDs are unique, and that a `Compat` value
matches its `API`. `golden_test.go` holds the whole resolved table, so a row
that changes another row's meaning shows up as a diff rather than as a surprise
months later.

## Testing your own code

Because `Driver` is two methods, testing an application against a model means
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

## Layout

Two rules explain most of it. A package is a unit of code, so there is one per
wire format and none per vendor. And a file is named for the subject it owns
and owns all of it — if two files hold part of one idea, one of them is wrong.

The rationale lives with the code it governs:

```sh
go doc github.com/genai-io/sdk-go/pkg/ai         # the request, the result, and what each file owns
go doc github.com/genai-io/sdk-go/pkg/ai/driver  # why a driver is a package and a vendor is a table row
go doc github.com/genai-io/sdk-go/pkg/ai/auth    # why credential resolution is a separate import
```

[`docs/architecture.md`](docs/architecture.md) is the part no single file owns:
how the pieces fit, which direction the dependencies run, and what a request
passes through between your call and the bytes on the wire.
