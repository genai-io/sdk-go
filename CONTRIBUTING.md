# Contributing

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
implemented here — OpenAI Chat Completions covers 18 of the 27 vendors in the
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
defaults, repairing history, validating, retrying — belongs to `Client` and is
written once for every protocol. Model listing and token counting are optional
interfaces, discovered by type assertion; a driver that omits one is never an
error.

Register from `init` so a blank import is enough to make the protocol
reachable.

A setting only your protocol has goes in a typed `ProtocolOptions` value rather
than a new field on `Request`:

```go
response, err := client.Complete(ctx, messages,
	ai.WithProtocolOptions(anthropic.Options{ThinkingDisplay: "omitted"}))
```

Your type implements `ai.ProtocolOptions` — a one-line marker method — so the
field is not a bare `any` and a value that was never meant to go there is a
compile error. A value of the *wrong driver's* type, or one sent to a protocol
that defines none, is an invalid request caught when that driver reads it. It
is never ignored silently. Construction settings work the same way through
`ai.ProtocolConfig`.

## Tests

The suite is one black-box package under `test/`. It imports the SDK the way an
application does and asserts on two things: the bytes that reached the
endpoint, and the value that came back. Every endpoint is a stub HTTP server,
so it needs no network and no credential.

```sh
go test ./test/
```

A new protocol needs its own entries in the cross-protocol tables there — the
ones that pin what each protocol sends for the same request — because that is
what keeps five drivers agreeing on one meaning.

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
