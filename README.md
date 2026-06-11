# sdk-go

Go SDK for [San](https://github.com/genai-io/san) — LLM client tools and agent integration.

## Packages

- **`pkg/llm`** — LLM Client SDK: direct model access through San's multi-provider
  abstraction, streaming and non-streaming completions, and model discovery.
- **`pkg/san`** — San Agent SDK: full agent lifecycle (create, configure, run),
  an event stream for external consumers, and a functional-options configuration API.

## Status

Early development. See [ADR-0002](docs/design/decisions/0002-sdk-architecture.md)
and [issue #1](https://github.com/genai-io/sdk-go/issues/1) for the design and roadmap.

## License

[Apache 2.0](LICENSE)
