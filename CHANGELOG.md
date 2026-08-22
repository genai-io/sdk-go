# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the API may change between minor releases.
Each such change is listed under **Changed** with what to write instead.

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

[0.1.1]: https://github.com/genai-io/sdk-go/releases/tag/v0.1.1
[0.1.0]: https://github.com/genai-io/sdk-go/releases/tag/v0.1.0
