// Package llm is a provider-agnostic client for large language models.
//
// The package is organised in three layers, and each one is useful on its own:
//
//	llm            canonical types (Message, Request, Response, Event), the
//	               Driver seam, and Client — the streaming/aggregation engine.
//	               Zero third-party dependencies.
//	llm/driver/*   one package per *wire protocol*, not per vendor. Anthropic
//	               Messages, OpenAI Chat Completions, OpenAI Responses and
//	               Google GenAI are the wire formats in use, plus Anthropic
//	               Vertex, which is Messages reached through Google's
//	               endpoint and credentials. A vendor that speaks one of them
//	               needs no code of its own.
//	llm/jsonschema JSON Schema from Go types, and a validator for what a tool
//	               call or a structured answer actually gets wrong. A leaf:
//	               it imports nothing, not even the rest of this SDK.
//	llm/catalog    the vendor and model directory as data: which protocol a
//	               vendor speaks, its base URL, credential environment
//	               variables, reasoning dialect, context window and pricing.
//
// A vendor is therefore a row in the catalog, not a package. Adding an
// OpenAI-compatible endpoint is a data change.
//
// # Talking to a model
//
// The high-level path resolves a model from the catalog and opens the driver
// its protocol requires:
//
//	import (
//		"github.com/genai-io/sdk-go/pkg/llm"
//		"github.com/genai-io/sdk-go/pkg/llm/catalog"
//		_ "github.com/genai-io/sdk-go/pkg/llm/driver/all"
//	)
//
//	model, err := catalog.Model("anthropic/claude-opus-4-6")
//	client, err := llm.Open(llm.Config{Model: model, APIKey: key})
//
//	resp, err := client.Complete(ctx, &llm.Prompt{
//		System:   "You are concise.",
//		Messages: []llm.Message{llm.User("Explain goroutine leaks.")},
//	}, nil)
//
// The blank import registers the four protocol drivers; import only the ones
// you need (for example llm/driver/anthropic) to keep the other vendor SDKs
// out of your build.
//
// # Prompt and Options
//
// A request is two values, not one. Prompt is the conversation — system
// prompt, messages, tools — and Options is how to run it: output cap,
// temperature, reasoning effort, tool choice. The split means one Prompt can
// be sent to two models with different settings, and that a nil *Options is a
// meaningful "use the defaults".
//
// # Streaming
//
// Stream returns a range-over-func iterator. Iteration ends after the
// EventDone event, or immediately after an error:
//
//	for event, err := range client.Stream(ctx, prompt, opts) {
//		if err != nil {
//			return err
//		}
//		switch event.Type {
//		case llm.EventTextDelta:
//			fmt.Print(event.Text)
//		case llm.EventDone:
//			resp = event.Response
//		}
//	}
//
// # Credentials
//
// Nothing in this package reads the environment or the filesystem: a Config
// carries the key it is given. Package llm/auth is the opt-in convenience that
// resolves keys from environment variables, and it is a separate import so a
// server-side or multi-tenant caller never inherits process-wide credentials.
package llm
