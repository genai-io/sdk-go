// Package driver is the namespace for wire-protocol implementations. It has no
// code of its own; each subdirectory is one protocol.
//
// # What decides where a package goes
//
// A driver does one job: turn an ai.Request into an
// HTTP request, and turn the reply back into a stream of ai.Delta. Four things
// differ between the endpoints this SDK reaches:
//
//	the wire format    the JSON shapes of the request and the reply
//	the host           where to send it
//	the credential     what to present
//	the quirks         which field carries reasoning, whether a TTL is accepted
//
// Only the first needs code. A host is ai.Config.BaseURL, a credential is
// ai.Config.APIKey, and the quirks are ai.Model.Compat — all of them values
// a caller supplies, none of them a reason to compile anything new.
//
// So: one package per wire format. Two endpoints that speak the same format
// share the same code, and a second package for the second endpoint could only
// be a copy of the first or an empty wrapper around it. That is the whole rule,
// and everything below follows from it rather than from taste.
//
// # What follows
//
// A vendor is a row in the catalog, not a package. Eighteen of the vendors
// this SDK reaches — DeepSeek, Moonshot, Ollama, Groq, xAI, Alibaba, Z.ai,
// OpenRouter and the rest — speak OpenAI Chat Completions, so they share
// openai/chat and differ only in data. Adding one is an entry in
// ai/catalog/vendors.go and no Go file at all.
//
// A directory that groups rather than implements earns its place when it
// scopes an internal package to exactly the code allowed to use it. openai/
// holds two protocols because OpenAI defines two, and openai/internal/oai holds
// what they share — the client, the shape of an error from the SDK they both
// use, the framing of an inline image. The compiler, not a convention, is what
// keeps the Anthropic and Gemini drivers out of it.
//
// internal/errs is the same move one directory up. Turning a failure into an
// ai.Error is identical everywhere once the protocol's own error type has been
// read, and every driver here is entitled to it — so it sits where every driver,
// and nothing outside driver/, can reach it.
//
// A protocol served from somebody else's cloud is a deployment, not a protocol.
// anthropic/vertex speaks Anthropic Messages, calls into package anthropic for
// every line of translation, and takes even its Google authentication from
// Anthropic's own SDK. It is a subpackage of the protocol it speaks, so that
// importing anthropic does not drag in a cloud credential stack.
//
// # Before adding a package here
//
// Ask what code it needs that no existing package has. If the answer is a base
// URL, an environment variable, a reasoning dialect and a table of context
// windows, it is a catalog row. If the answer is a request or reply shape
// nothing here can already produce or read, it is a protocol, and it gets a
// package.
//
// # The packages
//
//	anthropic         Anthropic Messages
//	anthropic/vertex  the same, served through Google Cloud Vertex AI
//	google            Google Gemini generateContent
//	openai/chat       OpenAI Chat Completions — the industry interchange format
//	openai/responses  OpenAI Responses — the only one that replays reasoning state
//	all               a blank import that registers every one of them
//	internal/errs     failure classification, shared by all of them
package driver
