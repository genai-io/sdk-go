// Package ai is a provider-neutral core for calling a language model.
//
// Its stable primitive is an ordered Content sequence. Message, Response and
// streamed Event all carry the same tagged Block values, so text, images,
// visible thinking, opaque reasoning state, tool calls and tool results cannot
// be reordered by parallel fields.
//
// The package has no provider SDK dependency and does not read credentials,
// environment variables or files. Wire protocols live in ai/driver/*;
// catalog data and opt-in credential discovery live in ai/catalog and
// ai/auth.
//
// # Complete a prompt
//
// Import a driver for registration, then open a configured model:
//
//	import (
//		"github.com/genai-io/sdk-go/pkg/ai"
//		"github.com/genai-io/sdk-go/pkg/ai/catalog"
//		_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
//	)
//
//	model, err := catalog.Model("anthropic/claude-opus-5")
//	client, err := ai.NewClient(ai.Config{Model: model, APIKey: key})
//	messages := []ai.Message{ai.UserMessage("Explain goroutine leaks.")}
//	response, err := client.Complete(ctx, messages,
//		ai.WithSystem("You are concise."),
//		ai.WithTemperature(0))
//	if err == nil {
//		fmt.Println(response.Text())
//	}
//
// Every constructor is named for what it returns: auth.Client and
// Provider.Client hand back a *Client, auth.Provider and Vendor.Provider hand
// back a *provider.Provider, NewDriver hands back a Driver. New is the one
// that follows Go's own convention instead — it returns this package's main
// type from parts you already hold, where NewClient resolves them from a
// Config.
//
// The conversation is an ordinary []Message; everything else is an Option.
// The same Option is a default when given to New and an override when given to
// a call, and passing one is what marks a setting as explicit — so
// WithTemperature(0) is deterministic sampling, while omitting it inherits.
//
// # Two ways in
//
// There are two package-level ways to get a Client, and the difference is
// whether the environment is allowed to answer:
//
//	ai.NewClient(Config)         you supply the model, the key and the host
//	auth.Client("vendor/model")  the catalog and the environment supply them
//
// Core ai never reads an environment variable or a file, which is what makes
// it safe in a server holding several tenants' keys. Package auth is the
// opt-in that does, and a command-line tool wants exactly that.
//
// Holding an ai/provider.Provider — a configured host with a live model list —
// its Client method is the same thing scoped to that host:
//
//	client, err := ep.Client("llama4")
//
// # Stream blocks
//
// Every content kind uses one start/delta/end lifecycle. Textual Delta blocks
// contain fragments; End blocks contain the complete value:
//
//	for event, err := range client.Stream(ctx, messages) {
//		if err != nil {
//			return err
//		}
//		switch event.Type {
//		case ai.EventBlockDelta:
//			if event.Block.Type == ai.BlockText {
//				fmt.Print(event.Block.Text)
//			}
//		case ai.EventBlockEnd:
//			if event.Block.Type == ai.BlockToolCall {
//				go runTool(*event.Block.ToolCall)
//			}
//		case ai.EventDone:
//			response = event.Response
//		}
//	}
//
// # Middleware
//
// Middleware wraps execution policy — retry, logging, caching, metering. It
// decorates the driver, not the client, so the composition is visible where
// the client is built: ai.New(ai.Wrap(driver, retry), model). The policy
// itself belongs to the caller, who alone knows the budget for a turn and what
// must not be logged. One rule is not theirs to discover: a retry may only
// replay a call that failed before any delta reached the caller.
//
// # Where things live
//
// A file is named for the subject it owns, and owns all of it. If two files
// both hold part of one idea, one of them is wrong — and if a file's name
// stops describing what is in it, the file has drifted. That rule is why the
// list below can be kept true, and why it is worth keeping:
//
//	request.go     one call: what is asked, under what settings, and how they layer
//	message.go     a turn, and the ordered blocks it carries
//	tool.go        a tool offered to the model, and its arguments checked on the way back
//	schema.go      asking a model to answer in a JSON shape, and the JSON Schema behind it
//
//	response.go    what one call produced, what it cost, and decoding it
//	errors.go      the failure categories, and how a driver's error becomes one
//
//	model.go       what a model is, including what it cannot do
//	compat.go      where one endpoint differs from its protocol's owner
//	pricing.go     a rate card, and the money one call cost
//
//	client.go      one call: prepared, wrapped, run, and streamed back
//	history.go     the repairs every protocol needs
//	validate.go    everything caught before the network
//	tokens.go      how large a prompt is, and whether it fit
//
//	driver.go      the protocol seam: what a driver is, is given, and how one is found
//
// A set of endpoints and their live model listings is ai/provider, which sits
// above Client rather than inside it.
//
// Another modality — image generation, say — would be a sibling package rather
// than more of this one: it shares a Model, a Config and an Error with what is
// here, and nothing else. A conversation is not a thing you ask for an image.
package ai
