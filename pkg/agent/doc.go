// Package agent runs a conversation: it calls a model, executes the tools the
// model asks for, and reports everything it does as events.
//
// It is built on pkg/ai and reads no environment variable and touches no
// file: what it does is what it was handed.
//
// # One turn at a time
//
//	a, err := agent.New(client, agent.WithSystem("You are terse."), agent.WithTools(read))
//
//	for e, err := range a.Run(ctx, ai.UserMessage("what is in config.json?")) {
//	    render(e)
//	}
//
// Run advances the conversation one turn and reports what it does on the way;
// the last event is TurnEnd, which carries how it went. Repeating it is the
// application's own loop:
//
//	for batch := range myMessages {
//	    for e, err := range a.Run(ctx, batch...) { render(e) }
//	}
//
// Events arrive on the ranging goroutine; a caller that must not slow the
// agent forwards them to a buffer of its own.
//
// # Vocabulary
//
// A *turn* is what one Run does: its input enters the conversation, then the
// loop takes steps until the model stops asking for tools. A *step* is one
// model call and the tools it asked for. An *inference* is one attempt at
// that call; a retried step holds several.
//
// # The conversation is ai.Message
//
// What the agent holds is what the model sees; there is no separate message
// type. WithMessageIDs gives every message an ID an application can point at;
// IDs are never sent.
//
// # Events
//
// Event lists them. Spans come in pairs, and only what takes time has one.
//
// The conversation is the fold of MessageAdded, which appends, and
// MessagesReplaced, which starts over.
//
// Compaction is opened by the hook doing it, through Compacting, and closed
// by the loop however the hook returns.
//
// Nothing is dropped except ToolUpdate: a tool's Report is dropped rather than
// stalling the tool, and ToolEnd carries the finished result either way.
//
// When an inference fails retryably, InferenceEnd carries the error and
// another InferenceStart follows with Attempt incremented. No MessageAdded
// comes between them, which tells a consumer the partial output it drew is
// void. A call an OnInferError hook recovered has the same shape, with any
// MessagesReplaced between the two.
//
// In a parallel batch, ToolEnd is emitted as each tool finishes, while the
// tool results enter the conversation in the order the model asked for them.
//
// # Files
//
//	agent.go   an agent: what it holds, how it is built, what you read and set
//	run.go     a turn: Run, and the loop behind it
//	event.go   what a turn reports
//	tool.go    a tool: defined from a Go type, offered, run
//	hook.go    the places a caller gets between the loop and what it is doing
package agent
