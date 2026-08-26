// Package agent runs a conversation: it calls a model, executes the tools the
// model asks for, and reports everything it does as events.
//
// It is built on pkg/ai and adds what a single call does not have — a
// conversation that grows, a toolset the model can reach, and hooks at the
// points where an application needs to interpose. It reads no environment
// variable and touches no file: what it does is what it was handed.
//
// # One exchange at a time
//
//	a, err := agent.New(client, agent.WithSystem("You are terse."), agent.WithTools(read))
//
//	for e, err := range a.Run(ctx, ai.UserMessage("what is in config.json?")) {
//	    render(e)
//	}
//
// Turn advances the conversation one exchange and reports what it does on the
// way; the last event is TurnEnd, which carries how it went.
//
// Repeating it is a for loop, and the loop is the application's — how messages
// are batched into exchanges, what a failure means, when to stop:
//
//	for batch := range myMessages {
//	    for e, err := range a.Run(ctx, batch...) { render(e) }
//	}
//
// A CLI reads stdin, an interface reads keys, a server reads requests, and
// none of those is a shape this package can guess.
//
// Events arrive on the ranging goroutine, so an agent that must run ahead of a
// slow reader is one whose caller forwards to a buffer of its own — how deep,
// and what to drop when it fills, being theirs to decide.
//
// # The conversation is ai.Message
//
// There is no separate agent message type, and so no conversion step. What the
// agent holds is what the model sees. An application with messages the model
// must not see — a notice, a status line — keeps those in its own interface
// state, which is where they belong: the conversation is not a transcript of
// the interface.
//
// # Events
//
// Eleven types, and each exists because a consumer would break without it.
//
//	MessageAdded                              a message entered the conversation
//	MessageStart  MessageUpdate  MessageEnd   the model producing one
//	ToolStart                    ToolEnd      a tool call, asked to answered
//	TurnStart                    TurnEnd      the exchange around them
//
// A span always comes in pairs, and only what takes time has one. What entered
// the conversation is its own event because a user's message and a batch of
// tool results enter without any span at all — they were whole before the
// agent saw them.
//
// The conversation is the fold of MessageAdded, started over by each
// MessagesReplaced — compaction and restore swap it whole, and a fold that only
// appended would hold what the agent threw away. Everything else reports work
// in progress.
//
// TurnEnd carries the summary: what the turn cost and why it stopped. It holds
// nothing a consumer could fold out of the stream itself.
//
// Nothing is dropped: the events arrive on the ranging goroutine, so there is
// no reader to fall behind.
//
// Three words are used precisely, because agent frameworks disagree about
// them. A *turn* is an exchange: someone said something, and the loop runs
// until the model stops asking for tools. An *inference* is one model call, of
// which a turn may hold several — it has no event of its own, because a model
// call is how one message gets made. A *run* is the loop's working life across
// many turns.
//
// # A retry needs no event of its own
//
// When a stream fails retryably, the attempt ends with a MessageEnd carrying
// the error and another MessageStart follows with Attempt incremented. No
// MessageAdded comes between them, and that absence is what tells a consumer
// the partial output it drew is void.
//
// # Two orders in a parallel batch
//
// ToolEnd is emitted as each tool finishes, so an interface can retire that
// spinner when it stops. The tool results appended to the conversation go in
// the order the model asked for them, so replaying a session produces the same
// transcript every time. Both matter and neither gives way.
//
// # Where things live
//
// A turn holds inferences, and the files are named for what they are:
//
//	agent.go   an agent: what it holds, how it is built, what you read and set
//	run.go  an exchange, from the outside: Turn, Interrupt
//	turn.go    a turn: reason, act, repeat — and how the tools are run
//
//	event.go   what an exchange reports
//	tool.go    a tool: defined from a Go type, offered, run
//	hook.go    the four places a caller gets between the loop and the model
//
// # What is deliberately not here
//
// The system prompt is a string, the toolset is a []Tool, and there is no
// queue in front of the agent at all. Composing a prompt from sections,
// holding a registry other subsystems mutate, deciding how deep a backlog may
// get and what to drop when it fills — each is something an application does,
// and each would have forced this package to invent an answer that fits one
// application. Keeping them out is what stops it from growing a second, worse
// copy of its caller.
//
// Persisting what happened, and restoring it, is agent/session.
package agent
