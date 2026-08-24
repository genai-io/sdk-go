// Package agent runs a conversation: it calls a model, executes the tools the
// model asks for, and reports everything it does as events.
//
// It is built on pkg/ai and adds what a single call does not have — a
// conversation that grows, a toolset the model can reach, and hooks at the
// points where an application needs to interpose. It reads no environment
// variable and touches no file: what it does is what it was handed.
//
// # A loop and two channels
//
//	a, err := agent.New(client, agent.WithSystem("You are terse."), agent.WithTools(read))
//
//	go a.Run(ctx)
//
//	a.In() <- ai.UserMessage("what is in config.json?")
//	for e := range a.Out() { … }
//
// In takes messages, Out reports what the agent does, and that is the whole
// interface. Close In to say there is no more work; the agent closes Out when
// it stops, so ranging over it ends by itself.
//
// A message sent while the agent is working joins the exchange in flight —
// after the tools running now finish, before the model is asked again. There
// is no second way in for "wait until this one is done", because someone who
// wants to wait can wait.
//
// An agent runs once and holds one conversation. Many exchanges belong to one
// run, fed through In; that is what the loop is for.
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
//	ToolStart     ToolUpdate     ToolEnd      a tool call, asked to answered
//	RunStart                     RunEnd       the agent's working life
//	TurnStart                    TurnEnd      one exchange within it
//
// A span always comes in pairs, and only what takes time has one. What entered
// the conversation is its own event because a user's message and a batch of
// tool results enter without any span at all — they were whole before the
// agent saw them.
//
// The conversation is the fold of MessageAdded, and that is the only event
// that changes it. Everything else reports work in progress.
//
// TurnEnd carries the summary: what the turn cost and why it stopped. It holds
// nothing a consumer could fold out of the stream itself.
//
// Only MessageUpdate and ToolUpdate are dropped when a consumer falls behind,
// because the event that closes each span carries the complete value.
// Everything else waits.
//
// Three words are used precisely, because pi and San disagree about them. A
// *turn* is an exchange: someone said something, and the loop runs until the
// model stops asking for tools. An *inference* is one model call, of which a
// turn may hold several — it has no event of its own, because a model call is
// how one message gets made. A *run* is the loop's working life across many
// turns.
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
// The three loops nest — a run holds turns, a turn holds inferences — and the
// files are named for the same three, so a reader who knows the vocabulary
// knows the layout:
//
//	agent.go   an agent: what it holds, how it is built, what you read and set
//	run.go     a run: many turns, fed by a channel, ended by closing it
//	turn.go    a turn: infer, act, repeat — and how the tools are run
//
//	event.go   what a run reports, and which reports may be dropped
//	tool.go    a tool: defined from a Go type, offered, run
//	hook.go    the three places a caller gets between the loop and the model
//
// # What is deliberately not here
//
// The system prompt is a string, the toolset is a []Tool, and the input queue
// is the caller's channel. Composing a prompt from sections, holding a
// registry other subsystems mutate, deciding what to do when a queue is full —
// each is something an application does, and each would have forced this
// package to invent an answer that only fits one application. Keeping them out
// is what stops it from growing a second, worse copy of San.
//
// Persisting what happened, and restoring it, is agent/session.
package agent
