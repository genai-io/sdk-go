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
// Run advances the conversation one exchange and reports what it does on the
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
// What an application does need on the conversation itself is a way to point
// at one turn — the row a person is editing, the entry its store already
// holds. WithMessageIDs names every message that enters, the loop's own
// answers and tool results included, and one that arrived already named keeps
// the name it came with. Nothing is sent: no protocol has a field for it, so
// naming a conversation cannot change what the model reads.
//
// # Events
//
// Twelve types, each one there because a consumer would break without it.
// Event lists them; what follows is what a list cannot say.
//
// A span always comes in pairs, and only what takes time has one. What entered
// the conversation is its own event because a user's message and a batch of
// tool results enter without any span at all — they were whole before the
// agent saw them.
//
// Compaction is the one span the loop cannot open by itself: shortening a
// conversation is a model call, but whether this boundary is about to make one
// is known only inside the hook deciding it. So the hook opens it, by calling
// Compacting, and the loop closes it however the hook returns. A hook that
// announces nothing produces no span, which is why the stream is not two
// events longer at every step.
//
// Compacting and Report are the same shape for the same reason: caller code
// that will be slow, telling the stream so through the context it was handed,
// and declaring nothing when it has nothing to say.
//
// The conversation is the fold of two events and no others: MessageAdded
// appends, and MessagesReplaced starts over, because everything announced
// before one of those is what the caller threw away. What SetMessages did is
// on the stream for exactly that reason: a consumer that folded only what was
// appended would hand back the history compaction just discarded.
//
// TurnEnd carries the summary: what the turn cost and why it stopped, the
// error included. It holds nothing a consumer could fold out of the stream
// itself.
//
// Nothing is dropped on the way out: the events arrive on the ranging
// goroutine, so there is no reader to fall behind. What a tool reports is the
// exception, because it crosses from the tool's own goroutine — progress is
// dropped rather than stalling a tool for it, and ToolEnd carries the finished
// result either way.
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
// A call an OnInferError hook recovered is the same shape, with the
// conversation it shortened announced between the two — so a consumer that
// already handles a retry handles this without a case of its own.
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
//	run.go     an exchange: Run, and the reason-and-act loop behind it
//
//	event.go   what an exchange reports
//	tool.go    a tool: defined from a Go type, offered, run
//	hook.go    the six places a caller gets between the loop and what it is
//	           doing, and the things each one is handed
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
// Compaction is the same division, and the reason Hook.PreStep exists: the loop
// knows when a conversation is about to outgrow its window and where it may
// safely be replaced, and what a shorter one should say — what to keep, and
// what to pay to find out — is not something this package could be right about.
//
// Hook.OnInferError is the other end of that: a window measured before a call
// is measured with an estimate, and the endpoint is the one that knows. When
// it answers that the prompt was too long, the loop has no answer of its own —
// replaying the same prompt fails the same way — so it asks, and shortening it
// there is the same six lines shortening it at a boundary was.
//
// Persisting what happened, and restoring it, is agent/session.
package agent
