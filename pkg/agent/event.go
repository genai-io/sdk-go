package agent

import "github.com/genai-io/sdk-go/pkg/ai"

// Event is one thing that happened.
//
// The set is closed: event() is unexported, so no package but this one can add
// to it. Each type below declares itself an Event with an empty event method,
// which is never called and exists only to be that gate — a consumer switching
// over Event knows this list is all there is.
//
//	MessageAdded  MessagesReplaced            the conversation changing
//	MessageStart  MessageUpdate  MessageEnd   the model producing a message
//	ToolStart     ToolUpdate     ToolEnd      a tool call, asked to answered
//	TurnStart                    TurnEnd      the exchange around them
//
// Every event that belongs to a turn carries its number, and every event that
// belongs to a span carries what opened it. A consumer reads what happened off
// the event; it never has to rebuild it by remembering what came before.
type Event interface{ event() }

// MessageAdded says something entered the conversation — the only event that
// changes what the model sees next. The conversation is the fold of these.
type MessageAdded struct {
	Turn    int
	Message ai.Message
}

func (MessageAdded) event() {}

// MessagesReplaced says the conversation was thrown away and another put in
// its place — what SetMessages did, reported at the start of the next
// exchange, because that is when the agent next has anywhere to report it.
//
// It is the other event that changes what the model sees, and the reason a
// fold over MessageAdded alone is not the conversation: everything announced
// before one of these is gone. Compaction is the case that matters — a
// consumer that ignores this hands back the history the agent just discarded.
type MessagesReplaced struct {
	Turn     int
	Messages []ai.Message
}

func (MessagesReplaced) event() {}

// MessageStart says the model has been asked and a message is on its way.
// Only a message the model produces has a span; the rest arrive whole.
type MessageStart struct {
	Turn int
	// Attempt rises on a retry — how a consumer knows the partial output it
	// drew is about to be replaced.
	Attempt int
	// Inference is the call going out, PreInfer's edits included. The client
	// merges its defaults and repairs the history after this, so it is what
	// was asked for, not the finished wire request.
	Inference *Inference
}

func (MessageStart) event() {}

// MessageUpdate is one streamed fragment, exactly as pkg/ai made it. Fragments
// append. A reader that falls behind loses them rather than holding the agent
// up — MessageAdded carries the whole thing, so they converge anyway.
type MessageUpdate struct{ Delta ai.Event }

func (MessageUpdate) event() {}

// MessageEnd closes the span with what the call produced, or why it did not.
// The message follows as a MessageAdded once the loop accepts it, and does not
// follow at all when Err is set — that absence is what tells a consumer the
// partial output it drew was discarded.
type MessageEnd struct {
	Turn    int
	Attempt int
	// Inference is the one MessageStart opened this span with, carried again
	// so that closing a span needs nothing but the event that closes it.
	Inference *Inference
	Response  *ai.Response
	Err       error
}

func (MessageEnd) event() {}

// ToolStart opens one tool execution, before the gate runs. Args is what the
// model sent, before any hook rewrote it.
type ToolStart struct {
	Turn int
	ID   string
	Name string
	Args string
}

func (ToolStart) event() {}

// ToolUpdate is a partial result from a tool that reports as it works: the
// output of a command as it arrives, a file list as it is walked. Unlike a
// message fragment it replaces rather than appends — a tool sends what it has,
// not what changed — and ToolEnd carries the finished result.
type ToolUpdate struct {
	Turn    int
	ID      string
	Name    string
	Partial Result
}

func (ToolUpdate) event() {}

// ToolEnd closes one tool execution with what the tool produced, emitted
// before PostTool runs so a reader is not waiting on hooks — a hook that
// replaces the result changes what the model is told, not this.
type ToolEnd struct {
	Turn int
	ID   string
	// Name and Args are the ones ToolStart opened with, carried again for the
	// same reason MessageEnd carries its request.
	Name   string
	Args   string
	Result Result
	Err    error
}

func (ToolEnd) event() {}

// TurnStart and TurnEnd bracket one exchange: the user said something, and the
// loop runs until the model stops asking for tools. A turn holds as many model
// calls as the tools require.
type TurnStart struct{ Turn int }

func (TurnStart) event() {}

// TurnEnd closes one exchange, and carries only what a consumer could not work
// out for itself: StopReason, which is a decision the loop made and appears
// nowhere else, and Usage, which everyone wants and has one right way to add
// up. A failed turn still reports both.
type TurnEnd struct {
	Turn int
	// Message is the last message the model produced this turn, zero if it
	// produced none. It is here because the obvious way to find it is wrong:
	// the conversation's last message is a batch of tool results when the turn
	// stopped on terminated or max_steps, and only the loop knows which one
	// was the model's.
	Message    ai.Message
	Usage      ai.Usage
	StopReason StopReason
	Err        error
}

func (TurnEnd) event() {}
