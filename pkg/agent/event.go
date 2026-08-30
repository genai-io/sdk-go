package agent

import "github.com/genai-io/sdk-go/pkg/ai"

// Event is one thing that happened.
//
// The set is closed — event() is unexported — so a consumer switching over it
// knows this list is all there is:
//
//	MessageAdded  MessagesReplaced            the conversation changing
//	MessageStart  MessageUpdate  MessageEnd   the model producing a message
//	ToolStart     ToolUpdate     ToolEnd      a tool call, asked to answered
//	TurnStart                    TurnEnd      the exchange around them
//
// Every event carries its turn, and every closing event carries what opened
// it, so reading one is reading rather than remembering.
type Event interface{ event() }

// MessageAdded says something entered the conversation. The conversation is
// the fold of these and MessagesReplaced.
type MessageAdded struct {
	Turn    int
	Message ai.Message
}

func (MessageAdded) event() {}

// MessagesReplaced says SetMessages threw the conversation away and put
// another in its place, reported at the start of the next exchange because
// that is when the agent next has anywhere to report it. Everything announced
// before one is gone: a consumer that ignores it hands back the history
// compaction just discarded.
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
	// Inference is the call going out, PreInfer's edits included; the client
	// merges its defaults after this. Read it, do not write it: hooks are
	// asked and events are told, and PreInfer is where a call is edited.
	Inference *Inference
}

func (MessageStart) event() {}

// MessageUpdate is one streamed fragment, exactly as pkg/ai made it. Fragments
// append, and a reader that falls behind loses them rather than holding the
// agent up — MessageAdded carries the whole thing anyway.
type MessageUpdate struct{ Delta ai.Event }

func (MessageUpdate) event() {}

// MessageEnd closes the span with what the call produced, or why it did not.
// The message follows as a MessageAdded once the loop accepts it, and does not
// follow at all when Err is set — that absence is what tells a consumer the
// partial output it drew was discarded.
type MessageEnd struct {
	Turn      int
	Attempt   int
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

// ToolUpdate is a partial result from a tool that reports as it works. Unlike
// a message fragment it replaces rather than appends — a tool sends what it
// has, not what changed — and ToolEnd carries the finished result.
type ToolUpdate struct {
	Turn    int
	ID      string
	Name    string
	Partial Result
}

func (ToolUpdate) event() {}

// ToolEnd closes one tool execution, emitted before PostTool runs so a reader
// is not waiting on hooks: a hook that replaces the result changes what the
// model is told, not this.
type ToolEnd struct {
	Turn   int
	ID     string
	Name   string
	Args   string
	Result Result
	Err    error
}

func (ToolEnd) event() {}

// TurnStart and TurnEnd bracket one exchange, which holds as many model calls
// as the tools require.
type TurnStart struct{ Turn int }

func (TurnStart) event() {}

// TurnEnd closes one exchange. A failed turn still reports its usage and why
// it stopped.
type TurnEnd struct {
	Turn int
	// Message is the last message the model produced, zero if it produced
	// none. The obvious way to find it is wrong: the conversation ends in tool
	// results when a turn stopped on terminated or max_steps, and only the
	// loop knows which message was the model's.
	Message    ai.Message
	Usage      ai.Usage
	StopReason StopReason
	Err        error
}

func (TurnEnd) event() {}
