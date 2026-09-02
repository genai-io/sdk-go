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

// MessageAdded says something entered the conversation, which is the fold of
// these and MessagesReplaced.
type MessageAdded struct {
	Turn    int
	Message ai.Message
}

func (MessageAdded) event() {}

// MessagesReplaced says SetMessages threw the conversation away, reported at
// the start of the next exchange because that is when the agent next has
// anywhere to say it. Everything announced before one is gone: a consumer that
// ignores it hands back the history compaction just discarded.
type MessagesReplaced struct {
	Turn     int
	Messages []ai.Message
}

func (MessagesReplaced) event() {}

// MessageStart says the model has been asked. Only a message the model
// produces has a span; the rest arrive whole.
type MessageStart struct {
	Turn int
	// Attempt rises on a retry — how a consumer knows the partial output it
	// drew is about to be replaced.
	Attempt int
	// Inference is the call going out, PreInfer's edits included. Read it, do
	// not write it: events are told, and PreInfer is where a call is edited.
	Inference *Inference
}

func (MessageStart) event() {}

// MessageUpdate is one streamed fragment, exactly as pkg/ai made it. A reader
// that falls behind loses fragments rather than holding the agent up —
// MessageAdded carries the whole thing anyway.
type MessageUpdate struct {
	Turn  int
	Delta ai.Event
}

func (MessageUpdate) event() {}

// Text is the fragment of the answer this update carries, empty when it
// carries anything else. Almost every consumer wants exactly this and nothing
// around it:
//
//	case agent.MessageUpdate:
//	    fmt.Print(v.Text())
func (u MessageUpdate) Text() string { return u.fragment(ai.BlockText) }

// Thinking is the fragment of reasoning this update carries, empty otherwise.
// A model that reasons out loud produces these before its answer.
func (u MessageUpdate) Thinking() string { return u.fragment(ai.BlockThinking) }

func (u MessageUpdate) fragment(kind ai.BlockType) string {
	if u.Delta.Type != ai.EventBlockDelta || u.Delta.Block.Type != kind {
		return ""
	}
	return u.Delta.Block.Text
}

// MessageEnd closes the span. A MessageAdded follows once the loop accepts the
// message, and does not follow at all when Err is set — that absence is what
// tells a consumer the partial output it drew was discarded.
type MessageEnd struct {
	Turn      int
	Attempt   int
	Inference *Inference
	Response  *ai.Response
	Err       error
}

func (MessageEnd) event() {}

// ToolStart opens one tool execution, before the gate runs, so Args is what
// the model sent rather than what a hook rewrote.
type ToolStart struct {
	Turn int
	ID   string
	Name string
	Args string
}

func (ToolStart) event() {}

// ToolUpdate is a partial result from a tool that reports as it works. Unlike
// a message fragment it replaces rather than appends, and ToolEnd carries the
// finished result.
type ToolUpdate struct {
	Turn    int
	ID      string
	Name    string
	Partial Result
}

func (ToolUpdate) event() {}

// ToolEnd closes one tool execution, emitted before PostTool so a reader is
// not waiting on hooks: a hook that replaces the result changes what the model
// is told, not this.
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
	// none. Reading it off the conversation is wrong: that ends in tool
	// results when a turn stopped on terminated or max_steps.
	Message    ai.Message
	Usage      ai.Usage
	StopReason StopReason
	Err        error
}

func (TurnEnd) event() {}
