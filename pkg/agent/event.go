package agent

import "github.com/genai-io/sdk-go/pkg/ai"

// Event is one thing that happened. The set is closed:
//
//	MessageAdded     MessagesReplaced               the conversation changing
//	InferenceStart   InferenceUpdate  InferenceEnd  one model call
//	ToolStart        ToolUpdate       ToolEnd       one tool call
//	CompactionStart                   CompactionEnd the conversation being shortened
//	TurnStart                         TurnEnd       the turn around them
//
// Every event carries its turn, and every closing event carries what opened
// it.
type Event interface{ event() }

// MessageAdded says a message entered the conversation. Folding MessageAdded
// and MessagesReplaced reproduces Messages.
type MessageAdded struct {
	Turn    int
	Message ai.Message
}

func (MessageAdded) event() {}

// CompactionStart says a PreStep or OnInferError hook has begun shortening
// the conversation. The hook opens it by calling Compacting; the loop closes
// it.
type CompactionStart struct {
	Turn int
	// Messages is the conversation being shortened and Tokens what it was
	// priced at, so a consumer can say what is going on and, once
	// MessagesReplaced arrives, by how much it helped.
	Messages []ai.Message
	Tokens   int
}

func (CompactionStart) event() {}

// CompactionEnd closes the span however the hook returned. Err is the hook's
// error; a conversation left alone has no error and no MessagesReplaced.
type CompactionEnd struct {
	Turn     int
	Messages []ai.Message
	Tokens   int
	Err      error
}

func (CompactionEnd) event() {}

// MessagesReplaced says the conversation was replaced, by SetMessages or a
// hook, and is now Messages. Everything announced before it is gone.
type MessagesReplaced struct {
	Turn     int
	Messages []ai.Message
}

func (MessagesReplaced) event() {}

// InferenceStart says one model call is going out.
type InferenceStart struct {
	Turn int
	// Attempt rises on a retry — how a consumer knows the partial output it
	// drew is about to be replaced.
	Attempt int
	// Inference is the call going out, PreInfer's edits included. Read it, do
	// not write it: events are told, and PreInfer is where a call is edited.
	Inference *Inference
}

func (InferenceStart) event() {}

// InferenceUpdate is one streamed fragment, exactly as pkg/ai made it.
type InferenceUpdate struct {
	Turn  int
	Delta ai.Event
}

func (InferenceUpdate) event() {}

// Text is the answer text this fragment carries, empty for any other kind.
//
//	case agent.InferenceUpdate:
//	    fmt.Print(v.Text())
func (u InferenceUpdate) Text() string {
	if u.Delta.Type != ai.EventBlockDelta || u.Delta.Block.Type != ai.BlockText {
		return ""
	}
	return u.Delta.Block.Text
}

// InferenceEnd closes the call. MessageAdded follows with the answer unless
// Err is set; then the partial output was discarded, and an InferenceStart
// with a higher Attempt may follow.
type InferenceEnd struct {
	Turn      int
	Attempt   int
	Inference *Inference
	Response  *ai.Response
	Err       error
}

func (InferenceEnd) event() {}

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

// TurnStart and TurnEnd bracket one turn, which holds one or more inferences.
type TurnStart struct{ Turn int }

func (TurnStart) event() {}

// TurnEnd closes one turn. A failed turn still reports its usage and why it
// stopped.
type TurnEnd struct {
	Turn int
	// Message is the last message the model produced, zero if it produced
	// none. Reading it off the conversation is wrong: that ends in tool
	// results when a turn stopped on terminated or max_steps.
	Message ai.Message
	// Usage is the tokens the turn spent, summed. Money is not summable
	// the same way: a turn routed to a second model spent them under two rate
	// cards, so fold a cost from each InferenceEnd instead.
	Usage      ai.Usage
	StopReason StopReason
	Err        error
}

func (TurnEnd) event() {}
