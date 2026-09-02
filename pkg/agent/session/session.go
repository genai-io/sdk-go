// Package session persists what an agent did, and restores it.
//
// # Where things live
//
//	session.go   the store contract, what an entry is, and how a session opens
//	recorder.go  the handler that turns events into entries
//	jsonl        a store on the filesystem: a directory per session
package session

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrNotFound is returned for a session that does not exist.
var ErrNotFound = errors.New("session: not found")

// Meta is what a session is, apart from what happened in it. Small on purpose:
// it has to be listable without reading any session's entries.
type Meta struct {
	ID string `json:"id"`
	// Title and Model are the application's to set, through whatever its store
	// offers for that — jsonl has SetMeta. Nothing in this package writes
	// them: which model answered is on each Inference entry, where it can
	// differ from one turn to the next.
	Title     string    `json:"title,omitempty"`
	Model     string    `json:"model,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Entries   int64     `json:"entries"`

	// Parent and ForkedAt record where a forked session came from: the
	// session it branched off, and the entry it branched at.
	Parent   string `json:"parent,omitempty"`
	ForkedAt int64  `json:"forked_at,omitempty"`
}

// EntryType says which field of an Entry carries its payload.
type EntryType string

const (
	// EntryMessage is something that entered the conversation. Folding these
	// is what restores a session.
	EntryMessage EntryType = "message"
	// EntrySnapshot is the conversation after being replaced — compaction, or
	// a history handed in from elsewhere. A fold starts from the last one,
	// because everything before it was thrown away.
	EntrySnapshot EntryType = "snapshot"
	// EntryInference is one model call. Not needed to resume; needed to
	// explain and to bill.
	EntryInference EntryType = "inference"
	// EntryToolRun is one tool execution.
	EntryToolRun EntryType = "tool"
	// EntryOutcome closes a turn with how it ended.
	EntryOutcome EntryType = "outcome"
)

// Entry is one durable record. Seq orders it within its session and is
// assigned by the store, so two writers cannot invent the same position.
type Entry struct {
	Seq int64     `json:"seq"`
	At  time.Time `json:"at"`
	// Turn is the exchange this entry belongs to, numbered from the session's
	// beginning. Here rather than on each payload because it says where the
	// entry sits, which is what Seq and At say too.
	Turn int       `json:"turn"`
	Type EntryType `json:"type"`

	Message   *ai.Message  `json:"message,omitempty"`
	Snapshot  []ai.Message `json:"snapshot,omitempty"`
	Inference *Inference   `json:"inference,omitempty"`
	ToolRun   *ToolRun     `json:"tool,omitempty"`
	Outcome   *Outcome     `json:"outcome,omitempty"`
}

// payload reports whether the entry carries what its Type says. A wire format
// can be given anything, and folding a record of nothing silently is how a
// conversation comes back with a hole in it.
func (e Entry) payload() bool {
	switch e.Type {
	case EntryMessage:
		return e.Message != nil
	case EntrySnapshot:
		// An empty one is a conversation cleared, which is a state a session
		// has to be able to hold and come back from.
		return true
	case EntryInference:
		return e.Inference != nil
	case EntryToolRun:
		return e.ToolRun != nil
	case EntryOutcome:
		return e.Outcome != nil
	}
	return false
}

// Inference is one model call as it is kept.
type Inference struct {
	Attempt    int           `json:"attempt"`
	Model      string        `json:"model,omitempty"`
	System     string        `json:"system,omitempty"`
	Tools      []string      `json:"tools,omitempty"`
	Usage      ai.Usage      `json:"usage"`
	StopReason ai.StopReason `json:"stop_reason,omitempty"`
	Error      string        `json:"error,omitempty"`
}

// ToolRun is one tool execution as it is kept — not Tool, which is the thing
// that runs and which both ai and agent already have. Result.Details is
// dropped, being for an interface that is no longer running.
type ToolRun struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Args    string `json:"args,omitempty"`
	Content string `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

// Outcome is how a turn ended.
//
// StopReason is why this entry exists: not the model's reason, which is on
// each Inference, but the loop's — max_steps, terminated and canceled happen
// nowhere else. Without it an interrupted session just stops, saying nothing.
type Outcome struct {
	StopReason agent.StopReason `json:"stop_reason,omitempty"`
	Usage      ai.Usage         `json:"usage"`
	Err        string           `json:"err,omitempty"`
}

// Stamped is the entry as a store writes it: numbered, and timestamped if
// whoever handed it over did not do so already.
//
// This and the two below are the store contract's own arithmetic, kept here
// rather than in each store: a third implementation that got any of them subtly
// different would be a store this package's tests pass on and nothing else
// works with.
func (e Entry) Stamped(seq int64, at time.Time) Entry {
	if e.Seq == 0 {
		e.Seq = seq
	}
	if e.At.IsZero() {
		e.At = at
	}
	return e
}

// Created is the metadata a store starts a session with: what it was asked
// for, with the fields the store owns set.
func (m Meta) Created(at time.Time) Meta {
	if m.CreatedAt.IsZero() {
		m.CreatedAt = at
	}
	m.UpdatedAt = at
	m.Entries = 0
	return m
}

// ByRecency orders sessions the way a list of them is read: most recently
// updated first.
func ByRecency(a, b Meta) int { return b.UpdatedAt.Compare(a.UpdatedAt) }

// Store is what a recorder writes to and a session is restored from — nothing
// more. Listing, renaming, forking and deleting are the application's business
// with the store it chose, and belong to that store's own type: an interface
// only has to name what this package calls.
type Store interface {
	// Create starts a session. A blank Meta.ID means the store assigns one.
	Create(ctx context.Context, meta Meta) (Meta, error)

	// Append writes entries in order, assigning Seq to any that lack one.
	Append(ctx context.Context, id string, entries ...Entry) error

	// Entries reads a session from the beginning. The iterator ends on the
	// first error.
	Entries(ctx context.Context, id string) iter.Seq2[Entry, error]

	// Meta reads one session's metadata, or ErrNotFound.
	Meta(ctx context.Context, id string) (Meta, error)
}

// Messages folds a session's entries back into a conversation: messages
// append, and a snapshot starts it over, because what came before one of those
// is what the agent threw away.
func Messages(ctx context.Context, store Store, id string) ([]ai.Message, error) {
	msgs, _, err := fold(ctx, store, id)
	return msgs, err
}

// fold reads a session once and answers both questions asked of it: what the
// conversation is, and how many exchanges it has held. The second is not
// derivable from the first — a snapshot resets the conversation and not the
// history — and reading the entries twice to learn it would be reading them
// twice.
func fold(ctx context.Context, store Store, id string) ([]ai.Message, int, error) {
	var msgs []ai.Message
	var turns int
	for entry, err := range store.Entries(ctx, id) {
		if err != nil {
			return nil, 0, err
		}
		if !entry.payload() {
			return nil, 0, fmt.Errorf("session: %s entry %d says %q and carries nothing",
				id, entry.Seq, entry.Type)
		}
		switch entry.Type {
		case EntrySnapshot:
			msgs = slices.Clone(entry.Snapshot)
		case EntryMessage:
			msgs = append(msgs, *entry.Message)
		case EntryOutcome:
			turns++
		}
	}
	return msgs, turns, nil
}

// Open starts a session or resumes one, returning a Recorder to feed events to
// and the conversation folded back out of what was stored. Pass an empty id for
// a new session; the id it was given is on the Recorder.
//
//	rec, history, _ := session.Open(ctx, store, "")   // or an existing id
//	a.SetMessages(history)
//	for e, err := range a.Run(ctx, ai.UserMessage(line)) {
//	    rec.Handle(ctx, e)
//	    render(e)
//	}
func Open(ctx context.Context, store Store, id string) (*Recorder, []ai.Message, error) {
	if id == "" {
		meta, err := store.Create(ctx, Meta{})
		if err != nil {
			return nil, nil, err
		}
		return newRecorder(store, meta.ID), nil, nil
	}

	if _, err := store.Meta(ctx, id); err != nil {
		return nil, nil, err
	}
	msgs, turns, err := fold(ctx, store, id)
	if err != nil {
		return nil, nil, err
	}

	rec := newRecorder(store, id)
	// The agent numbers turns from one every time it runs, because what came
	// back from storage was someone else's counting. The session is the one
	// place that knows both numbers, so it is where they are reconciled —
	// without which a resumed session holds two exchanges both called turn 1.
	rec.turnsBefore = turns
	// And this is the history the agent is about to be handed. It will
	// announce it as replaced, correctly and unavoidably; recording it would
	// write a copy of what was just read, once per resume, forever.
	rec.restored = msgs
	return rec, msgs, nil
}
