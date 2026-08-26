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
	"iter"
	"slices"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrNotFound is returned for a session that does not exist.
var ErrNotFound = errors.New("session: not found")

// Meta is what a session is, apart from what happened in it. It is small on
// purpose: it has to be listable without reading any session's entries.
type Meta struct {
	ID        string    `json:"id"`
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
	// EntrySnapshot is the conversation as it stood after being replaced —
	// compaction, or a history handed in from elsewhere. A fold starts from
	// the last one of these, because everything before it was thrown away.
	EntrySnapshot EntryType = "snapshot"
	// EntryInference is one model call: what was asked, what it cost, how it
	// ended. Not needed to resume; needed to explain and to bill.
	EntryInference EntryType = "inference"
	// EntryTool is one tool execution.
	EntryTool EntryType = "tool"
	// EntryTurn closes one exchange with its outcome.
	EntryTurn EntryType = "turn"
)

// Entry is one durable record. Seq orders it within its session and is
// assigned by the store, so two writers cannot invent the same position.
type Entry struct {
	Seq  int64     `json:"seq"`
	At   time.Time `json:"at"`
	Type EntryType `json:"type"`

	Message   *ai.Message  `json:"message,omitempty"`
	Snapshot  []ai.Message `json:"snapshot,omitempty"`
	Inference *Inference   `json:"inference,omitempty"`
	Tool      *Tool        `json:"tool,omitempty"`
	Turn      *Turn        `json:"turn,omitempty"`
}

// Inference is one model call as it is kept.
type Inference struct {
	Turn       int           `json:"turn"`
	Attempt    int           `json:"attempt"`
	Model      string        `json:"model,omitempty"`
	System     string        `json:"system,omitempty"`
	Tools      []string      `json:"tools,omitempty"`
	Usage      ai.Usage      `json:"usage"`
	StopReason ai.StopReason `json:"stop_reason,omitempty"`
	Error      string        `json:"error,omitempty"`
}

// Tool is one tool execution as it is kept. Details is dropped: it is for an
// interface that is no longer running.
type Tool struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Args    string `json:"args,omitempty"`
	Content string `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

// Turn is one exchange as it is kept. How many inferences it took and how many
// tools ran are not fields: the Inference and Tool entries of this same session
// are where those are counted from.
type Turn struct {
	Turn       int              `json:"turn"`
	StopReason agent.StopReason `json:"stop_reason,omitempty"`
	Usage      ai.Usage         `json:"usage"`
	Err        string           `json:"err,omitempty"`
}

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
	var msgs []ai.Message
	for entry, err := range store.Entries(ctx, id) {
		if err != nil {
			return nil, err
		}
		switch {
		case entry.Type == EntrySnapshot:
			msgs = slices.Clone(entry.Snapshot)
		case entry.Type == EntryMessage && entry.Message != nil:
			msgs = append(msgs, *entry.Message)
		}
	}
	return msgs, nil
}

// Recorder turns the events an agent reports into stored entries. It is a
// plain function you call from your own event loop:
//
//	rec, _ := session.Open(ctx, store, "")          // or an existing id
//	for e := range out {
//	    rec.Handle(e)
//	    render(e)
//	}
func Open(ctx context.Context, store Store, id string) (*Recorder, []ai.Message, error) {
	if id == "" {
		meta, err := store.Create(ctx, Meta{})
		if err != nil {
			return nil, nil, err
		}
		return NewRecorder(store, meta.ID), nil, nil
	}

	if _, err := store.Meta(ctx, id); err != nil {
		return nil, nil, err
	}
	msgs, err := Messages(ctx, store, id)
	if err != nil {
		return nil, nil, err
	}
	return NewRecorder(store, id), msgs, nil
}
