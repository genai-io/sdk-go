// Package session persists native agent messages and ordered events, and restores conversations.
package session

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrNotFound is returned for a session that does not exist.
var ErrNotFound = errors.New("session: not found")

// Meta is what a session is, apart from what happened in it. Small on purpose:
// stores can derive Entries and UpdatedAt from their authoritative log.
type Meta struct {
	ID        string    `json:"id"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Entries   int64     `json:"entries"`
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
	// EntryCustom is the application's own, stored without being read.
	EntryCustom EntryType = "custom"
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
	Custom    *Custom      `json:"custom,omitempty"`
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
	case EntryCustom:
		return e.Custom != nil
	}
	return false
}

// Custom is an application's own event — a permission answered, a hook fired —
// kept in this log rather than one of its own because the order across both is
// the fact worth keeping.
//
// Data is stored and handed back, never read: the value is the application's,
// and so is the question it answers. Namespace Kind ("permission.decided"), so
// a store holding more than one application's sessions stays legible.
//
// Written through [Writer.Record], which knows what turn to file it under.
type Custom struct {
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data,omitempty"`
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
// that runs and which both ai and agent already have.
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
// This and the two below are the store contract's own arithmetic, kept here so
// a third implementation cannot get it subtly different.
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
