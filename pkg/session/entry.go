// Package session persists an agent's conversation and restores it.
package session

import (
	"errors"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

var (
	// ErrNotFound is returned for a session that does not exist.
	ErrNotFound = errors.New("session: not found")
	// ErrClosed is reported by a Session after Close.
	ErrClosed = errors.New("session: closed")
)

// EntryType says what an Entry does to the conversation.
type EntryType string

const (
	// EntryMessage appends Messages to the conversation.
	EntryMessage EntryType = "message"
	// EntryConversationReplaced replaces the conversation with Messages —
	// compaction, or a history handed in from elsewhere. A restore starts over
	// from the last one, because everything before it was thrown away.
	EntryConversationReplaced EntryType = "conversation.replaced"
)

// Entry is one durable change to the conversation. A log of them, in order,
// is the session.
type Entry struct {
	At       time.Time    `json:"at"`
	Type     EntryType    `json:"type"`
	Messages []ai.Message `json:"messages,omitempty"`
}
