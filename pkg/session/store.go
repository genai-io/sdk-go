// Package session persists an agent's conversation and restores it.
package session

import (
	"context"
	"errors"

	"github.com/genai-io/sdk-go/pkg/ai"
)

var (
	// ErrNotFound is returned for a session that does not exist.
	ErrNotFound = errors.New("session: not found")
	// ErrClosed is reported by a Session after Close.
	ErrClosed = errors.New("session: closed")
)

// Store keeps conversations. Only a session's claim holder changes it.
type Store interface {
	// Create starts a session with an empty conversation and returns its id.
	Create(ctx context.Context) (string, error)
	// Claim makes the caller the session's only writer until release.
	Claim(ctx context.Context, id string) (release func() error, err error)
	// Append adds a message to the end of the conversation.
	Append(ctx context.Context, id string, msg ai.Message) error
	// Replace swaps the whole conversation for msgs — compaction, or a
	// history handed in from elsewhere.
	Replace(ctx context.Context, id string, msgs []ai.Message) error
	// Load returns the conversation, which is the caller's.
	Load(ctx context.Context, id string) ([]ai.Message, error)
}
