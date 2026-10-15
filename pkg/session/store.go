// Package session saves an agent's conversation and loads it back.
//
//	history, err := store.Load(ctx, id)   // session.ErrNotFound for a new one
//	a.SetMessages(history)
//	for e, err := range a.Run(ctx, msg) { … }
//	err = store.Save(ctx, id, a.Messages())
//
// A save is the whole conversation, so compaction needs nothing of its own:
// the next save is the compacted conversation. Save as often as losing work
// would hurt — after each run, or on every agent.MessageAdded.
package session

import (
	"context"
	"errors"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ErrNotFound is returned by Load for a session that was never saved.
var ErrNotFound = errors.New("session: not found")

// Store keeps conversations by id.
type Store interface {
	// Save replaces the session's conversation with msgs, creating the
	// session if it does not exist.
	Save(ctx context.Context, id string, msgs []ai.Message) error
	// Load returns the session's conversation, which is the caller's.
	Load(ctx context.Context, id string) ([]ai.Message, error)
}
