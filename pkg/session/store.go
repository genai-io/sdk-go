package session

import (
	"context"
	"iter"
)

// Store is what a Session appends to and a conversation is restored from.
type Store interface {
	// Create starts an empty session and returns its id.
	Create(ctx context.Context) (string, error)
	// Claim makes the caller the session's only writer until release.
	Claim(ctx context.Context, id string) (release func() error, err error)
	// Append adds one entry to the end of a claimed session.
	Append(ctx context.Context, id string, entry Entry) error
	// Entries reads from the beginning and stops at the first error. Each
	// entry it yields is the caller's.
	Entries(ctx context.Context, id string) iter.Seq2[Entry, error]
}
