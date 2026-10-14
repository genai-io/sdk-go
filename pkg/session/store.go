package session

import (
	"context"
	"iter"
)

// Store is what a Session appends to and a conversation is restored from.
type Store interface {
	// Claim obtains exclusive session ownership before restoration or writing.
	// The returned release must be idempotent.
	Claim(ctx context.Context, id string) (func() error, error)
	// Create starts a session. A blank Meta.ID means the store assigns one.
	Create(ctx context.Context, meta Meta) (Meta, error)
	// Append writes entries in order, assigning Seq to any that lack one.
	Append(ctx context.Context, id string, entries ...Entry) error
	// Entries reads from the beginning and stops at the first error.
	Entries(ctx context.Context, id string) iter.Seq2[Entry, error]
	// Meta reads one session's metadata, or ErrNotFound.
	Meta(ctx context.Context, id string) (Meta, error)
}
