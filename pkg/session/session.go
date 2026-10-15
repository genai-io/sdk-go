package session

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Session is one claimed session: its only writer until Close.
type Session struct {
	store   Store
	id      string
	release func() error

	mu sync.Mutex
	// resumed is the history Open returned, until the conversation moves on.
	// The agent announces it as replaced when it is handed it, and recording
	// that would write a copy of what was just read, once per resume.
	resumed []ai.Message
	err     error
	closed  bool
}

// Create starts a session and claims it.
func Create(ctx context.Context, store Store) (*Session, error) {
	id, err := store.Create(ctx)
	if err != nil {
		return nil, err
	}
	release, err := store.Claim(ctx, id)
	if err != nil {
		return nil, err
	}
	return &Session{store: store, id: id, release: release}, nil
}

// Open claims an existing session and restores its conversation.
func Open(ctx context.Context, store Store, id string) (*Session, []ai.Message, error) {
	release, err := store.Claim(ctx, id)
	if err != nil {
		return nil, nil, err
	}
	messages, err := restore(ctx, store, id)
	if err != nil {
		return nil, nil, errors.Join(err, release())
	}
	// A copy, so editing the returned history is a real replacement.
	resumed := slices.Clone(messages)
	for i := range resumed {
		resumed[i].Content = messages[i].Content.Clone()
	}
	return &Session{store: store, id: id, release: release, resumed: resumed}, messages, nil
}

// restore folds a session's entries into its conversation.
func restore(ctx context.Context, store Store, id string) ([]ai.Message, error) {
	var messages []ai.Message
	for entry, err := range store.Entries(ctx, id) {
		if err != nil {
			return nil, err
		}
		switch entry.Type {
		case EntryMessage:
			messages = append(messages, entry.Messages...)
		case EntryConversationReplaced:
			messages = entry.Messages
		default:
			return nil, fmt.Errorf("session: %s: unknown entry type %q", id, entry.Type)
		}
	}
	return messages, nil
}

// RecordEvent persists what an agent event does to the conversation and
// ignores the rest. Call it from your event loop, in order, before rendering.
//
// After the first failed write it does nothing, and Err says so: a
// conversation with a hole in it is not shorter but broken, so stopping
// leaves a prefix that still restores.
func (s *Session) RecordEvent(ctx context.Context, e agent.Event) {
	var entry Entry
	switch v := e.(type) {
	case agent.MessageAdded:
		entry = Entry{Type: EntryMessage, Messages: []ai.Message{v.Message}}
	case agent.MessagesReplaced:
		entry = Entry{Type: EntryConversationReplaced, Messages: v.Messages}
	default:
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil || s.closed {
		return
	}
	resumed := s.resumed
	s.resumed = nil
	if entry.Type == EntryConversationReplaced && resumed != nil && reflect.DeepEqual(resumed, entry.Messages) {
		return
	}
	entry.At = time.Now().UTC()
	s.err = s.store.Append(ctx, s.id, entry)
}

// Close gives up the claim.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.release()
}

// Err reports the write that stopped recording, or ErrClosed after Close.
func (s *Session) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil && s.closed {
		return ErrClosed
	}
	return s.err
}

// ID reports which session this records into.
func (s *Session) ID() string { return s.id }
