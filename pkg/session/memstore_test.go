package session_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

// memStore keeps sessions in memory: the second implementation of
// session.Store, which is what keeps jsonl replaceable, and what these tests
// record into.
type memStore struct {
	mu       sync.Mutex
	sessions map[string]*held
}

type held struct {
	entries []session.Entry
	claimed bool
}

var _ session.Store = (*memStore)(nil)

func (s *memStore) Create(context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sessions == nil {
		s.sessions = map[string]*held{}
	}
	id := fmt.Sprintf("mem-%d", len(s.sessions)+1)
	s.sessions[id] = &held{}
	return id, nil
}

func (s *memStore) Claim(ctx context.Context, id string) (func() error, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	h, ok := s.sessions[id]
	if !ok {
		return nil, fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}
	if h.claimed {
		return nil, errors.New("memory: session " + id + " already claimed")
	}
	h.claimed = true
	return func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		h.claimed = false
		return nil
	}, nil
}

func (s *memStore) Append(_ context.Context, id string, e session.Entry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	h, ok := s.sessions[id]
	if !ok {
		return fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}
	h.entries = append(h.entries, cloneEntry(e))
	return nil
}

// Entries reads a session from the beginning. A cancelled context ends the
// read with its error, since stopping quietly looks like a shorter session.
func (s *memStore) Entries(ctx context.Context, id string) iter.Seq2[session.Entry, error] {
	s.mu.Lock()
	h, ok := s.sessions[id]
	var snapshot []session.Entry
	if ok {
		snapshot = slices.Clone(h.entries)
	}
	s.mu.Unlock()

	return func(yield func(session.Entry, error) bool) {
		if !ok {
			yield(session.Entry{}, fmt.Errorf("memory: %s: %w", id, session.ErrNotFound))
			return
		}
		for _, e := range snapshot {
			if err := ctx.Err(); err != nil {
				yield(session.Entry{}, err)
				return
			}
			if !yield(cloneEntry(e), nil) {
				return
			}
		}
	}
}

// cloneEntry copies what a caller could still be holding: a store keeps what
// was appended, not what the caller later did to it.
func cloneEntry(e session.Entry) session.Entry {
	e.Messages = slices.Clone(e.Messages)
	for i := range e.Messages {
		e.Messages[i].Content = e.Messages[i].Content.Clone()
	}
	return e
}

func msg(text string) session.Entry {
	return session.Entry{Type: session.EntryMessage, Messages: []ai.Message{ai.UserMessage(text)}}
}
