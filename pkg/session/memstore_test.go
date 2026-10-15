package session_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

// memStore keeps conversations in memory: the second implementation of
// session.Store, which is what keeps jsonl replaceable, and what these tests
// record into. ops is what was asked of it, for tests about what a session
// writes rather than what it restores.
type memStore struct {
	mu       sync.Mutex
	sessions map[string]*held
}

type held struct {
	messages []ai.Message
	ops      []string
	claimed  bool
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
	h, err := s.held(id)
	if err != nil {
		return nil, err
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

func (s *memStore) Append(_ context.Context, id string, msg ai.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	h, err := s.held(id)
	if err != nil {
		return err
	}
	h.messages = append(h.messages, clone([]ai.Message{msg})...)
	h.ops = append(h.ops, "append")
	return nil
}

func (s *memStore) Replace(_ context.Context, id string, msgs []ai.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	h, err := s.held(id)
	if err != nil {
		return err
	}
	h.messages = clone(msgs)
	h.ops = append(h.ops, "replace")
	return nil
}

func (s *memStore) Load(ctx context.Context, id string) ([]ai.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	h, err := s.held(id)
	if err != nil {
		return nil, err
	}
	return clone(h.messages), nil
}

func (s *memStore) held(id string) (*held, error) {
	h, ok := s.sessions[id]
	if !ok {
		return nil, fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}
	return h, nil
}

// opsOf is what a session asked of the store, in order.
func (s *memStore) opsOf(id string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.sessions[id].ops)
}

// clone copies what a caller could still be holding: a store keeps what it
// was given, not what the caller later did to it.
func clone(msgs []ai.Message) []ai.Message {
	out := slices.Clone(msgs)
	for i := range out {
		out[i].Content = out[i].Content.Clone()
	}
	return out
}
