package session_test

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/session"
)

// memStore keeps sessions in memory: the second implementation of
// session.Store, which is what keeps jsonl replaceable, and what these tests
// record into.
type memStore struct {
	mu       sync.Mutex
	sessions map[string]*held
	seq      int64 // names sessions the caller did not name
}

type held struct {
	meta    session.Meta
	entries []session.Entry
	claimed bool
}

var _ session.Store = (*memStore)(nil)

// Create starts a session, assigning an id when none was given.
func (s *memStore) Create(_ context.Context, meta session.Meta) (session.Meta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if meta.ID == "" {
		s.seq++
		meta.ID = fmt.Sprintf("mem-%d", s.seq)
	}
	if s.sessions == nil {
		s.sessions = map[string]*held{}
	}
	if _, exists := s.sessions[meta.ID]; exists {
		return session.Meta{}, fmt.Errorf("memory: session %s already exists", meta.ID)
	}

	now := time.Now().UTC()
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.UpdatedAt = now
	meta.Entries = 0
	s.sessions[meta.ID] = &held{meta: meta}
	return meta, nil
}

// Append writes entries in order, assigning Seq to any that lack one.
func (s *memStore) Append(_ context.Context, id string, entries ...session.Entry) error {
	if len(entries) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	h, ok := s.sessions[id]
	if !ok {
		return fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}

	now := time.Now().UTC()
	for _, e := range entries {
		h.meta.Entries++
		// Cloned on the way in: a caller that reuses the slice it handed over
		// must not be able to rewrite what was recorded.
		h.entries = append(h.entries, cloneEntry(e.Stamped(h.meta.Entries, now)))
	}
	h.meta.UpdatedAt = now
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

// cloneEntry copies the parts of an entry a caller could still be holding.
func cloneEntry(e session.Entry) session.Entry {
	if e.Message != nil {
		msg := *e.Message
		msg.Content = slices.Clone(e.Message.Content)
		e.Message = &msg
	}
	e.Messages = slices.Clone(e.Messages)
	if e.Inference != nil {
		inf := *e.Inference
		inf.Tools = slices.Clone(e.Inference.Tools)
		e.Inference = &inf
	}
	if e.ToolRun != nil {
		run := *e.ToolRun
		e.ToolRun = &run
	}
	if e.Outcome != nil {
		out := *e.Outcome
		e.Outcome = &out
	}
	return e
}

func (s *memStore) Claim(ctx context.Context, id string) (func() error, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	held, ok := s.sessions[id]
	if !ok {
		return nil, fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}
	if held.claimed {
		return nil, fmt.Errorf("memory: session %s already claimed", id)
	}
	held.claimed = true
	var once sync.Once
	return func() error { once.Do(func() { s.mu.Lock(); defer s.mu.Unlock(); held.claimed = false }); return nil }, nil
}
