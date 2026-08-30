// Package memory keeps sessions in the process that made them.
//
// It is the second implementation of session.Store, which is what makes the
// first one replaceable: an interface with one implementation has never been
// asked whether it describes anything but that implementation. It is also what
// the session package's own tests record into, so they exercise the contract
// rather than the filesystem.
//
// Nothing here survives the process. For a session that should, see the jsonl
// store beside it.
package memory

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent/session"
)

// Store holds sessions in memory. The zero value is ready to use.
type Store struct {
	mu       sync.Mutex
	sessions map[string]*held
	seq      int64 // names sessions the caller did not name
}

type held struct {
	meta    session.Meta
	entries []session.Entry
}

var _ session.Store = (*Store)(nil)

// Open returns an empty store, mirroring jsonl.Open so the two are swapped by
// changing one line.
func Open() *Store { return &Store{} }

// Create starts a session, assigning an id when none was given.
func (s *Store) Create(_ context.Context, meta session.Meta) (session.Meta, error) {
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
func (s *Store) Append(_ context.Context, id string, entries ...session.Entry) error {
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
		if e.Seq == 0 {
			e.Seq = h.meta.Entries
		}
		if e.At.IsZero() {
			e.At = now
		}
		// Cloned on the way in: a caller that reuses the slice it handed over
		// must not be able to rewrite what was recorded.
		h.entries = append(h.entries, cloneEntry(e))
	}
	h.meta.UpdatedAt = now
	return nil
}

// Entries reads a session from the beginning.
func (s *Store) Entries(_ context.Context, id string) iter.Seq2[session.Entry, error] {
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
			if !yield(cloneEntry(e), nil) {
				return
			}
		}
	}
}

// Meta reads one session's metadata.
func (s *Store) Meta(_ context.Context, id string) (session.Meta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	h, ok := s.sessions[id]
	if !ok {
		return session.Meta{}, fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}
	return h.meta, nil
}

// List returns every session, most recently updated first.
func (s *Store) List(_ context.Context) ([]session.Meta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]session.Meta, 0, len(s.sessions))
	for _, h := range s.sessions {
		out = append(out, h.meta)
	}
	slices.SortFunc(out, func(a, b session.Meta) int { return b.UpdatedAt.Compare(a.UpdatedAt) })
	return out, nil
}

// Delete removes a session and everything in it.
func (s *Store) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
	return nil
}

// cloneEntry copies the parts of an entry a caller could still be holding.
func cloneEntry(e session.Entry) session.Entry {
	if e.Message != nil {
		msg := *e.Message
		msg.Content = slices.Clone(e.Message.Content)
		e.Message = &msg
	}
	e.Snapshot = slices.Clone(e.Snapshot)
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
