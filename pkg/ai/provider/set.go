package provider

import (
	"context"
	"slices"
	"strings"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Set is a set of providers, keyed by ID.
//
// The key is matched without regard to case, as the catalog matches a vendor
// ID: a reference like "DeepSeek/deepseek-v4-pro" is typed by a person.
type Set struct {
	mu sync.RWMutex
	m  map[string]*Provider
}

// NewSet builds a set from the given providers.
func NewSet(list ...*Provider) *Set {
	s := &Set{m: make(map[string]*Provider, len(list))}
	for _, e := range list {
		s.Set(e)
	}
	return s
}

// Set adds or replaces a provider by ID.
func (s *Set) Set(pr *Provider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = make(map[string]*Provider)
	}
	s.m[key(pr.ID())] = pr
}

// Delete removes a provider.
func (s *Set) Delete(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, key(id))
}

// Get returns one provider by ID.
func (s *Set) Get(id string) (*Provider, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.m[key(id)]
	return e, ok
}

// key normalizes a provider ID for lookup. The provider keeps its own
// spelling for display; only the map key is folded.
func key(id string) string { return strings.ToLower(strings.TrimSpace(id)) }

// All returns every provider, sorted by ID for a stable listing.
func (s *Set) All() []*Provider {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Provider, 0, len(s.m))
	for _, e := range s.m {
		out = append(out, e)
	}
	slices.SortFunc(out, func(a, b *Provider) int { return strings.Compare(a.ID(), b.ID()) })
	return out
}

// Models returns every provider's known models, in provider order. It never
// blocks and never fails.
func (s *Set) Models() []ai.Model {
	var out []ai.Model
	for _, e := range s.All() {
		out = append(out, e.Models()...)
	}
	return out
}

// Model resolves a "vendor/id" reference against the set. A model ID may
// itself contain a slash, so only the first segment names the provider.
func (s *Set) Model(ref string) (ai.Model, bool) {
	id, rest, ok := strings.Cut(ref, "/")
	if !ok {
		return ai.Model{}, false
	}
	e, found := s.Get(id)
	if !found {
		return ai.Model{}, false
	}
	return e.Model(rest)
}

// RefreshResult reports what one fan-out refresh managed.
type RefreshResult struct {
	// Errors holds the failure for each provider that could not be refreshed.
	// A provider absent from the map succeeded.
	Errors map[string]error
}

// OK reports whether every provider refreshed.
func (r RefreshResult) OK() bool { return len(r.Errors) == 0 }

// Refresh fetches every provider's listing concurrently.
func (s *Set) Refresh(ctx context.Context) RefreshResult {
	providers := s.All()
	errs := make([]error, len(providers))

	var wg sync.WaitGroup
	for i, e := range providers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = e.Refresh(ctx)
		}()
	}
	wg.Wait()

	result := RefreshResult{}
	for i, err := range errs {
		if err != nil {
			if result.Errors == nil {
				result.Errors = make(map[string]error)
			}
			result.Errors[providers[i].ID()] = err
		}
	}
	return result
}
