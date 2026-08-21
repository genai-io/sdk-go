package endpoint

import (
	"context"
	"slices"
	"strings"
	"sync"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Set is a set of endpoints, keyed by ID.
//
// It is a plain collection: reads are synchronous, Refresh fans out and
// reports per-endpoint failures rather than returning one error, because a
// single dead endpoint must not empty the list.
type Set struct {
	mu sync.RWMutex
	m  map[string]*Endpoint
}

// NewSet builds a set from the given endpoints.
func NewSet(list ...*Endpoint) *Set {
	ps := &Set{m: make(map[string]*Endpoint, len(list))}
	for _, e := range list {
		ps.Set(e)
	}
	return ps
}

// Set adds or replaces an endpoint by ID.
func (ps *Set) Set(e *Endpoint) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.m == nil {
		ps.m = make(map[string]*Endpoint)
	}
	ps.m[e.ID()] = e
}

// Delete removes an endpoint.
func (ps *Set) Delete(id string) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	delete(ps.m, id)
}

// Get returns one provider.
func (ps *Set) Get(id string) (*Endpoint, bool) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	e, ok := ps.m[id]
	return e, ok
}

// All returns every endpoint, sorted by ID for a stable listing.
func (ps *Set) All() []*Endpoint {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	out := make([]*Endpoint, 0, len(ps.m))
	for _, e := range ps.m {
		out = append(out, e)
	}
	slices.SortFunc(out, func(a, b *Endpoint) int { return strings.Compare(a.ID(), b.ID()) })
	return out
}

// Models returns every endpoint's known models, in endpoint order. It never
// blocks and never fails.
func (ps *Set) Models() []ai.Model {
	var out []ai.Model
	for _, e := range ps.All() {
		out = append(out, e.Models()...)
	}
	return out
}

// Model resolves a "vendor/id" reference against the set. A model ID may
// itself contain a slash, so only the first segment names the endpoint.
func (ps *Set) Model(ref string) (ai.Model, bool) {
	id, rest, ok := strings.Cut(ref, "/")
	if !ok {
		return ai.Model{}, false
	}
	e, found := ps.Get(id)
	if !found {
		return ai.Model{}, false
	}
	return e.Model(rest)
}

// RefreshResult reports what one fan-out refresh managed.
type RefreshResult struct {
	// Errors holds the failure for each endpoint that could not be refreshed.
	// A endpoint absent from the map succeeded.
	Errors map[string]error
}

// OK reports whether every endpoint refreshed.
func (r RefreshResult) OK() bool { return len(r.Errors) == 0 }

// Refresh fetches every endpoint's listing concurrently.
//
// It does not return an error: one endpoint being down must not fail the whole
// call, and each endpoint keeps whatever it last knew. Read RefreshResult to
// see which ones failed.
func (ps *Set) Refresh(ctx context.Context) RefreshResult {
	endpoints := ps.All()
	errs := make([]error, len(endpoints))

	var wg sync.WaitGroup
	for i, e := range endpoints {
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
			result.Errors[endpoints[i].ID()] = err
		}
	}
	return result
}
