package auth

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"sync"
	"time"
)

// transport injects a vendor's current token into every request.
type transport struct {
	base   http.RoundTripper
	source *tokenSource
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := t.source.token(req.Context())
	if err != nil {
		return nil, err
	}
	// Cloning: a RoundTripper must not modify the request it is given.
	clone := req.Clone(req.Context())
	clone.Header.Set("Authorization", "Bearer "+token)
	return t.base.RoundTrip(clone)
}

// HTTPClient returns a client that presents a vendor's stored credential,
// renewing it as it expires.
func HTTPClient(vendorID string, c Credential, store Store, base *http.Client) (*http.Client, error) {
	f, ok := flowFor(vendorID)
	if !ok {
		return nil, fmt.Errorf("auth: %s does not sign in interactively", vendorID)
	}
	if base == nil {
		base = &http.Client{Timeout: 10 * time.Minute}
	}
	inner := base.Transport
	if inner == nil {
		inner = http.DefaultTransport
	}
	out := *base
	out.Transport = &transport{base: inner, source: sharedSource(vendorID, f, store, c)}
	return &out, nil
}

// tokenSource is what a transport asks for a token: it caches the current one,
// renews it when it runs out, and persists whatever the renewal changed.
type tokenSource struct {
	vendor string
	flow   Flow
	store  Store
	client *http.Client

	mu      sync.Mutex
	cred    Credential
	current string
	expires time.Time
}

// sources holds one tokenSource per vendor and store.
//
// A refresh token rotates: spending it invalidates the one before it. Two
// clients for the same vendor, each with a source of its own, renew from the
// same stored token and the second renewal is refused outright — after which
// whichever wrote last decides what is left on disk. Sharing the source makes
// the renewal happen once, under one lock, with one result.
var sources struct {
	mu sync.Mutex
	m  map[sourceKey]*tokenSource
}

// sourceKey is what makes two clients the same client: the same vendor, read
// out of and written back to the same store.
type sourceKey struct {
	vendor string
	store  Store
}

func sharedSource(vendorID string, f Flow, store Store, c Credential) *tokenSource {
	fresh := func() *tokenSource {
		return &tokenSource{
			vendor: vendorID,
			flow:   f,
			store:  store,
			client: &http.Client{Timeout: 30 * time.Second},
			cred:   c,
		}
	}
	// A Store whose type cannot be a map key — a struct holding a map, passed
	// by value — gets a source of its own rather than a panic. It still
	// re-reads the store before renewing, which is most of the protection.
	if store != nil && !reflect.TypeOf(store).Comparable() {
		return fresh()
	}

	sources.mu.Lock()
	defer sources.mu.Unlock()
	if sources.m == nil {
		sources.m = make(map[sourceKey]*tokenSource)
	}
	key := sourceKey{vendor: vendorID, store: store}
	if s, ok := sources.m[key]; ok {
		return s
	}
	s := fresh()
	sources.m[key] = s
	return s
}

// forgetSources drops what is cached for a vendor, so a fresh sign-in or a
// sign-out is not shadowed by the token the previous one left in memory.
func forgetSources(vendorID string) {
	sources.mu.Lock()
	defer sources.mu.Unlock()
	for key := range sources.m {
		if key.vendor == vendorID {
			delete(sources.m, key)
		}
	}
}

func (s *tokenSource) token(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.current != "" && !expired(s.expires) {
		return s.current, nil
	}
	// Read the store again before renewing. Another process may have signed in
	// or refreshed since this source was built, and a rotated refresh token can
	// only be spent once — renewing from a copy that has already been
	// superseded fails outright and takes the sign-in with it.
	if s.store != nil {
		if stored, found, err := s.store.Load(s.vendor); err == nil && found && stored.Access != "" {
			s.cred = stored
		}
	}

	present, expires, updated, err := s.flow.Token(ctx, s.client, s.cred)
	if err != nil {
		return "", err
	}
	if updated != s.cred {
		rotated := updated.Refresh != s.cred.Refresh
		updated.Vendor = s.vendor
		s.cred = updated
		if s.store != nil {
			if err := s.store.Save(updated); err != nil {
				// A rotated refresh token that could not be written is gone:
				// the one still in the store has been spent, so the next run
				// starts signed out with nothing to renew from. That is worth
				// failing this request over, because it is the last moment
				// anyone can be told.
				if rotated {
					return "", fmt.Errorf("auth: %s issued a new refresh token that could not be stored, "+
						"and the stored one is now spent: %w", s.vendor, err)
				}
				// Nothing that cannot be recovered was lost — the endpoint a
				// sign-in rediscovers, at worst. A session that cannot write
				// to disk still works, and failing here would cost the turn.
			}
		}
	}
	s.current, s.expires = present, expires
	return present, nil
}
