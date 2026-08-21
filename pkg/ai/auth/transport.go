package auth

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Transport injects a vendor's current token into every request.
//
// A RoundTripper rather than a fixed header because Copilot's token lasts
// about half an hour — well inside a long session — so it cannot be resolved
// once at start-up. It overwrites any Authorization the driver set, which is
// how a driver that insists on a placeholder key still ends up sending a real
// one.
type Transport struct {
	base   http.RoundTripper
	source *tokenSource
}

func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
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
	f, ok := flows[vendorID]
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
	out.Transport = &Transport{
		base: inner,
		source: &tokenSource{
			vendor: vendorID,
			flow:   f,
			store:  store,
			client: &http.Client{Timeout: 30 * time.Second},
			cred:   c,
		},
	}
	return &out, nil
}

// tokenSource is what Transport asks for a token: it caches the current one,
// renews it when it runs out, and persists whatever the renewal changed.
type tokenSource struct {
	vendor string
	flow   flow
	store  Store
	client *http.Client

	mu      sync.Mutex
	cred    Credential
	current string
	expires time.Time
}

func (s *tokenSource) token(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.current != "" && (s.expires.IsZero() || time.Now().Add(time.Minute).Before(s.expires)) {
		return s.current, nil
	}
	present, expires, updated, err := s.flow.token(ctx, s.client, s.cred)
	if err != nil {
		return "", err
	}
	if updated != s.cred {
		updated.Vendor = s.vendor
		s.cred = updated
		// A session that cannot write to disk can still run; losing the
		// refreshed credential costs one extra sign-in later, and failing the
		// request costs the turn.
		if s.store != nil {
			_ = s.store.Save(updated)
		}
	}
	s.current, s.expires = present, expires
	return present, nil
}
