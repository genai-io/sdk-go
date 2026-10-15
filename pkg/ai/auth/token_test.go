package auth

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
)

// The test vendors. The stub flows are added once and their behavior varied
// through the hooks below.
const (
	stubVendor   = "test-stub"
	rotateVendor = "test-rotating"
	copilotStub  = "test-copilot"
)

var stubHooks struct {
	mu    sync.Mutex
	token func(c Credential) (string, time.Time, Credential, error)
}

func init() {
	answer := func(ctx context.Context, client *http.Client, c Credential) (string, time.Time, Credential, error) {
		stubHooks.mu.Lock()
		fn := stubHooks.token
		stubHooks.mu.Unlock()
		if fn == nil {
			return c.Access, c.ExpiresAt, c, nil
		}
		return fn(c)
	}
	signIn := func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error) {
		return Credential{Access: "signed-in"}, nil
	}
	flows[stubVendor] = Flow{Method: "stub", Login: signIn, Token: answer}
	flows[rotateVendor] = Flow{Method: "stub", Login: signIn, Token: answer}
}

// withTokenHook installs the stub's renewal for one test.
func withTokenHook(t *testing.T, fn func(c Credential) (string, time.Time, Credential, error)) {
	t.Helper()
	stubHooks.mu.Lock()
	stubHooks.token = fn
	stubHooks.mu.Unlock()
	t.Cleanup(func() {
		stubHooks.mu.Lock()
		stubHooks.token = nil
		stubHooks.mu.Unlock()
		forgetSources(stubVendor)
		forgetSources(rotateVendor)
	})
	forgetSources(stubVendor)
	forgetSources(rotateVendor)
}

// TestInteractiveKnowsTheRegisteredVendors covers the two grants this package
// ships and a stub added by the tests.
func TestInteractiveKnowsTheRegisteredVendors(t *testing.T) {
	for _, id := range []string{"copilot", "openai-codex", stubVendor} {
		method, ok := Interactive(id)
		if !ok {
			t.Errorf("Interactive(%q) said no; a registered flow is not reachable", id)
		}
		if method == "" {
			t.Errorf("Interactive(%q) named no grant", id)
		}
	}
	if _, ok := Interactive("deepseek"); ok {
		t.Error("a key-based vendor was reported as signing in interactively")
	}
}

func TestExpiredIsOneRule(t *testing.T) {
	tests := map[string]struct {
		at   time.Time
		want bool
	}{
		"no stated lifetime":     {time.Time{}, false},
		"good for another hour":  {time.Now().Add(time.Hour), false},
		"gone":                   {time.Now().Add(-time.Minute), true},
		"goes inside the margin": {time.Now().Add(expiryMargin / 2), true},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			if got := expired(tc.at); got != tc.want {
				t.Errorf("expired = %v, want %v", got, tc.want)
			}
			if got := (Credential{ExpiresAt: tc.at}).Expired(); got != tc.want {
				t.Errorf("Credential.Expired = %v, want %v: the two must not disagree", got, tc.want)
			}
		})
	}
}

// TestTheTransportPresentsAndRenews covers the whole path: the token is minted
// on the first request, cached for the second, renewed when it runs out inside
// the margin, and what the renewal changed is written back.
func TestTheTransportPresentsAndRenews(t *testing.T) {
	var minted int
	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		minted++
		c.Endpoint = fmt.Sprintf("https://endpoint-%d.test", minted)
		// Each token is good for less than the margin, so the next request
		// has to renew.
		return fmt.Sprintf("tok-%d", minted), time.Now().Add(expiryMargin / 2), c, nil
	})

	var presented []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		presented = append(presented, r.Header.Get("Authorization"))
	}))
	defer upstream.Close()

	store := &memStore{}
	if err := store.Save(Credential{Vendor: stubVendor, Access: "stored"}); err != nil {
		t.Fatal(err)
	}
	client, err := httpClient(stubVendor, Credential{Access: "stored"}, store, nil)
	if err != nil {
		t.Fatal(err)
	}

	for range 2 {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, upstream.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		res, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if err := res.Body.Close(); err != nil {
			t.Fatal(err)
		}
	}

	if len(presented) != 2 || presented[0] != "Bearer tok-1" || presented[1] != "Bearer tok-2" {
		t.Errorf("presented %v, want a fresh token on each request once the first ran out", presented)
	}
	// What the renewal learned is persisted, or the next run rediscovers it.
	stored, _, err := store.Load(stubVendor)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Endpoint != "https://endpoint-2.test" {
		t.Errorf("stored endpoint = %q, want what the last renewal reported", stored.Endpoint)
	}
	if stored.Vendor != stubVendor {
		t.Errorf("stored vendor = %q, want it stamped so it can be loaded back", stored.Vendor)
	}
}

func TestATokenWithNoStatedLifetimeIsMintedOnce(t *testing.T) {
	var minted int
	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		minted++
		return "tok", time.Time{}, c, nil
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer upstream.Close()

	client, err := httpClient(stubVendor, Credential{Access: "stored"}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, upstream.URL, nil)
		res, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_ = res.Body.Close()
	}
	if minted != 1 {
		t.Errorf("minted %d tokens, want one: a token with no stated lifetime does not expire", minted)
	}
}

func TestClientsWithoutAStoreKeepTheirOwnCredentials(t *testing.T) {
	withTokenHook(t, nil)
	presented := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		presented <- r.Header.Get("Authorization")
	}))
	defer server.Close()
	for _, token := range []string{"account-a", "account-b"} {
		client, err := httpClient(stubVendor, Credential{Access: token}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL, nil)
		res, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_ = res.Body.Close()
		if got := <-presented; got != "Bearer "+token {
			t.Fatalf("presented %q, want account's own token %q", got, token)
		}
	}
}

func TestAuthenticatedRedirectsStayOnTheOriginalOrigin(t *testing.T) {
	withTokenHook(t, nil)
	var visited atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		visited.Store(true)
	}))
	defer target.Close()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/same":
			http.Redirect(w, r, "/done", http.StatusFound)
		case "/other":
			http.Redirect(w, r, target.URL, http.StatusFound)
		case "/downgrade":
			http.Redirect(w, r, "http://"+r.Host+"/done", http.StatusFound)
		default:
			if r.Header.Get("Authorization") != "Bearer secret" {
				t.Error("same-origin redirect lost its credential")
			}
		}
	}))
	defer server.Close()
	checks := 0
	base := server.Client()
	base.CheckRedirect = func(*http.Request, []*http.Request) error {
		checks++
		return nil
	}
	client, err := httpClient(stubVendor, Credential{Access: "secret"}, nil, base)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/same", "/other", "/downgrade"} {
		req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL+path, nil)
		res, err := client.Do(req)
		if res != nil {
			_ = res.Body.Close()
		}
		if (err != nil) != (path != "/same") {
			t.Errorf("%s: err = %v", path, err)
		}
	}
	if visited.Load() || checks != 3 {
		t.Errorf("cross-origin visited=%v; existing redirect checks=%d, want false and 3", visited.Load(), checks)
	}
}

func TestAuthenticatedRedirectsWithoutOriginAreRejected(t *testing.T) {
	withTokenHook(t, nil)
	for _, previous := range []*http.Request{nil, {}} {
		visited := false
		base := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Host == "origin.test" {
				return &http.Response{StatusCode: http.StatusFound, Body: http.NoBody,
					Header: http.Header{"Location": {"https://other.test"}}, Request: previous}, nil
			}
			visited = true
			return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
		})}
		client, err := httpClient(stubVendor, Credential{Access: "secret"}, nil, base)
		if err != nil {
			t.Fatal(err)
		}
		req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://origin.test", nil)
		res, err := client.Do(req)
		if res != nil {
			_ = res.Body.Close()
		}
		if err == nil || visited {
			t.Errorf("previous request=%v: err=%v, cross-origin visited=%v", previous, err, visited)
		}
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// TestTwoClientsShareOneRenewal is the race this package had: a refresh token
// rotates, so two clients for one vendor renewing independently means the
// second renewal is refused and one of them is left signed out.
func TestTwoClientsShareOneRenewal(t *testing.T) {
	var mu sync.Mutex
	var renewals int
	spent := map[string]bool{}

	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		mu.Lock()
		defer mu.Unlock()
		if spent[c.Refresh] {
			return "", time.Time{}, c, fmt.Errorf("refresh token %q has already been spent", c.Refresh)
		}
		spent[c.Refresh] = true
		renewals++
		c.Refresh = fmt.Sprintf("r%d", renewals+1)
		return fmt.Sprintf("tok-%d", renewals), time.Now().Add(time.Hour), c, nil
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer upstream.Close()

	store := newTestFileStore(t)
	otherHandle, err := NewFileStore(store.path)
	if err != nil {
		t.Fatal(err)
	}
	start := Credential{Vendor: rotateVendor, Access: "a", Refresh: "r1"}
	if err := store.Save(start); err != nil {
		t.Fatal(err)
	}

	first, err := httpClient(rotateVendor, start, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := httpClient(rotateVendor, start, otherHandle, nil)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i, client := range []*http.Client{first, second} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, upstream.URL, nil)
			res, err := client.Do(req)
			if err != nil {
				errs[i] = err
				return
			}
			_ = res.Body.Close()
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("client %d failed: %v", i, err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if renewals != 1 {
		t.Errorf("the refresh token was spent %d times, want once: two clients for one vendor "+
			"and one store share a renewal", renewals)
	}
}

// TestARotatedTokenThatCannotBeStoredFailsLoudly. The stored refresh token has
// already been spent by then, so carrying on quietly means the next run starts
// signed out with nothing to renew from and no idea why.
func TestARotatedTokenThatCannotBeStoredFailsLoudly(t *testing.T) {
	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		c.Refresh = "rotated"
		return "tok", time.Now().Add(time.Hour), c, nil
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer upstream.Close()

	client, err := httpClient(rotateVendor, Credential{Access: "a", Refresh: "r1"}, readOnlyStore{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, upstream.URL, nil)
	res, err := client.Do(req) //nolint:bodyclose // the request never reaches the server
	if err == nil {
		_ = res.Body.Close()
		t.Fatal("the request went through although the rotated token could not be stored")
	}
	if !strings.Contains(err.Error(), "spent") {
		t.Errorf("err = %v, want it to say the stored token is now spent", err)
	}
}

// TestASaveFailureThatLosesNothingDoesNotFailTheTurn is the other half of that
// rule: an endpoint a sign-in would rediscover is not worth a failed request,
// and a session with a read-only config directory still has to work.
func TestASaveFailureThatLosesNothingDoesNotFailTheTurn(t *testing.T) {
	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		c.Endpoint = "https://rediscovered.test"
		return "tok", time.Now().Add(time.Hour), c, nil
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer upstream.Close()

	client, err := httpClient(stubVendor, Credential{Access: "a"}, readOnlyStore{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, upstream.URL, nil)
	res, err := client.Do(req)
	if err != nil {
		t.Fatalf("the request failed over a save that lost nothing: %v", err)
	}
	_ = res.Body.Close()
}

func TestARenewalFailureIsReportedToTheCaller(t *testing.T) {
	boom := errors.New("the provider said no")
	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		return "", time.Time{}, c, boom
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer upstream.Close()

	client, err := httpClient(stubVendor, Credential{Access: "a"}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, upstream.URL, nil)
	if _, err := client.Do(req); !errors.Is(err, boom) { //nolint:bodyclose // there is no response
		t.Errorf("err = %v, want the renewal failure", err)
	}
}

func TestHTTPClientNeedsARegisteredFlow(t *testing.T) {
	if _, err := httpClient("deepseek", Credential{}, nil, nil); err == nil {
		t.Error("built a token-renewing client for a vendor that takes an API key")
	}
}

// readOnlyStore stands in for a config directory nobody can write to.
type readOnlyStore struct{}

func (readOnlyStore) Load(string) (Credential, bool, error) { return Credential{}, false, nil }
func (readOnlyStore) Save(Credential) error                 { return errors.New("read-only") }
func (readOnlyStore) Delete(string) error                   { return errors.New("read-only") }
func (readOnlyStore) List() ([]string, error)               { return nil, nil }

// An expired Codex credential is refreshed, as a public client names itself in
// the form, and a provider that does not rotate the refresh token leaves the
// stored one in place.
func TestCodexRefreshesAnExpiredCredential(t *testing.T) {
	var form url.Values
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		form = r.PostForm
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"access_token":"fresh","expires_in":3600}`)
	}))
	defer server.Close()

	cfg := codexDefaults
	cfg.Endpoint.TokenURL = server.URL
	flow := newCodexFlow(cfg)

	stale := Credential{Access: "old", Refresh: "rt", ExpiresAt: time.Now().Add(-time.Minute)}
	present, expires, updated, err := flow.Token(t.Context(), server.Client(), stale)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if present != "fresh" || updated.Refresh != "rt" || !expires.After(time.Now()) {
		t.Errorf("got %q, %v, %+v; want the refreshed token and the old refresh token kept", present, expires, updated)
	}
	if form.Get("grant_type") != "refresh_token" || form.Get("refresh_token") != "rt" || form.Get("client_id") != cfg.ClientID {
		t.Errorf("refresh sent %v", form)
	}
}
