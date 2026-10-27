package auth

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
	"golang.org/x/oauth2"
)

func TestIndependentSourcesShareOneRefresh(t *testing.T) {
	var refreshes atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path != "/token" {
			return
		}
		if refreshes.Add(1) != 1 {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = fmt.Fprint(w, `{"error":{"code":"refresh_token_reused"}}`)
			return
		}
		// Keep the first exchange open while the other source tries to refresh.
		time.Sleep(100 * time.Millisecond)
		_, _ = fmt.Fprint(w, `{"access_token":"fresh","refresh_token":"r2","expires_in":3600}`)
	}))
	defer server.Close()
	cfg := codexDefaults
	cfg.Endpoint.TokenURL = server.URL + "/token"
	f := newCodexFlow(cfg)
	store := newTestFileStore(t)
	stale := Credential{Vendor: codexVendor, Access: "old", Refresh: "r1", ExpiresAt: time.Now().Add(-time.Hour)}
	if err := store.Save(stale); err != nil {
		t.Fatal(err)
	}
	// Independent handles and sources stand in for separate processes; sharing
	// the process-local source cache would hide the rotating-token race.
	sources := []*tokenSource{
		{vendor: codexVendor, flow: f, store: store, client: server.Client(), cred: stale},
		{vendor: codexVendor, flow: f, store: &fileStore{path: store.path}, client: server.Client(), cred: stale},
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	for _, source := range sources {
		wg.Go(func() {
			<-start
			token, err := source.token(t.Context())
			if err != nil {
				t.Error(err)
			} else if token != "fresh" {
				t.Errorf("token = %q, want fresh", token)
			}
		})
	}
	close(start)
	wg.Wait()
	if got := refreshes.Load(); got != 1 {
		t.Fatalf("refreshes = %d, want 1: a rotating token was spent twice", got)
	}
}

type unreadableStore struct{ Store }

var errUnreadableStore = errors.New("credential store is unreadable")

func (unreadableStore) Load(string) (Credential, bool, error) {
	return Credential{}, false, errUnreadableStore
}

func TestRefreshDoesNotSpendATokenWhenReloadFails(t *testing.T) {
	withTokenHook(t, func(c Credential) (string, time.Time, Credential, error) {
		t.Error("refresh ran without reading the latest stored credential")
		return "token", time.Time{}, c, nil
	})
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("request escaped the transport after a credential read failure")
	}))
	defer server.Close()
	client, err := HTTPClient(stubVendor, Credential{Access: "old"}, unreadableStore{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Get(server.URL); !errors.Is(err, errUnreadableStore) { //nolint:bodyclose // no request should leave the transport
		t.Fatalf("err = %v, want the credential read failure", err)
	}
}

func TestWaitingForRefreshLockCanBeCanceled(t *testing.T) {
	store := newTestFileStore(t)
	err := store.WithRefreshLock(t.Context(), codexVendor, func() error {
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Millisecond)
		defer cancel()
		return (&fileStore{path: store.path}).WithRefreshLock(ctx, codexVendor, func() error {
			t.Error("entered a refresh lock already held by another handle")
			return nil
		})
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the waiting caller's deadline", err)
	}
	if err := store.WithRefreshLock(t.Context(), codexVendor, func() error { return nil }); err != nil {
		t.Fatalf("refresh lock was not released: %v", err)
	}
}

func TestCodexTerminalRefreshErrorsRequireSignIn(t *testing.T) {
	for _, body := range []string{
		`{"error":{"code":"refresh_token_reused","message":"refresh token was already used"}}`,
		`{"error":"invalid_grant","error_description":"expired"}`,
	} {
		t.Run(body, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = fmt.Fprint(w, body)
			}))
			defer server.Close()
			cfg := codexDefaults
			cfg.Endpoint.TokenURL = server.URL
			f := newCodexFlow(cfg)
			_, _, _, err := f.token(t.Context(), server.Client(), Credential{Refresh: "r1", ExpiresAt: time.Now().Add(-time.Hour)})
			if !ai.IsAuth(err) || ai.IsRetryable(err) {
				t.Fatalf("err = %v, want a non-retryable authentication error", err)
			}
			if _, ok := errors.AsType[*oauth2.RetrieveError](err); !ok {
				t.Fatal("the authentication error lost its original OAuth response")
			}
		})
	}
}

func TestCodexTemporaryRefreshFailureKeepsItsOriginalError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = fmt.Fprint(w, `{"error":{"code":"refresh_token_reused"}}`)
	}))
	defer server.Close()
	cfg := codexDefaults
	cfg.Endpoint.TokenURL = server.URL
	f := newCodexFlow(cfg)
	_, _, _, err := f.token(t.Context(), server.Client(), Credential{Refresh: "r1", ExpiresAt: time.Now().Add(-time.Hour)})
	if _, ok := errors.AsType[*oauth2.RetrieveError](err); !ok || ai.IsAuth(err) {
		t.Fatalf("err = %v, want the original temporary OAuth failure", err)
	}
}
