package responses

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

type codexRoundTripper func(*http.Request) (*http.Response, error)

func (f codexRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCodexCatalogDiscoversLatestVersionWithoutCredentials(t *testing.T) {
	var lookups atomic.Int32
	metadata := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lookups.Add(1)
		if r.Method != http.MethodGet || r.URL.EscapedPath() != "/@openai%2Fcodex/latest" {
			t.Errorf("metadata request = %s %s", r.Method, r.URL.EscapedPath())
		}
		for _, header := range []string{"Authorization", "chatgpt-account-id", "session_id", "Cookie", "x-api-key"} {
			if r.Header.Get(header) != "" {
				t.Errorf("provider header %s reached metadata discovery", header)
			}
		}
		fmt.Fprint(w, `{"version":"0.200.0"}`)
	}))
	t.Cleanup(metadata.Close)
	cache := &codexVersionCache{fetch: func(ctx context.Context) (string, error) {
		return fetchCodexVersion(ctx, codexMetadataClient, metadata.URL+"/@openai%2Fcodex/latest")
	}}

	catalog := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" || r.URL.Query().Get("client_version") != "0.200.0" {
			t.Errorf("catalog request = %s", r.URL)
		}
		if r.Header.Get("Authorization") != "Bearer catalog-only" {
			t.Error("catalog credential missing")
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"models":[{"slug":"new-model"}]}`)
	}))
	t.Cleanup(catalog.Close)
	client := &http.Client{Transport: codexRoundTripper(func(r *http.Request) (*http.Response, error) {
		clone := r.Clone(r.Context())
		for _, header := range []string{"Authorization", "chatgpt-account-id", "session_id", "Cookie", "x-api-key"} {
			clone.Header.Set(header, "catalog-only")
		}
		clone.Header.Set("Authorization", "Bearer catalog-only")
		return catalog.Client().Transport.RoundTrip(clone)
	})}
	for range 2 {
		d := driverFor(t, ai.Config{HTTPClient: client, Model: ai.Model{
			BaseURL: catalog.URL, Compat: ai.OpenAIResponsesCompat{CodexModelListing: true},
		}}).(*driver)
		d.versions = cache
		models, err := d.Models(context.Background())
		if err != nil || len(models) != 1 || models[0].ID != "new-model" {
			t.Fatalf("models = %+v, err = %v", models, err)
		}
	}
	if got := lookups.Load(); got != 1 {
		t.Errorf("metadata lookups across drivers = %d, want 1", got)
	}
}

func TestCodexCatalogVersionFallbackAndOverride(t *testing.T) {
	for _, pinned := range []string{"", "0.123.4"} {
		t.Run(pinned, func(t *testing.T) {
			want := pinned
			if want == "" {
				want = codexFallbackVersion
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.URL.Query().Get("client_version"); got != want {
					t.Errorf("catalog version = %q, want %q", got, want)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"models":[{"slug":"new-model"}]}`)
			}))
			t.Cleanup(srv.Close)
			d := driverFor(t, ai.Config{Model: ai.Model{
				BaseURL: srv.URL,
				Compat:  ai.OpenAIResponsesCompat{CodexModelListing: true, CodexClientVersion: pinned},
			}}).(*driver)
			lookups := 0
			d.versions = &codexVersionCache{fetch: func(context.Context) (string, error) {
				lookups++
				return "", context.DeadlineExceeded
			}}
			if _, err := d.Models(context.Background()); err != nil {
				t.Fatal(err)
			}
			if lookups != 0 && pinned != "" || lookups != 1 && pinned == "" {
				t.Errorf("metadata lookups = %d for override %q", lookups, pinned)
			}
		})
	}
}

func TestCodexVersionCacheRefreshAndFallback(t *testing.T) {
	calls, fail := 0, false
	latest := "0.200.0"
	cache := &codexVersionCache{fetch: func(context.Context) (string, error) {
		calls++
		if fail {
			return "", errors.New("metadata unavailable")
		}
		return latest, nil
	}}
	ctx := context.Background()
	if got := cache.resolve(ctx); got != latest {
		t.Fatalf("version = %q, want %q", got, latest)
	}
	cache.resolve(ctx)
	if calls != 1 || time.Until(cache.expires) < codexVersionTTL-time.Minute {
		t.Fatalf("successful lookup not cached: calls=%d, expires=%v", calls, cache.expires)
	}

	cache.expires = time.Time{}
	fail = true
	if got := cache.resolve(ctx); got != latest {
		t.Fatalf("warm fallback = %q, want %q", got, latest)
	}
	cache.resolve(ctx)
	if calls != 2 || time.Until(cache.expires) > codexVersionRetry || time.Until(cache.expires) < codexVersionRetry-time.Minute {
		t.Fatalf("failed lookup retry: calls=%d, expires=%v", calls, cache.expires)
	}

	cache.expires = time.Time{}
	fail, latest = false, "0.201.0"
	if got := cache.resolve(ctx); got != latest || calls != 3 {
		t.Fatalf("refreshed version = %q, calls=%d", got, calls)
	}

	coldCalls := 0
	cold := &codexVersionCache{fetch: func(context.Context) (string, error) {
		coldCalls++
		return "", errors.New("offline")
	}}
	for range 2 {
		if got := cold.resolve(ctx); got != codexFallbackVersion {
			t.Fatalf("cold fallback = %q", got)
		}
	}
	if coldCalls != 1 {
		t.Errorf("cold failed lookups = %d, want 1", coldCalls)
	}
}

func TestCodexVersionCacheCoalescesAndAllowsCanceledWaiters(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	var calls atomic.Int32
	cache := &codexVersionCache{fetch: func(context.Context) (string, error) {
		calls.Add(1)
		close(started)
		<-release
		return "0.200.0", nil
	}}
	results := make(chan string, 16)
	go func() { results <- cache.resolve(context.Background()) }()
	<-started
	for range 15 {
		go func() { results <- cache.resolve(context.Background()) }()
	}
	ctx, cancel := context.WithCancel(context.Background())
	canceled := make(chan string, 1)
	go func() { canceled <- cache.resolve(ctx) }()
	cancel()
	select {
	case got := <-canceled:
		if got != codexFallbackVersion {
			t.Errorf("canceled waiter = %q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("canceled waiter blocked on another lookup")
	}
	once.Do(func() { close(release) })
	for range 16 {
		select {
		case got := <-results:
			if got != "0.200.0" {
				t.Errorf("concurrent version = %q", got)
			}
		case <-time.After(time.Second):
			t.Fatal("lookup did not finish")
		}
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("concurrent lookups = %d, want 1", got)
	}
}

func TestCanceledCodexVersionLookupDoesNotPoisonCache(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	cache := &codexVersionCache{fetch: func(ctx context.Context) (string, error) {
		calls++
		if calls == 1 {
			cancel()
			return "", ctx.Err()
		}
		return "0.200.0", nil
	}}
	if got := cache.resolve(ctx); got != codexFallbackVersion || !cache.expires.IsZero() {
		t.Fatalf("canceled lookup = %q, expires=%v", got, cache.expires)
	}
	cache.resolve(ctx)
	if calls != 1 {
		t.Fatalf("already-canceled caller started a lookup: calls=%d", calls)
	}
	if got := cache.resolve(context.Background()); got != "0.200.0" || calls != 2 {
		t.Fatalf("lookup after cancellation = %q, calls=%d", got, calls)
	}
}

func TestFetchCodexVersionValidatesMetadata(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		status     int
		want       string
	}{
		{name: "stable", body: `{"version":"0.169.1"}`, want: "0.169.1"},
		{name: "http error", status: http.StatusServiceUnavailable, body: `{"version":"0.169.1"}`},
		{name: "malformed", body: `{`},
		{name: "missing", body: `{}`},
		{name: "null", body: `{"version":null}`},
		{name: "number", body: `{"version":169}`},
		{name: "empty", body: `{"version":""}`},
		{name: "prerelease", body: `{"version":"0.170.0-alpha.1"}`},
		{name: "not semver", body: `{"version":"latest"}`},
		{name: "leading zero", body: `{"version":"0.0169.1"}`},
		{name: "query injection", body: `{"version":"0.169.1&other=true"}`},
		{name: "long version", body: `{"version":"` + strings.Repeat("1", 40) + `.1.1"}`},
		{name: "trailing data", body: `{"version":"0.169.1"}{}`},
		{name: "oversized", body: `{"version":"0.169.1"}` + strings.Repeat(" ", codexVersionMaxBytes)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.status != 0 {
					w.WriteHeader(tt.status)
				}
				io.WriteString(w, tt.body)
			}))
			t.Cleanup(srv.Close)
			got, err := fetchCodexVersion(context.Background(), srv.Client(), srv.URL)
			if got != tt.want || (err == nil) != (tt.want != "") {
				t.Fatalf("version = %q, err = %v, want %q", got, err, tt.want)
			}
		})
	}
}

func TestFetchCodexVersionBoundsDeadline(t *testing.T) {
	for _, limit := range []time.Duration{time.Minute, 100 * time.Millisecond} {
		ctx, cancel := context.WithTimeout(context.Background(), limit)
		client := &http.Client{Transport: codexRoundTripper(func(r *http.Request) (*http.Response, error) {
			deadline, ok := r.Context().Deadline()
			if !ok || time.Until(deadline) > min(limit, codexVersionTimeout) {
				t.Errorf("metadata deadline = %v, want at most %s", deadline, min(limit, codexVersionTimeout))
			}
			return nil, context.DeadlineExceeded
		})}
		_, err := fetchCodexVersion(ctx, client, codexLatestURL)
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("metadata error = %v, want deadline exceeded", err)
		}
	}
}
