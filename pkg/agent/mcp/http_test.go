package mcp

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestHeaderRedirectsStayOnTheOriginalOrigin(t *testing.T) {
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
	client := withHeaders(base, map[string]string{"Authorization": "Bearer secret"})
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
