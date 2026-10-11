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

func TestHeaderRedirectsWithoutOriginAreRejected(t *testing.T) {
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
		client := withHeaders(base, map[string]string{"Authorization": "Bearer secret"})
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
