package oauth

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"golang.org/x/oauth2"
)

// The person is shown the code and the page to enter it on, and the grant
// keeps polling past a pending answer to the token it is then issued.
func TestDeviceShowsTheCodeAndWaitsForTheToken(t *testing.T) {
	polls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/code":
			_, _ = fmt.Fprint(w, `{"device_code":"dc","user_code":"UC-1","verification_uri":"https://provider.test/activate","interval":1,"expires_in":60}`)
		case "/token":
			polls++
			if polls == 1 {
				_, _ = fmt.Fprint(w, `{"error":"authorization_pending"}`)
				return
			}
			_, _ = fmt.Fprint(w, `{"access_token":"at","expires_in":1800}`)
		}
	}))
	defer server.Close()

	var shown Prompt
	ui := InteractionFunc(func(ctx context.Context, p Prompt) error { shown = p; return nil })
	token, err := Device(t.Context(), &oauth2.Config{ClientID: "client", Endpoint: oauth2.Endpoint{
		DeviceAuthURL: server.URL + "/code", TokenURL: server.URL + "/token",
	}}, ui)
	if err != nil {
		t.Fatalf("Device: %v", err)
	}
	if token.AccessToken != "at" || polls != 2 {
		t.Errorf("token = %q after %d polls, want the one issued once the person finished", token.AccessToken, polls)
	}
	if shown.UserCode != "UC-1" || shown.URL != "https://provider.test/activate" || shown.ExpiresAt.IsZero() {
		t.Errorf("the person was shown %+v, want the code, the page and when it stops working", shown)
	}
}

func TestDeviceReportsAFailureToIssueCodes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":"unauthorized_client","error_description":"unknown client"}`)
	}))
	defer server.Close()

	_, err := Device(t.Context(), &oauth2.Config{ClientID: "nobody", Endpoint: oauth2.Endpoint{
		DeviceAuthURL: server.URL, TokenURL: server.URL,
	}}, nil)
	if err == nil || !strings.Contains(err.Error(), "unauthorized_client") {
		t.Fatalf("err = %v, want the provider's own code", err)
	}
}

// The code comes back to the loopback listener and is exchanged with the
// verifier whose challenge went out on the authorize URL.
func TestCodeExchangesTheCallbackWithItsVerifier(t *testing.T) {
	var challenge string
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
		if r.PostForm.Get("code") != "the-code" || base64.RawURLEncoding.EncodeToString(sum[:]) != challenge {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprint(w, `{"error":"invalid_grant"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"access_token":"at","refresh_token":"rt","expires_in":3600}`)
	}))
	defer provider.Close()

	redirect := loopbackURL(t)
	ui := InteractionFunc(func(ctx context.Context, p Prompt) error {
		u, err := url.Parse(p.URL)
		if err != nil {
			return err
		}
		challenge = u.Query().Get("code_challenge")
		return visit(redirect + "?code=the-code&state=" + url.QueryEscape(u.Query().Get("state")))
	})

	token, err := Code(t.Context(), codeConfig(redirect, provider.URL), ui)
	if err != nil {
		t.Fatalf("Code: %v", err)
	}
	if token.AccessToken != "at" || token.RefreshToken != "rt" {
		t.Errorf("token = %+v, want what the exchange issued", token)
	}
}

// The check that stops a callback from somebody else's sign-in attempt — or
// from an attacker's — being taken for this one.
func TestCodeRejectsAMismatchedState(t *testing.T) {
	redirect := loopbackURL(t)
	ui := InteractionFunc(func(ctx context.Context, p Prompt) error {
		u, err := url.Parse(p.URL)
		if err != nil {
			return err
		}
		return visit(redirect + "?code=the-code&state=" + url.QueryEscape(u.Query().Get("state")+"-wrong"))
	})

	_, err := Code(t.Context(), codeConfig(redirect, "https://provider.test"), ui)
	if err == nil || !strings.Contains(err.Error(), "did not match") {
		t.Fatalf("err = %v, want the callback refused", err)
	}
}

func TestCodeReportsWhatTheProviderRefused(t *testing.T) {
	redirect := loopbackURL(t)
	ui := InteractionFunc(func(ctx context.Context, p Prompt) error {
		return visit(redirect + "?error=access_denied&error_description=nope")
	})

	_, err := Code(t.Context(), codeConfig(redirect, "https://provider.test"), ui)
	if err == nil || !strings.Contains(err.Error(), "access_denied") {
		t.Fatalf("err = %v, want the provider's own refusal", err)
	}
}

func codeConfig(redirect, provider string) *oauth2.Config {
	return &oauth2.Config{
		ClientID:    "client",
		RedirectURL: redirect,
		Endpoint: oauth2.Endpoint{
			AuthURL: provider + "/authorize", TokenURL: provider + "/token", AuthStyle: oauth2.AuthStyleInParams,
		},
	}
}

// visit is the browser coming back to the loopback listener.
func visit(u string) error {
	res, err := http.Get(u) //nolint:noctx // a local one-shot request in a test
	if err != nil {
		return err
	}
	return res.Body.Close()
}

// loopbackURL reserves a port for a redirect the flow will listen on itself.
func loopbackURL(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("http://127.0.0.1:%d/callback", port)
}
