package oauth

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestTokenValid(t *testing.T) {
	tests := map[string]struct {
		token Token
		want  bool
	}{
		"no token at all":         {Token{}, false},
		"no stated lifetime":      {Token{Access: "t"}, true},
		"good for another hour":   {Token{Access: "t", Expires: time.Now().Add(time.Hour)}, true},
		"already run out":         {Token{Access: "t", Expires: time.Now().Add(-time.Second)}, false},
		"runs out inside the gap": {Token{Access: "t", Expires: time.Now().Add(ExpiryMargin / 2)}, false},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			if got := tc.token.Valid(); got != tc.want {
				t.Errorf("Valid = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestErrorDenied(t *testing.T) {
	tests := map[string]struct {
		code string
		want bool
	}{
		"the person refused":    {"access_denied", true},
		"the attempt ran out":   {"expired_token", true},
		"the provider is down":  {"http_error", false},
		"still waiting on them": {"authorization_pending", false},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			if got := (&Error{Code: tc.code}).Denied(); got != tc.want {
				t.Errorf("Denied = %v, want %v: retrying a refusal will never succeed", got, tc.want)
			}
		})
	}
}

func TestErrorMessageCarriesWhatTheProviderSaid(t *testing.T) {
	err := &Error{Code: "access_denied", Description: "the user said no", Status: 403}
	for _, want := range []string{"access_denied", "the user said no", "403"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Error() = %q, want it to mention %q", err.Error(), want)
		}
	}
}

// TestAuthorizeURLAndExchange covers the pair a web application uses when it
// already has a callback route and does not want this package opening a
// listener of its own.
func TestAuthorizeURLAndExchange(t *testing.T) {
	var form url.Values
	token := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		form = r.PostForm
		_, _ = fmt.Fprint(w, `{"access_token":"at","refresh_token":"rt","expires_in":3600}`)
	}))
	defer token.Close()

	endpoints := CodeEndpoints{
		Authorize: "https://provider.test/authorize",
		Token:     token.URL,
		Redirect:  "https://app.test/callback",
	}
	cfg := Config{ClientID: "client", Scopes: []string{"a", "b"}}

	authorize, verifier, err := AuthorizeURL(cfg, endpoints, "the-state")
	if err != nil {
		t.Fatalf("AuthorizeURL: %v", err)
	}
	u, err := url.Parse(authorize)
	if err != nil {
		t.Fatalf("the authorize URL does not parse: %v", err)
	}
	q := u.Query()
	for field, want := range map[string]string{
		"response_type":         "code",
		"client_id":             "client",
		"state":                 "the-state",
		"code_challenge_method": "S256",
		"scope":                 "a b",
		"redirect_uri":          "https://app.test/callback",
	} {
		if q.Get(field) != want {
			t.Errorf("%s = %q, want %q", field, q.Get(field), want)
		}
	}
	// The verifier is the secret; only its hash goes to the provider.
	if q.Get("code_challenge") == "" || q.Get("code_challenge") == verifier {
		t.Error("the challenge is missing or is the verifier itself, which defeats PKCE")
	}

	got, err := Exchange(t.Context(), cfg, endpoints, "the-code", verifier)
	if err != nil {
		t.Fatalf("Exchange: %v", err)
	}
	if got.Access != "at" || got.Refresh != "rt" {
		t.Errorf("token = %+v, want the stub's", got)
	}
	if got.Expires.IsZero() {
		t.Error("expires_in was not turned into a time")
	}
	if form.Get("code_verifier") != verifier || form.Get("grant_type") != "authorization_code" {
		t.Errorf("exchange sent %v, want the code grant with the verifier", form)
	}
}

// TestCodeRejectsAMismatchedState is the check that stops a callback from
// somebody else's sign-in attempt — or from an attacker's — being taken for
// this one.
func TestCodeRejectsAMismatchedState(t *testing.T) {
	redirect := loopbackURL(t)

	ui := InteractionFunc(func(ctx context.Context, p Prompt) error {
		u, err := url.Parse(p.URL)
		if err != nil {
			return err
		}
		// Come back with everything right except the state.
		callback := redirect + "?code=the-code&state=" + url.QueryEscape(u.Query().Get("state")+"-wrong")
		res, err := http.Get(callback) //nolint:noctx // a local one-shot request in a test
		if err != nil {
			return err
		}
		return res.Body.Close()
	})

	_, err := Code(t.Context(), Config{ClientID: "client"}, CodeEndpoints{
		Authorize: "https://provider.test/authorize",
		Token:     "https://provider.test/token",
		Redirect:  redirect,
	}, ui)

	var oauthErr *Error
	if !errors.As(err, &oauthErr) || oauthErr.Code != "invalid_state" {
		t.Fatalf("err = %v, want an invalid_state refusal", err)
	}
}

func TestCodeReportsWhatTheProviderRefused(t *testing.T) {
	redirect := loopbackURL(t)
	ui := InteractionFunc(func(ctx context.Context, p Prompt) error {
		res, err := http.Get(redirect + "?error=access_denied&error_description=nope") //nolint:noctx // a local one-shot request in a test
		if err != nil {
			return err
		}
		return res.Body.Close()
	})

	_, err := Code(t.Context(), Config{ClientID: "client"}, CodeEndpoints{
		Authorize: "https://provider.test/authorize",
		Redirect:  redirect,
	}, ui)

	var oauthErr *Error
	if !errors.As(err, &oauthErr) || !oauthErr.Denied() {
		t.Fatalf("err = %v, want the provider's own refusal", err)
	}
}

// TestDeviceBacksOffWhenToldTo pins the one thing that turns a working device
// grant into a refused one: a provider that says slow_down and is ignored
// stops answering.
func TestDeviceBacksOffWhenToldTo(t *testing.T) {
	// Real seconds would make this test cost seven of them; the behavior under
	// test is that the gap grows, not what it grows to.
	floor, step := pollFloor, slowDownStep
	pollFloor, slowDownStep = time.Millisecond, 50*time.Millisecond
	t.Cleanup(func() { pollFloor, slowDownStep = floor, step })

	var mu sync.Mutex
	var polls []time.Time

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/code":
			_, _ = fmt.Fprint(w, `{"device_code":"dc","user_code":"UC-1","verification_uri":"https://provider.test/activate","interval":0,"expires_in":60}`)
		case "/token":
			mu.Lock()
			polls = append(polls, time.Now())
			n := len(polls)
			mu.Unlock()
			switch n {
			case 1:
				_, _ = fmt.Fprint(w, `{"error":"authorization_pending"}`)
			case 2:
				_, _ = fmt.Fprint(w, `{"error":"slow_down"}`)
			default:
				_, _ = fmt.Fprint(w, `{"access_token":"at","expires_in":1800}`)
			}
		}
	}))
	defer server.Close()

	var shown Prompt
	ui := InteractionFunc(func(ctx context.Context, p Prompt) error { shown = p; return nil })

	token, err := Device(t.Context(), Config{ClientID: "client"},
		DeviceEndpoints{Code: server.URL + "/code", Token: server.URL + "/token"}, ui)
	if err != nil {
		t.Fatalf("Device: %v", err)
	}
	if token.Access != "at" {
		t.Errorf("token = %+v, want the one issued after the back-off", token)
	}
	if shown.UserCode != "UC-1" || shown.URL == "" {
		t.Errorf("the person was shown %+v, want the code and the page to enter it on", shown)
	}
	if shown.ExpiresAt.IsZero() {
		t.Error("the person was not told when the attempt stops being accepted")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(polls) != 3 {
		t.Fatalf("polled %d times, want three", len(polls))
	}
	if gap := polls[2].Sub(polls[1]); gap < slowDownStep {
		t.Errorf("the poll after slow_down came %v later, want at least the %v back-off", gap, slowDownStep)
	}
}

func TestDeviceReportsAFailureToIssueCodes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":"unauthorized_client","error_description":"unknown client"}`)
	}))
	defer server.Close()

	_, err := Device(t.Context(), Config{ClientID: "nobody"},
		DeviceEndpoints{Code: server.URL, Token: server.URL}, nil)

	var oauthErr *Error
	if !errors.As(err, &oauthErr) || oauthErr.Code != "unauthorized_client" {
		t.Fatalf("err = %v, want the provider's own code", err)
	}
}

// TestDeviceReadsAFormEncodedReply covers the providers that answer the code
// endpoint in the shape an early draft used rather than in JSON.
func TestDeviceReadsAFormEncodedReply(t *testing.T) {
	restore := pollFloor
	pollFloor = time.Millisecond
	t.Cleanup(func() { pollFloor = restore })

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/code") {
			// io.WriteString, not fmt.Fprint: the body is form-encoded, and the
			// %2F in it reads to vet as a formatting directive.
			_, _ = io.WriteString(w, "device_code=dc&user_code=UC-2&verification_uri=https%3A%2F%2Fprovider.test&interval=1&expires_in=600")
			return
		}
		_, _ = fmt.Fprint(w, `{"access_token":"at"}`)
	}))
	defer server.Close()

	token, err := Device(t.Context(), Config{ClientID: "client"},
		DeviceEndpoints{Code: server.URL + "/code", Token: server.URL + "/token"}, nil)
	if err != nil {
		t.Fatalf("Device: %v", err)
	}
	if token.Access != "at" {
		t.Errorf("token = %+v, want the one the form-encoded flow produced", token)
	}
}

func TestRefresh(t *testing.T) {
	tests := map[string]struct {
		body        string
		sent        string
		wantRefresh string
		wantErr     string
	}{
		"a provider that rotates the refresh token": {
			body: `{"access_token":"new","refresh_token":"rotated"}`, sent: "old", wantRefresh: "rotated"},
		// One that does not send a new one expects the old to keep working;
		// dropping it would sign the person out at the next renewal.
		"a provider that does not": {
			body: `{"access_token":"new"}`, sent: "old", wantRefresh: "old"},
		"nothing to refresh from": {sent: "", wantErr: "invalid_grant"},
		"the grant was rejected": {
			body: `{"error":"invalid_grant","error_description":"expired"}`, sent: "old", wantErr: "invalid_grant"},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = fmt.Fprint(w, tc.body)
			}))
			defer server.Close()

			got, err := Refresh(t.Context(), Config{ClientID: "client"}, server.URL, tc.sent)
			if tc.wantErr != "" {
				var oauthErr *Error
				if !errors.As(err, &oauthErr) || oauthErr.Code != tc.wantErr {
					t.Fatalf("err = %v, want %s", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Refresh: %v", err)
			}
			if got.Refresh != tc.wantRefresh {
				t.Errorf("Refresh token = %q, want %q", got.Refresh, tc.wantRefresh)
			}
		})
	}
}

// TestATokenEndpointThatAnswersWithSomethingElse covers the gateway error page
// and the HTML login screen: reporting those as a JSON parse failure points at
// the wrong thing entirely.
func TestATokenEndpointThatAnswersWithSomethingElse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = fmt.Fprint(w, "<html><body>502 Bad Gateway</body></html>")
	}))
	defer server.Close()

	_, err := Refresh(t.Context(), Config{ClientID: "client"}, server.URL, "old")
	var oauthErr *Error
	if !errors.As(err, &oauthErr) {
		t.Fatalf("err = %v, want an *Error", err)
	}
	if oauthErr.Code != "invalid_response" || oauthErr.Status != http.StatusBadGateway {
		t.Errorf("err = %v, want invalid_response carrying the status", oauthErr)
	}
	if !strings.Contains(oauthErr.Description, "502") {
		t.Errorf("description = %q, want what the endpoint actually said", oauthErr.Description)
	}
}

func TestTruncateKeepsAnErrorReadable(t *testing.T) {
	if got := truncate("  short  ", 200); got != "short" {
		t.Errorf("truncate = %q, want it trimmed", got)
	}
	long := strings.Repeat("x", 300)
	if got := truncate(long, 200); len(got) != 200+len("…") {
		t.Errorf("truncate kept %d bytes, want 200 plus the ellipsis", len(got))
	}
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
