package oauth_test

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm/auth/oauth"
)

func silent() oauth.Interaction {
	return oauth.InteractionFunc(func(context.Context, oauth.Prompt) error { return nil })
}

// ── device authorization grant ──

func TestDeviceGrant(t *testing.T) {
	var polls atomic.Int32
	var shown oauth.Prompt

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		switch r.URL.Path {
		case "/code":
			if r.Form.Get("client_id") != "cid" || r.Form.Get("scope") != "read:user" {
				t.Errorf("code request = %v", r.Form)
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"device_code":"dev","user_code":"ABCD-1234",
				"verification_uri":"https://example.test/activate","expires_in":900,"interval":1}`)
		case "/token":
			w.Header().Set("Content-Type", "application/json")
			// The person has not finished yet on the first two polls.
			if polls.Add(1) < 3 {
				fmt.Fprint(w, `{"error":"authorization_pending"}`)
				return
			}
			fmt.Fprint(w, `{"access_token":"gho_real","token_type":"bearer","scope":"read:user"}`)
		}
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	token, err := oauth.Device(ctx,
		oauth.Config{ClientID: "cid", Scopes: []string{"read:user"}},
		oauth.DeviceEndpoints{Code: server.URL + "/code", Token: server.URL + "/token"},
		oauth.InteractionFunc(func(_ context.Context, p oauth.Prompt) error { shown = p; return nil }))
	if err != nil {
		t.Fatalf("Device: %v", err)
	}
	if token.Access != "gho_real" {
		t.Errorf("Access = %q", token.Access)
	}
	if polls.Load() < 3 {
		t.Errorf("polled %d times, want it to have waited", polls.Load())
	}
	// The person cannot finish without seeing both halves.
	if shown.UserCode != "ABCD-1234" || shown.URL != "https://example.test/activate" {
		t.Errorf("prompt = %+v", shown)
	}
}

// Polling faster than permitted is how a provider starts refusing the attempt
// altogether, so slow_down has to actually slow it down.
func TestDeviceGrantHonoursSlowDown(t *testing.T) {
	// RFC 8628 widens the interval by five seconds, so this test necessarily
	// spends them; there is no clock to inject without making the poll loop
	// answer to a test rather than to the specification.
	if testing.Short() {
		t.Skip("waits out a widened poll interval")
	}
	var polls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/code" {
			fmt.Fprint(w, `{"device_code":"dev","user_code":"C","verification_uri":"u","expires_in":900,"interval":1}`)
			return
		}
		if polls.Add(1) == 1 {
			fmt.Fprint(w, `{"error":"slow_down"}`)
			return
		}
		fmt.Fprint(w, `{"access_token":"t"}`)
	}))
	defer server.Close()

	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if _, err := oauth.Device(ctx, oauth.Config{ClientID: "c"},
		oauth.DeviceEndpoints{Code: server.URL + "/code", Token: server.URL + "/token"}, silent()); err != nil {
		t.Fatalf("Device: %v", err)
	}
	// One second for the first poll, then at least six for the widened one.
	if elapsed := time.Since(start); elapsed < 6*time.Second {
		t.Errorf("finished in %v; slow_down did not widen the interval", elapsed)
	}
}

// A refusal is not an outage: retrying will not help, and the caller needs to
// be able to tell the two apart.
func TestDeviceGrantReportsRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/code" {
			fmt.Fprint(w, `{"device_code":"d","user_code":"C","verification_uri":"u","expires_in":900,"interval":1}`)
			return
		}
		fmt.Fprint(w, `{"error":"access_denied","error_description":"the user cancelled"}`)
	}))
	defer server.Close()

	_, err := oauth.Device(context.Background(), oauth.Config{ClientID: "c"},
		oauth.DeviceEndpoints{Code: server.URL + "/code", Token: server.URL + "/token"}, silent())
	var oerr *oauth.Error
	if !errors.As(err, &oerr) {
		t.Fatalf("err = %v, want *oauth.Error", err)
	}
	if !oerr.Denied() {
		t.Errorf("err = %v, want it reported as a refusal", err)
	}
	if !strings.Contains(err.Error(), "cancelled") {
		t.Errorf("err = %v, want the provider's message", err)
	}
}

// GitHub answers the code endpoint form-encoded unless asked otherwise, and
// the two shapes are not interchangeable.
func TestDeviceGrantAcceptsFormEncodedCodeResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/code" {
			w.Header().Set("Content-Type", "application/x-www-form-urlencoded")
			fmt.Fprint(w, "device_code=d&user_code=UC&verification_uri=https%3A%2F%2Fu&expires_in=900&interval=1")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"t"}`)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	token, err := oauth.Device(ctx, oauth.Config{ClientID: "c"},
		oauth.DeviceEndpoints{Code: server.URL + "/code", Token: server.URL + "/token"}, silent())
	if err != nil {
		t.Fatalf("Device: %v", err)
	}
	if token.Access != "t" {
		t.Errorf("Access = %q", token.Access)
	}
}

// ── authorization code grant with PKCE ──

func TestCodeGrant(t *testing.T) {
	var gotVerifier, gotCode string
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		gotVerifier, gotCode = r.Form.Get("code_verifier"), r.Form.Get("code")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"at","refresh_token":"rt","expires_in":3600}`)
	}))
	defer tokenServer.Close()

	redirect := freeLoopback(t)
	endpoints := oauth.CodeEndpoints{
		Authorize: "https://auth.test/authorize",
		Token:     tokenServer.URL,
		Redirect:  redirect,
	}

	// The Interaction stands in for the person: it reads the authorize URL and
	// completes the redirect the way a browser would.
	var challenge string
	ui := oauth.InteractionFunc(func(ctx context.Context, p oauth.Prompt) error {
		u, err := url.Parse(p.URL)
		if err != nil {
			return err
		}
		q := u.Query()
		challenge = q.Get("code_challenge")
		if q.Get("code_challenge_method") != "S256" {
			t.Errorf("challenge method = %q", q.Get("code_challenge_method"))
		}
		go func() {
			cb := redirect + "?code=the-code&state=" + url.QueryEscape(q.Get("state"))
			res, err := http.Get(cb) //nolint:noctx // test callback
			if err == nil {
				res.Body.Close()
			}
		}()
		return nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	token, err := oauth.Code(ctx, oauth.Config{ClientID: "cid", Scopes: []string{"openid"}}, endpoints, ui)
	if err != nil {
		t.Fatalf("Code: %v", err)
	}
	if token.Access != "at" || token.Refresh != "rt" {
		t.Errorf("token = %+v", token)
	}
	if token.Expires.IsZero() {
		t.Error("expires_in was not turned into a deadline")
	}
	if gotCode != "the-code" {
		t.Errorf("exchanged code = %q", gotCode)
	}
	// Only the hash travels to the authorization endpoint; the verifier is
	// sent only on the exchange, which is what makes an intercepted code
	// useless.
	if gotVerifier == "" || gotVerifier == challenge {
		t.Errorf("verifier %q must differ from challenge %q", gotVerifier, challenge)
	}
}

// Without the state check, a third party can complete somebody else's sign-in
// by sending them a crafted callback.
func TestCodeGrantRefusesAMismatchedState(t *testing.T) {
	redirect := freeLoopback(t)
	ui := oauth.InteractionFunc(func(ctx context.Context, p oauth.Prompt) error {
		go func() {
			res, err := http.Get(redirect + "?code=c&state=not-the-state") //nolint:noctx // test callback
			if err == nil {
				res.Body.Close()
			}
		}()
		return nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err := oauth.Code(ctx, oauth.Config{ClientID: "c"},
		oauth.CodeEndpoints{Authorize: "https://auth.test/a", Token: "https://auth.test/t", Redirect: redirect}, ui)

	var oerr *oauth.Error
	if !errors.As(err, &oerr) || oerr.Code != "invalid_state" {
		t.Fatalf("err = %v, want an invalid_state refusal", err)
	}
}

func TestCodeGrantReportsProviderRefusal(t *testing.T) {
	redirect := freeLoopback(t)
	ui := oauth.InteractionFunc(func(ctx context.Context, p oauth.Prompt) error {
		go func() {
			res, err := http.Get(redirect + "?error=access_denied&error_description=nope") //nolint:noctx // test callback
			if err == nil {
				res.Body.Close()
			}
		}()
		return nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err := oauth.Code(ctx, oauth.Config{ClientID: "c"},
		oauth.CodeEndpoints{Authorize: "https://a", Token: "https://t", Redirect: redirect}, ui)

	var oerr *oauth.Error
	if !errors.As(err, &oerr) || !oerr.Denied() {
		t.Fatalf("err = %v, want a refusal", err)
	}
}

// ── refresh ──

func TestRefreshCarriesForwardANonRotatingToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		if r.Form.Get("grant_type") != "refresh_token" || r.Form.Get("refresh_token") != "old" {
			t.Errorf("form = %v", r.Form)
		}
		w.Header().Set("Content-Type", "application/json")
		// This provider does not rotate refresh tokens.
		fmt.Fprint(w, `{"access_token":"new","expires_in":3600}`)
	}))
	defer server.Close()

	token, err := oauth.Refresh(context.Background(), oauth.Config{ClientID: "c"}, server.URL, "old")
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if token.Access != "new" {
		t.Errorf("Access = %q", token.Access)
	}
	// Losing it would mean the next expiry forces a fresh sign-in.
	if token.Refresh != "old" {
		t.Errorf("Refresh = %q, want the existing token carried forward", token.Refresh)
	}
}

func TestRefreshWithoutAToken(t *testing.T) {
	if _, err := oauth.Refresh(context.Background(), oauth.Config{}, "https://t", ""); err == nil {
		t.Error("expected an error with no refresh token")
	}
}

// A token that expires while a request is in flight produces a 401 that looks
// like a bad credential rather than a stale one, so validity leaves a margin.
func TestTokenValidityLeavesAMargin(t *testing.T) {
	if (oauth.Token{Access: "a", Expires: time.Now().Add(10 * time.Second)}).Valid() {
		t.Error("a token expiring in ten seconds should not be treated as usable")
	}
	if !(oauth.Token{Access: "a", Expires: time.Now().Add(time.Hour)}).Valid() {
		t.Error("a token with an hour left should be usable")
	}
	if !(oauth.Token{Access: "a"}).Valid() {
		t.Error("a token with no stated lifetime should be usable")
	}
	if (oauth.Token{}).Valid() {
		t.Error("an empty token is not usable")
	}
}

// freeLoopback reserves a port and returns a redirect URL on it.
func freeLoopback(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return fmt.Sprintf("http://127.0.0.1:%d/auth/callback", port)
}
