package oauth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// CodeEndpoints are the two URLs an authorization code grant uses —
// RFC 6749 §4.1, with PKCE from RFC 7636.
type CodeEndpoints struct {
	// Authorize is the page the person is sent to.
	Authorize string
	// Token exchanges the returned code for a token.
	Token string
	// Redirect is where the provider sends the person back. It must be one
	// the provider has registered, and for a public client it is a loopback
	// address whose port this flow listens on.
	Redirect string
}

// Code runs the authorization code grant with PKCE.
func Code(ctx context.Context, cfg Config, endpoints CodeEndpoints, ui Interaction) (Token, error) {
	verifier, challenge, err := pkcePair()
	if err != nil {
		return Token{}, err
	}
	state, err := randomString(32)
	if err != nil {
		return Token{}, err
	}

	redirect, err := url.Parse(endpoints.Redirect)
	if err != nil {
		return Token{}, fmt.Errorf("oauth: redirect %q is not a URL: %w", endpoints.Redirect, err)
	}
	listener, err := net.Listen("tcp", redirect.Host)
	if err != nil {
		return Token{}, fmt.Errorf("oauth: cannot listen on %s for the sign-in redirect "+
			"(another process may be holding it): %w", redirect.Host, err)
	}
	// The deferred Shutdown below is what actually stops serving; closing the
	// listener after it is belt and braces, with nothing left to report.
	defer func() { _ = listener.Close() }()

	results := make(chan callbackResult, 1)
	server := &http.Server{
		Handler:           callbackHandler(redirect.Path, state, results),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go server.Serve(listener) //nolint:errcheck // Shutdown below reports the real outcome
	defer func() {
		shutdown, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()

	authorize := authorizeURL(cfg, endpoints, challenge, state)
	if ui != nil {
		if err := ui.Prompt(ctx, Prompt{URL: authorize}); err != nil {
			return Token{}, err
		}
	}

	select {
	case <-ctx.Done():
		return Token{}, ctx.Err()
	case result := <-results:
		if result.err != nil {
			return Token{}, result.err
		}
		return exchange(ctx, cfg, endpoints, result.code, verifier)
	}
}

// AuthorizeURL builds the page a person is sent to, for a caller driving the
// redirect itself — a web application that already has a callback route, and
// does not want this package opening a listener.
func AuthorizeURL(cfg Config, endpoints CodeEndpoints, state string) (authorize, verifier string, err error) {
	verifier, challenge, err := pkcePair()
	if err != nil {
		return "", "", err
	}
	return authorizeURL(cfg, endpoints, challenge, state), verifier, nil
}

// Exchange completes a grant whose redirect the caller handled itself.
func Exchange(ctx context.Context, cfg Config, endpoints CodeEndpoints, code, verifier string) (Token, error) {
	return exchange(ctx, cfg, endpoints, code, verifier)
}

func authorizeURL(cfg Config, endpoints CodeEndpoints, challenge, state string) string {
	query := url.Values{
		"response_type":         {"code"},
		"client_id":             {cfg.ClientID},
		"redirect_uri":          {endpoints.Redirect},
		"state":                 {state},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
	}
	if len(cfg.Scopes) > 0 {
		query.Set("scope", strings.Join(cfg.Scopes, " "))
	}
	sep := "?"
	if strings.Contains(endpoints.Authorize, "?") {
		sep = "&"
	}
	return endpoints.Authorize + sep + query.Encode()
}

func exchange(ctx context.Context, cfg Config, endpoints CodeEndpoints, code, verifier string) (Token, error) {
	res, err := postForm(ctx, cfg.client(), endpoints.Token, url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {endpoints.Redirect},
		"client_id":     {cfg.ClientID},
		"code_verifier": {verifier},
	})
	if err != nil {
		return Token{}, err
	}
	return res.token(), nil
}

type callbackResult struct {
	code string
	err  error
}

// callbackHandler receives the person back from the provider.
func callbackHandler(path, state string, results chan<- callbackResult) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if path != "" && r.URL.Path != path {
			http.NotFound(w, r)
			return
		}
		query := r.URL.Query()

		if errCode := query.Get("error"); errCode != "" {
			finish(w, "Sign-in was not completed. You can close this window.")
			send(results, callbackResult{err: &Error{
				Code:        errCode,
				Description: query.Get("error_description"),
			}})
			return
		}
		if subtle.ConstantTimeCompare([]byte(query.Get("state")), []byte(state)) != 1 {
			finish(w, "This sign-in could not be verified. You can close this window.")
			send(results, callbackResult{err: &Error{
				Code:        "invalid_state",
				Description: "the callback did not match this sign-in attempt",
			}})
			return
		}
		code := query.Get("code")
		if code == "" {
			finish(w, "Sign-in returned no code. You can close this window.")
			send(results, callbackResult{err: &Error{Code: "invalid_request", Description: "no code in the callback"}})
			return
		}
		finish(w, "Signed in. You can close this window and return to your terminal.")
		send(results, callbackResult{code: code})
	})
}

// send never blocks: the listener may still receive a favicon request or a
// reload after the first result has been taken.
func send(results chan<- callbackResult, r callbackResult) {
	select {
	case results <- r:
	default:
	}
}

func finish(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// A browser that hung up mid-write has still delivered the code; the
	// grant reports through the results channel, not through this page.
	_, _ = fmt.Fprintf(w, "<!doctype html><meta charset=utf-8><title>Sign-in</title>"+
		"<body style=\"font:16px system-ui;padding:3rem\"><p>%s</p>", message)
}

// pkcePair generates a verifier and its S256 challenge.
func pkcePair() (verifier, challenge string, err error) {
	verifier, err = randomString(64)
	if err != nil {
		return "", "", err
	}
	sum := sha256.Sum256([]byte(verifier))
	return verifier, base64.RawURLEncoding.EncodeToString(sum[:]), nil
}

// randomString returns n bytes of entropy as unreserved base64url characters,
// which is what the specification requires of a verifier.
func randomString(n int) (string, error) {
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return "", errors.New("oauth: no entropy available")
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}
