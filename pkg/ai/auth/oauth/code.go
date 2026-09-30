package oauth

import (
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"time"

	"golang.org/x/oauth2"
)

// Code runs the authorization code grant with PKCE. cfg.RedirectURL must be a
// loopback address the provider has registered for this client: the flow
// listens on its port for the person to come back.
func Code(ctx context.Context, cfg *oauth2.Config, ui Interaction) (*oauth2.Token, error) {
	verifier := oauth2.GenerateVerifier()
	state := oauth2.GenerateVerifier()

	redirect, err := url.Parse(cfg.RedirectURL)
	if err != nil {
		return nil, fmt.Errorf("oauth: redirect %q is not a URL: %w", cfg.RedirectURL, err)
	}
	listener, err := net.Listen("tcp", redirect.Host)
	if err != nil {
		return nil, fmt.Errorf("oauth: cannot listen on %s for the sign-in redirect "+
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

	if ui != nil {
		if err := ui.Prompt(ctx, Prompt{URL: cfg.AuthCodeURL(state, oauth2.S256ChallengeOption(verifier))}); err != nil {
			return nil, err
		}
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case result := <-results:
		if result.err != nil {
			return nil, result.err
		}
		return cfg.Exchange(ctx, result.code, oauth2.VerifierOption(verifier))
	}
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
			send(results, callbackResult{err: fmt.Errorf("oauth: the provider refused the sign-in: %s %s",
				errCode, query.Get("error_description"))})
			return
		}
		if subtle.ConstantTimeCompare([]byte(query.Get("state")), []byte(state)) != 1 {
			finish(w, "This sign-in could not be verified. You can close this window.")
			send(results, callbackResult{err: errors.New("oauth: the callback did not match this sign-in attempt")})
			return
		}
		code := query.Get("code")
		if code == "" {
			finish(w, "Sign-in returned no code. You can close this window.")
			send(results, callbackResult{err: errors.New("oauth: the callback carried no code")})
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
