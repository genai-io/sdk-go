package auth

import (
	"cmp"
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

// The two interactive grants an LLM provider uses to authenticate a person
// rather than a service: the device authorization grant (RFC 8628) and the
// authorization code grant with PKCE (RFC 7636). The protocol is
// golang.org/x/oauth2's; what is added here is showing the person what to do,
// and receiving the browser when the provider sends it back. Requests go
// through the client that ctx carries under oauth2.HTTPClient.

// Prompt is what a person has to do to finish signing in: a page to open and,
// for the device grant, a code to type into it.
type Prompt struct {
	// URL is the page to open.
	URL string
	// UserCode is the code to enter there. Empty for a browser-redirect flow,
	// where opening the URL is the whole instruction.
	UserCode string
	// ExpiresAt is when the attempt stops being accepted.
	ExpiresAt time.Time
}

// Interaction shows a Prompt to whoever is signing in.
type Interaction func(ctx context.Context, p Prompt) error

// deviceGrant runs the device authorization grant: ask for a code, show it to the
// person, then poll until they have finished in a browser. cfg.Endpoint needs
// DeviceAuthURL and TokenURL.
func deviceGrant(ctx context.Context, cfg *oauth2.Config, ui Interaction) (*oauth2.Token, error) {
	code, err := cfg.DeviceAuth(ctx)
	if err != nil {
		return nil, err
	}
	// A provider that states no lifetime gets the specification's own example
	// rather than a poll that never ends.
	if code.Expiry.IsZero() {
		code.Expiry = time.Now().Add(15 * time.Minute)
	}
	if ui != nil {
		// The complete form embeds the code, so a person who can open it does
		// not have to type anything.
		err := ui(ctx, Prompt{
			URL:       cmp.Or(code.VerificationURIComplete, code.VerificationURI),
			UserCode:  code.UserCode,
			ExpiresAt: code.Expiry,
		})
		if err != nil {
			return nil, err
		}
	}
	return cfg.DeviceAccessToken(ctx, code)
}

// codeGrant runs the authorization code grant with PKCE. cfg.RedirectURL must be a
// loopback address the provider has registered for this client: the flow
// listens on its port for the person to come back.
func codeGrant(ctx context.Context, cfg *oauth2.Config, ui Interaction) (*oauth2.Token, error) {
	verifier := oauth2.GenerateVerifier()
	state := oauth2.GenerateVerifier()

	redirect, err := url.Parse(cfg.RedirectURL)
	if err != nil {
		return nil, fmt.Errorf("auth: redirect %q is not a URL: %w", cfg.RedirectURL, err)
	}
	listener, err := net.Listen("tcp", redirect.Host)
	if err != nil {
		return nil, fmt.Errorf("auth: cannot listen on %s for the sign-in redirect "+
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
		if err := ui(ctx, Prompt{URL: cfg.AuthCodeURL(state, oauth2.S256ChallengeOption(verifier))}); err != nil {
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
			send(results, callbackResult{err: fmt.Errorf("auth: the provider refused the sign-in: %s %s",
				errCode, query.Get("error_description"))})
			return
		}
		if subtle.ConstantTimeCompare([]byte(query.Get("state")), []byte(state)) != 1 {
			finish(w, "This sign-in could not be verified. You can close this window.")
			send(results, callbackResult{err: errors.New("auth: the callback did not match this sign-in attempt")})
			return
		}
		code := query.Get("code")
		if code == "" {
			finish(w, "Sign-in returned no code. You can close this window.")
			send(results, callbackResult{err: errors.New("auth: the callback carried no code")})
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
