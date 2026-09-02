// Package oauth implements the two interactive grants an LLM provider uses to
// authenticate a person rather than a service: the device authorization grant
// (RFC 8628) and the authorization code grant with PKCE (RFC 7636).
//
// # Where things live
//
//	oauth.go   what both grants share: tokens, config, errors, refresh
//	device.go  the device authorization grant, for a machine with no browser
//	code.go    the authorization code grant with PKCE, for one with a browser
package oauth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Token is what a grant produces.
type Token struct {
	Access  string    `json:"access_token"`
	Refresh string    `json:"refresh_token,omitempty"`
	Type    string    `json:"token_type,omitempty"`
	Expires time.Time `json:"expires_at,omitempty"`
	// Scope is what the provider actually granted, which can be narrower than
	// what was asked for.
	Scope string `json:"scope,omitempty"`
}

// Valid reports whether the token can still be used, leaving ExpiryMargin so a
// request does not start with a token that expires while in flight.
func (t Token) Valid() bool {
	if t.Access == "" {
		return false
	}
	if t.Expires.IsZero() {
		return true // no stated lifetime: the provider will say when it stops working
	}
	return time.Now().Add(ExpiryMargin).Before(t.Expires)
}

// ExpiryMargin is how early a token is treated as expired. A request that
// starts with thirty seconds left can still finish after it has run out, and
// the resulting 401 looks like a bad credential rather than a stale one.
//
// It is exported because package auth applies the same margin to a stored
// credential, and two packages disagreeing about it is unreproducible.
const ExpiryMargin = 60 * time.Second

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
type Interaction interface {
	Prompt(ctx context.Context, p Prompt) error
}

// InteractionFunc adapts a function to Interaction.
type InteractionFunc func(ctx context.Context, p Prompt) error

func (f InteractionFunc) Prompt(ctx context.Context, p Prompt) error { return f(ctx, p) }

// Config identifies the client and the endpoints a flow talks to.
type Config struct {
	// ClientID is the public identifier the provider issued for this
	// application. These grants are for public clients, so there is no secret.
	ClientID string
	// Scopes are requested at authorization time.
	Scopes []string
	// HTTPClient is used for every request. Nil means http.DefaultClient.
	HTTPClient *http.Client
}

func (c Config) client() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

// Error is a failed OAuth exchange, carrying the provider's own error code so
// a caller can tell a denial from an outage.
type Error struct {
	// Code is the OAuth error code, e.g. "access_denied", "expired_token".
	Code string
	// Description is the provider's message, when it sent one.
	Description string
	// Status is the HTTP status, or 0.
	Status int
}

func (e *Error) Error() string {
	msg := "oauth: " + e.Code
	if e.Description != "" {
		msg += ": " + e.Description
	}
	if e.Status != 0 {
		msg += fmt.Sprintf(" (http %d)", e.Status)
	}
	return msg
}

// Denied reports whether the person refused, rather than something going
// wrong. Retrying will not help.
func (e *Error) Denied() bool {
	return e.Code == "access_denied" || e.Code == "expired_token"
}

// tokenResponse is the wire shape of a token endpoint's reply, in both the
// success and failure cases.
type tokenResponse struct {
	AccessToken      string `json:"access_token"`
	RefreshToken     string `json:"refresh_token"`
	TokenType        string `json:"token_type"`
	ExpiresIn        int64  `json:"expires_in"`
	Scope            string `json:"scope"`
	Error            string `json:"error"`
	ErrorDescription string `json:"error_description"`
}

func (r tokenResponse) token() Token {
	t := Token{
		Access:  r.AccessToken,
		Refresh: r.RefreshToken,
		Type:    r.TokenType,
		Scope:   r.Scope,
	}
	if r.ExpiresIn > 0 {
		t.Expires = time.Now().Add(time.Duration(r.ExpiresIn) * time.Second)
	}
	return t
}

// postForm sends a form-encoded request and decodes the reply.
func postForm(ctx context.Context, client *http.Client, endpoint string, form url.Values) (tokenResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return tokenResponse{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	res, err := client.Do(req)
	if err != nil {
		return tokenResponse{}, err
	}
	// Closing a body that has been read out is a formality; there is nothing
	// to report and nothing to do about it.
	defer func() { _ = res.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return tokenResponse{}, err
	}

	var out tokenResponse
	if err := json.Unmarshal(body, &out); err != nil {
		// A provider that answered with something else — an HTML error page, a
		// gateway message — is reported with its status rather than as a JSON
		// parse failure, which would point at the wrong thing.
		return tokenResponse{}, &Error{
			Code:        "invalid_response",
			Description: truncate(string(body), 200),
			Status:      res.StatusCode,
		}
	}
	if out.Error != "" {
		return out, &Error{Code: out.Error, Description: out.ErrorDescription, Status: res.StatusCode}
	}
	if res.StatusCode >= 400 {
		return out, &Error{Code: "http_error", Description: truncate(string(body), 200), Status: res.StatusCode}
	}
	return out, nil
}

// Refresh exchanges a refresh token for a new access token.
func Refresh(ctx context.Context, cfg Config, endpoint, refreshToken string) (Token, error) {
	if refreshToken == "" {
		return Token{}, &Error{Code: "invalid_grant", Description: "no refresh token stored"}
	}
	res, err := postForm(ctx, cfg.client(), endpoint, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {cfg.ClientID},
	})
	if err != nil {
		return Token{}, err
	}
	token := res.token()
	// A provider that rotates refresh tokens sends a new one; one that does
	// not expects the old to keep working, so it is carried forward rather
	// than lost.
	if token.Refresh == "" {
		token.Refresh = refreshToken
	}
	return token, nil
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func atoiDefault(s string, fallback int) int {
	if n, err := strconv.Atoi(s); err == nil && n > 0 {
		return n
	}
	return fallback
}
