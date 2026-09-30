// Package oauth runs the two interactive grants an LLM provider uses to
// authenticate a person rather than a service: the device authorization grant
// (RFC 8628) and the authorization code grant with PKCE (RFC 7636).
//
// The protocol is golang.org/x/oauth2's. What this package adds is what that
// one leaves to its caller: showing the person what to do, and receiving the
// browser when the provider sends it back. Requests go through the client
// that ctx carries under oauth2.HTTPClient, or http.DefaultClient.
package oauth

import (
	"cmp"
	"context"
	"time"

	"golang.org/x/oauth2"
)

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

// Device runs the device authorization grant: ask for a code, show it to the
// person, then poll until they have finished in a browser. cfg.Endpoint needs
// DeviceAuthURL and TokenURL.
func Device(ctx context.Context, cfg *oauth2.Config, ui Interaction) (*oauth2.Token, error) {
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
		err := ui.Prompt(ctx, Prompt{
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
