package auth

import (
	"cmp"
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"golang.org/x/oauth2"
)

// Flow is one vendor's interactive sign-in: the grant that authenticates a
// person, and the renewal that keeps the result usable afterwards.
type Flow struct {
	// Method describes the grant for a caller listing what is available, such
	// as "device code" or "browser (PKCE)".
	Method string

	// Login runs the grant and returns what should be stored. The client and
	// the interaction are supplied so a caller can drive the sign-in
	// somewhere other than a terminal.
	Login func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error)

	// Token returns the value to present on a request, renewing as needed. It
	// receives whatever is stored and returns whatever should replace it, so a
	// refresh token that rotated is written back rather than lost.
	Token func(ctx context.Context, client *http.Client, c Credential) (present string, expires time.Time, updated Credential, err error)
}

// flows are the interactive vendors, by vendor ID.
var flows = map[string]Flow{
	copilotVendor:  newCopilotFlow(copilotDefaults),
	"openai-codex": newCodexFlow(codexDefaults),
}

// Interactive reports whether a vendor signs in through a browser rather than
// an API key, and by which grant. The second result is empty for a vendor that
// takes an API key.
func Interactive(vendorID string) (method string, ok bool) {
	f, ok := flows[vendorID]
	if !ok {
		return "", false
	}
	return f.Method, true
}

// LoginOptions configures Login.
type LoginOptions struct {
	// Store is where the credential is kept. Nil uses the default file store.
	Store Store
	// Interaction shows the person what to do. Nil prints to stderr and tries
	// to open a browser.
	Interaction oauth.Interaction
	// HTTPClient is used for the sign-in exchanges.
	HTTPClient *http.Client
}

// Login runs a vendor's interactive sign-in and stores the result.
func Login(ctx context.Context, vendorID string, opts LoginOptions) (Credential, error) {
	f, ok := flows[vendorID]
	if !ok {
		if _, known := catalog.Find(vendorID); !known {
			return Credential{}, &UnknownVendorError{Vendor: vendorID}
		}
		return Credential{}, fmt.Errorf("auth: %s does not sign in interactively; set its API key variable instead", vendorID)
	}

	ui := opts.Interaction
	if ui == nil {
		ui = ConsoleInteraction()
	}
	client := cmp.Or(opts.HTTPClient, &http.Client{Timeout: 30 * time.Second})

	// Also on the context, which is where golang.org/x/oauth2 looks for it.
	credential, err := f.Login(context.WithValue(ctx, oauth2.HTTPClient, client), client, ui)
	if err != nil {
		return Credential{}, err
	}
	credential.Vendor = vendorID

	store, err := resolveStore(opts.Store)
	if err != nil {
		return credential, err
	}
	// Whatever was cached for this vendor is now the previous sign-in.
	forgetSources(vendorID)
	if err := store.Save(credential); err != nil {
		// The sign-in worked; only persisting it did not. Handing back the
		// credential lets the caller carry on with this session.
		return credential, err
	}
	return credential, nil
}

// Logout forgets a vendor's stored credential. It does not revoke it with the
// provider, which only the provider's own settings page can do.
func Logout(vendorID string, store Store) error {
	s, err := resolveStore(store)
	if err != nil {
		return err
	}
	forgetSources(vendorID)
	return s.Delete(vendorID)
}
