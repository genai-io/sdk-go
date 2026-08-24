package auth

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// A few vendors authenticate a person rather than a service: there is no API
// key to paste, only a subscription and a browser. Without an interactive
// sign-in those vendors are unreachable, which is why their catalog entries
// used to say "supply the exchanged token yourself".

// flow is one vendor's interactive sign-in.
type flow struct {
	// method describes the grant for a caller listing what is available.
	method string
	// login runs the grant and returns what should be stored.
	login func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error)
	// token returns the value to present on a request, renewing as needed.
	token func(ctx context.Context, client *http.Client, c Credential) (present string, expires time.Time, updated Credential, err error)
}

var flows = map[string]flow{
	"copilot":      copilotFlow,
	"openai-codex": codexFlow,
}

// Interactive reports whether a vendor signs in through a browser rather than
// an API key, and by which grant. The second result is empty for a vendor that
// takes an API key.
func Interactive(vendorID string) (method string, ok bool) {
	f, ok := flows[vendorID]
	if !ok {
		return "", false
	}
	return f.method, true
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
	client := opts.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}

	credential, err := f.login(ctx, client, ui)
	if err != nil {
		return Credential{}, err
	}
	credential.Vendor = vendorID

	store, err := resolveStore(opts.Store)
	if err != nil {
		return credential, err
	}
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
	return s.Delete(vendorID)
}
