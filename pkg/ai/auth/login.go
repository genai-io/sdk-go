package auth

import (
	"cmp"
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"golang.org/x/oauth2"
)

// flow is one vendor's interactive sign-in: the grant that authenticates a
// person, and the renewal that keeps the result usable afterwards.
type flow struct {
	// method describes the grant, such as "device code" or "browser (PKCE)".
	method string

	// login runs the grant and returns what should be stored.
	login func(ctx context.Context, client *http.Client, ui Interaction) (Credential, error)

	// token returns the value to present on a request, renewing as needed. It
	// receives whatever is stored and returns whatever should replace it, so a
	// refresh token that rotated is written back rather than lost.
	token func(ctx context.Context, client *http.Client, c Credential) (present string, expires time.Time, updated Credential, err error)

	// identify sets what the backend needs on a request beyond the token:
	// both backends serve an editor integration rather than the public API
	// and refuse a caller that does not present itself as one. session is
	// fixed for the life of one HTTPClient. Nil when nothing varies.
	identify func(h http.Header, token, session string)
}

// flows are the interactive vendors, by vendor ID.
var flows = map[string]flow{
	copilotVendor: newCopilotFlow(copilotDefaults),
	codexVendor:   newCodexFlow(codexDefaults),
}

// LoginOptions configures Login.
type LoginOptions struct {
	// Store is where the credential is kept. Nil uses the default file store.
	Store Store
	// Interaction shows the person what to do. Nil prints to stderr and tries
	// to open a browser.
	Interaction Interaction
	// HTTPClient is used for the sign-in exchanges.
	HTTPClient *http.Client
}

// Login runs a vendor's interactive sign-in and stores the result.
func Login(ctx context.Context, vendorID string, opts LoginOptions) (Credential, error) {
	f, ok := flows[vendorID]
	if !ok {
		if _, known := catalog.Find(vendorID); !known {
			return Credential{}, &catalog.UnknownVendorError{Vendor: vendorID}
		}
		return Credential{}, fmt.Errorf("auth: %s does not sign in interactively; set its API key variable instead", vendorID)
	}

	ui := opts.Interaction
	if ui == nil {
		ui = consoleInteraction
	}
	client := cmp.Or(opts.HTTPClient, &http.Client{Timeout: 30 * time.Second})

	// Also on the context, which is where golang.org/x/oauth2 looks for it.
	credential, err := f.login(context.WithValue(ctx, oauth2.HTTPClient, client), client, ui)
	if err != nil {
		return Credential{}, err
	}
	credential.Vendor = vendorID

	store, err := resolveStore(opts.Store)
	if err != nil {
		return credential, err
	}
	// Whatever was cached for this vendor is now the previous sign-in.
	if err := withRefreshLock(ctx, vendorID, store, func() error {
		forgetSources(vendorID)
		return store.Save(credential)
	}); err != nil {
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
	return withRefreshLock(context.Background(), vendorID, s, func() error {
		forgetSources(vendorID)
		return s.Delete(vendorID)
	})
}
