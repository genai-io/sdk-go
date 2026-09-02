package auth

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// Flow is one vendor's interactive sign-in: the grant that authenticates a
// person, and the renewal that keeps the result usable afterwards.
//
// The set of interactive vendors is open on purpose: this package ships the two
// grants it knows about, and anyone whose vendor signs in some third way
// registers their own rather than waiting for this package to hear of it.
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

// flows is the registry of interactive vendors.
var flows struct {
	mu sync.RWMutex
	m  map[string]Flow
}

// RegisterFlow declares how a vendor signs in. The two built-in grants register
// themselves from init, as driver packages do with ai.RegisterAPI; anyone else
// calls this. Registering one vendor twice panics: one flow would be dead.
func RegisterFlow(vendorID string, f Flow) {
	if vendorID == "" {
		panic("auth: RegisterFlow with no vendor ID")
	}
	if f.Login == nil || f.Token == nil {
		panic("auth: RegisterFlow for " + vendorID + " needs both Login and Token")
	}
	flows.mu.Lock()
	defer flows.mu.Unlock()
	if flows.m == nil {
		flows.m = make(map[string]Flow)
	}
	if _, dup := flows.m[vendorID]; dup {
		panic("auth: RegisterFlow called twice for " + vendorID)
	}
	flows.m[vendorID] = f
}

// flowFor returns a vendor's registered sign-in.
func flowFor(vendorID string) (Flow, bool) {
	flows.mu.RLock()
	defer flows.mu.RUnlock()
	f, ok := flows.m[vendorID]
	return f, ok
}

// Interactive reports whether a vendor signs in through a browser rather than
// an API key, and by which grant. The second result is empty for a vendor that
// takes an API key.
func Interactive(vendorID string) (method string, ok bool) {
	f, ok := flowFor(vendorID)
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
	f, ok := flowFor(vendorID)
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

	credential, err := f.Login(ctx, client, ui)
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
