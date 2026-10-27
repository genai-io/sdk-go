package auth

import (
	"context"
	"time"
)

// Credential is what was obtained for a vendor, and what has to survive
// between runs so a person signs in once rather than every time.
type Credential struct {
	// Vendor is the catalog vendor this belongs to.
	Vendor string `json:"vendor"`
	// Access is the token to present, or the API key for a key-based vendor.
	Access string `json:"access"`
	// Refresh renews Access without another sign-in. Empty for a token that
	// does not expire, and for an API key.
	Refresh string `json:"refresh,omitempty"`
	// ExpiresAt is when Access stops working. Zero means it does not.
	ExpiresAt time.Time `json:"expires_at,omitempty"`
	// BaseURL is the base URL the sign-in resolved to, for a vendor that
	// tells you where to talk to it only after you have authenticated.
	BaseURL string `json:"base_url,omitempty"`
}

// Expired reports whether the credential needs renewing.
func (c Credential) Expired() bool { return expired(c.ExpiresAt) }

// expired is the one expiry rule in this package, so a credential cannot be
// fresh by one caller's reckoning and stale by another's. A zero time never
// expires; anything else is gone expiryMargin early.
func expired(at time.Time) bool {
	return !at.IsZero() && !time.Now().Add(expiryMargin).Before(at)
}

// expiryMargin is how early a token is treated as expired. A request that
// starts with thirty seconds left can still finish after it has run out, and
// the resulting 401 looks like a bad credential rather than a stale one.
const expiryMargin = 60 * time.Second

// Store keeps credentials between runs.
type Store interface {
	// Load returns the credential kept for a vendor. The second result is
	// false when there is none, which is not an error — nobody has signed in
	// yet is the normal state.
	Load(vendor string) (Credential, bool, error)

	// Save keeps a credential, replacing any held for the same vendor. A
	// credential with no Vendor is an error: it could never be loaded back,
	// so accepting it would silently lose the sign-in it represents.
	Save(c Credential) error

	// Delete forgets a vendor's credential. Deleting one that is not there
	// succeeds.
	Delete(vendor string) error
}

// RefreshLocker is an optional Store capability for credentials shared across
// processes. WithRefreshLock must serialize fn with other holders for the same
// vendor, and allow fn to call Load and Save without taking this lock again.
type RefreshLocker interface {
	WithRefreshLock(ctx context.Context, vendor string, fn func() error) error
}

func withRefreshLock(ctx context.Context, vendor string, store Store, fn func() error) error {
	if locker, ok := store.(RefreshLocker); ok {
		return locker.WithRefreshLock(ctx, vendor, fn)
	}
	return fn()
}

// resolveStore is s, or the default file store when s is nil.
func resolveStore(s Store) (Store, error) {
	if s != nil {
		return s, nil
	}
	return newFileStore("")
}
