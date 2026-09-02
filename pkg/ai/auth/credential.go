package auth

import (
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
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
	// Endpoint is the base URL the sign-in resolved to, for a vendor that
	// tells you where to talk to it only after you have authenticated.
	Endpoint string `json:"endpoint,omitempty"`
}

// Expired reports whether the credential needs renewing.
func (c Credential) Expired() bool { return expired(c.ExpiresAt) }

// expired is the one expiry rule in this package, so a credential cannot be
// fresh by one caller's reckoning and stale by another's. A zero time never
// expires; anything else is treated as gone oauth.ExpiryMargin early, because
// a request that starts with seconds left still finishes after they run out
// and comes back as a 401 that reads like a bad credential.
func expired(at time.Time) bool {
	return !at.IsZero() && !time.Now().Add(oauth.ExpiryMargin).Before(at)
}

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

	// List names the vendors with a stored credential, in no particular order.
	List() ([]string, error)
}

// DefaultStore is where Login, Config, and Endpoint keep credentials when
// they are not handed a store.
var DefaultStore Store

func resolveStore(s Store) (Store, error) {
	if s != nil {
		return s, nil
	}
	if DefaultStore != nil {
		return DefaultStore, nil
	}
	return NewFileStore("")
}
