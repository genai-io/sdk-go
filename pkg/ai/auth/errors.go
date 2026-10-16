package auth

import (
	"errors"
	"fmt"
)

// NotSignedInError reports a vendor that needs an interactive sign-in and has
// no stored credential.
type NotSignedInError struct{ Vendor string }

func (e *NotSignedInError) Error() string {
	return fmt.Sprintf("auth: not signed in to %s; run auth.Login (%s)", e.Vendor, flows[e.Vendor].method)
}

// errNoVendor reports a credential that names no vendor. Every Store rejects
// one: it is keyed by vendor, so a credential without one could never be
// loaded back, and accepting it would silently lose a sign-in.
var errNoVendor = errors.New("auth: credential has no vendor")
