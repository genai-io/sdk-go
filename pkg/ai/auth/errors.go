package auth

import (
	"fmt"
	"strings"
)

// NotSignedInError reports a vendor that needs an interactive sign-in and has
// no stored credential.
type NotSignedInError struct{ Vendor string }

func (e *NotSignedInError) Error() string {
	method, _ := Interactive(e.Vendor)
	return fmt.Sprintf("auth: not signed in to %s; run auth.Login (%s)", e.Vendor, method)
}

// MissingKeyError reports that none of a vendor's credential variables is set.
type MissingKeyError struct {
	Vendor  string
	EnvVars []string
	Note    string
}

func (e *MissingKeyError) Error() string {
	msg := fmt.Sprintf("auth: no credential for %s; set %s", e.Vendor, strings.Join(e.EnvVars, " or "))
	if e.Note != "" {
		msg += " (" + e.Note + ")"
	}
	return msg
}

// MissingEndpointError reports a vendor whose endpoint variable is not set.
type MissingEndpointError struct {
	Vendor string
	EnvVar string
	Note   string
}

func (e *MissingEndpointError) Error() string {
	msg := fmt.Sprintf("auth: no endpoint for %s; set %s", e.Vendor, e.EnvVar)
	if e.Note != "" {
		msg += " (" + e.Note + ")"
	}
	return msg
}

// UnknownVendorError reports a vendor ID the catalog does not carry.
type UnknownVendorError struct{ Vendor string }

func (e *UnknownVendorError) Error() string {
	return "auth: no catalog vendor " + e.Vendor
}

// errNoVendor reports a credential that names no vendor. Every Store rejects
// one: it is keyed by vendor, so a credential without one could never be
// loaded back, and accepting it would silently lose a sign-in.
func errNoVendor() error {
	return fmt.Errorf("auth: credential has no vendor")
}
