package catalog

import (
	"fmt"
	"strings"
)

// UnknownVendorError reports a vendor ID the catalog does not carry.
type UnknownVendorError struct{ Vendor string }

func (e *UnknownVendorError) Error() string {
	return fmt.Sprintf("catalog: no vendor %q", e.Vendor)
}

// MissingEnvError reports a variable a vendor cannot run without — its key,
// its host or its deployment — none of which is set.
type MissingEnvError struct {
	Vendor string
	// EnvVars are the variables that would have supplied it; any one will do.
	EnvVars []string
	// Note is the vendor's own, for what setting them does not cover.
	Note string
}

func (e *MissingEnvError) Error() string {
	msg := fmt.Sprintf("catalog: %s is not configured; set %s", e.Vendor, strings.Join(e.EnvVars, " or "))
	if e.Note != "" {
		msg += " (" + e.Note + ")"
	}
	return msg
}
