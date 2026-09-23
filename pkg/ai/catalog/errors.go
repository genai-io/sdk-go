package catalog

import (
	"fmt"
	"strings"
)

// UnknownModelError reports a model reference that names no vendor. The
// catalog lists vendors, not models, so a reference is "vendor/model".
type UnknownModelError struct{ Ref string }

func (e *UnknownModelError) Error() string {
	return fmt.Sprintf("catalog: %q names no vendor; write it as \"vendor/%s\"", e.Ref, e.Ref)
}

// MissingDeploymentError reports a deployment-scoped setting a vendor cannot
// run without — a Vertex project, say. It is not a missing credential: the
// variables it names say where a model runs, not who is calling. It carries
// them rather than a finished sentence, so auth reports it in its own words.
type MissingDeploymentError struct {
	// EnvVars are the variables that would have supplied the setting.
	EnvVars []string
	// Note is anything the caller has to know beyond setting them.
	Note string
}

func (e *MissingDeploymentError) Error() string {
	msg := "catalog: this vendor needs a deployment; set " + strings.Join(e.EnvVars, " or ")
	if e.Note != "" {
		msg += " (" + e.Note + ")"
	}
	return msg
}
