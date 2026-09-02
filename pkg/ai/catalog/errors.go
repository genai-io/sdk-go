package catalog

import (
	"fmt"
	"strings"
)

// UnknownModelError reports a bare model ID no vendor lists. Qualify it with a
// vendor ("deepseek/some-new-model") to use a model the catalog has not caught
// up with.
type UnknownModelError struct{ Ref string }

func (e *UnknownModelError) Error() string {
	return fmt.Sprintf("catalog: no vendor lists model %q; qualify it as \"vendor/%s\"", e.Ref, e.Ref)
}

// AmbiguousModelError reports a bare model ID served by more than one vendor.
type AmbiguousModelError struct {
	Ref        string
	Candidates []string
}

func (e *AmbiguousModelError) Error() string {
	return fmt.Sprintf("catalog: model %q is served by several vendors (%s); qualify it",
		e.Ref, strings.Join(e.Candidates, ", "))
}

// MissingDeploymentError reports a deployment-scoped setting a vendor cannot
// run without — a Vertex project, say. It is not a missing credential: the
// variables it names say where a model runs, not who is calling. It carries
// them rather than a finished sentence so auth, the package that actually read
// the environment, can report it in its own words.
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
