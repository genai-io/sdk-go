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
