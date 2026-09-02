package auth

import (
	"errors"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// Key returns the first non-empty value among a vendor's credential
// variables, and the name of the variable it came from.
func Key(v catalog.Vendor) (key, envVar string) {
	for _, name := range v.KeyEnv {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value, name
		}
	}
	return "", ""
}

// BaseURL returns the vendor's endpoint after applying its environment
// override, if one is set.
func BaseURL(v catalog.Vendor) string {
	if v.BaseURLEnv == "" {
		return v.BaseURL
	}
	return v.ResolveBaseURL(os.Getenv(v.BaseURLEnv))
}

// Deployment reads a vendor's deployment-scoped settings — the ones that name
// where a model runs rather than who is calling — into the value its driver
// expects as Config.ProtocolConfig, and fails when one the endpoint cannot run
// without is unset. It returns nil for a vendor that has none.
//
// Which variables those are, and what shape they make, is the row's business:
// this package supplies the lookup and knows nothing else about it. Deciding
// here would mean auth carrying a list of every driver's private arrangements.
func Deployment(v catalog.Vendor) (ai.ProtocolConfig, error) {
	if v.Deployment == nil {
		return nil, nil
	}
	cfg, err := v.Deployment(os.Getenv)
	var missing *catalog.MissingDeploymentError
	if errors.As(err, &missing) {
		// Reported as a missing credential because that is what it is from
		// where the caller stands: a variable they have to set before this
		// vendor works, with the same one error shape to handle.
		return nil, &MissingKeyError{Vendor: v.ID, EnvVars: missing.EnvVars, Note: missing.Note}
	}
	return cfg, err
}

// checkBaseURL fails early when a vendor that has no default endpoint was not
// told where its models live, so the caller learns which variable to set
// instead of watching their credential go to the protocol owner's host.
func checkBaseURL(v catalog.Vendor, cfg ai.Config) error {
	if !v.RequiresBaseURL || cfg.BaseURL != "" {
		return nil
	}
	return &MissingEndpointError{Vendor: v.ID, EnvVar: v.BaseURLEnv, Note: v.Note}
}
