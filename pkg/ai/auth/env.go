package auth

import (
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// Reading a key-based vendor's settings out of the environment.
//
// Which variables those are is catalog data, not code: a vendor entry names
// them, and everything here simply reads what it names.

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
// expects as Config.Native. It returns nil for a vendor that has none.
func Deployment(v catalog.Vendor) any {
	if len(v.DeploymentEnv) == 0 {
		return nil
	}
	if v.API == ai.APIAnthropicVertex {
		return ai.VertexConfig{
			Project: strings.TrimSpace(os.Getenv(v.DeploymentEnv["project"])),
			Region:  strings.TrimSpace(os.Getenv(v.DeploymentEnv["region"])),
		}
	}
	return nil
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

// checkDeployment fails early when a deployment-scoped setting a driver
// requires is missing, so the caller learns which variable to set instead of
// meeting an auth error on the first request.
func checkDeployment(v catalog.Vendor, cfg ai.Config) error {
	if v.API != ai.APIAnthropicVertex {
		return nil
	}
	deployment, err := ai.ConfigNativeAs[ai.VertexConfig](cfg)
	if err != nil {
		return err
	}
	if deployment.Project == "" {
		return &MissingKeyError{
			Vendor:  v.ID,
			EnvVars: []string{v.DeploymentEnv["project"]},
			Note:    "Vertex needs a Google Cloud project. Credentials themselves come from Application Default Credentials, not from a variable.",
		}
	}
	return nil
}
