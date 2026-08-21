// Package auth resolves credentials from the environment.
//
// It is deliberately a separate import. Package llm never reads the
// environment or the filesystem, so a server handling several tenants cannot
// accidentally inherit a process-wide key; a CLI, which does want exactly
// that, opts in here:
//
//	client, err := auth.Open("anthropic/claude-opus-4-6")
//
// The variables consulted are the ones each vendor documents, listed in the
// catalog as KeyEnv and BaseURLEnv. Nothing is written back, and no credential
// file is read — an interactive login (GitHub Copilot's device code, a ChatGPT
// subscription) produces a token that the caller passes in as Config.APIKey.
package auth

import (
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
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

// Available reports the vendors that have a usable credential in the
// environment, in display order. A vendor needing no credential — a local
// Ollama — counts as available.
func Available() []catalog.Vendor {
	var out []catalog.Vendor
	store, _ := resolveStore(nil)
	for _, v := range catalog.All() {
		// A vendor that signs in through a browser is available exactly when
		// somebody has signed in.
		if _, interactive := Interactive(v.ID); interactive {
			if store != nil {
				if cred, found, err := store.Load(v.ID); err == nil && found && cred.Access != "" {
					out = append(out, v)
				}
			}
			continue
		}
		if len(v.KeyEnv) == 0 {
			out = append(out, v)
			continue
		}
		if key, _ := Key(v); key != "" {
			out = append(out, v)
		}
	}
	return out
}

// Config builds an llm.Config for a model reference, filling the credential
// and endpoint from the environment.
//
// It fails when the vendor needs a key and none of its variables are set,
// rather than sending an empty credential and surfacing the problem later as
// an opaque 401.
func Config(ref string) (llm.Config, error) {
	model, err := catalog.Model(ref)
	if err != nil {
		return llm.Config{}, err
	}
	v, ok := catalog.Find(model.Vendor)
	if !ok {
		// A hand-built or listing-derived model with no catalog vendor: there
		// is nothing to resolve, so pass it through as-is.
		return llm.Config{Model: model}, nil
	}

	if _, interactive := Interactive(v.ID); interactive {
		return interactiveConfig(v, model, nil)
	}

	key, _ := Key(v)
	if key == "" && len(v.KeyEnv) > 0 {
		return llm.Config{}, &MissingKeyError{Vendor: v.ID, EnvVars: v.KeyEnv, Note: v.Note}
	}
	cfg := llm.Config{Model: model, APIKey: key, BaseURL: BaseURL(v), Native: Deployment(v)}
	if err := checkDeployment(v, cfg); err != nil {
		return llm.Config{}, err
	}
	return cfg, nil
}

// interactiveConfig builds a Config for a vendor that signs in through a
// browser. It performs no network I/O: the endpoint was recorded at sign-in,
// and the token is minted lazily by the transport on the first request.
func interactiveConfig(v catalog.Vendor, model llm.Model, store Store) (llm.Config, error) {
	s, err := resolveStore(store)
	if err != nil {
		return llm.Config{}, err
	}
	cred, found, err := s.Load(v.ID)
	if err != nil {
		return llm.Config{}, err
	}
	if !found || cred.Access == "" {
		return llm.Config{}, &NotSignedInError{Vendor: v.ID}
	}
	client, err := HTTPClient(v.ID, cred, s, nil)
	if err != nil {
		return llm.Config{}, err
	}
	baseURL := cred.Endpoint
	if baseURL == "" {
		baseURL = v.BaseURL
	}
	// APIKey stays empty: the transport supplies a fresh token per request,
	// and a static one here would go stale mid-session.
	return llm.Config{Model: model, BaseURL: baseURL, HTTPClient: client}, nil
}

// NotSignedInError reports a vendor that needs an interactive sign-in and has
// no stored credential.
type NotSignedInError struct{ Vendor string }

func (e *NotSignedInError) Error() string {
	method, _ := Interactive(e.Vendor)
	return fmt.Sprintf("auth: not signed in to %s; run auth.Login (%s)", e.Vendor, method)
}

// Deployment reads a vendor's deployment-scoped settings — the ones that name
// where a model runs rather than who is calling — into the value its driver
// expects as Config.Native. It returns nil for a vendor that has none.
func Deployment(v catalog.Vendor) any {
	if len(v.DeploymentEnv) == 0 {
		return nil
	}
	if v.API == llm.APIAnthropicVertex {
		return llm.VertexConfig{
			Project: strings.TrimSpace(os.Getenv(v.DeploymentEnv["project"])),
			Region:  strings.TrimSpace(os.Getenv(v.DeploymentEnv["region"])),
		}
	}
	return nil
}

// checkDeployment fails early when a deployment-scoped setting a driver
// requires is missing, so the caller learns which variable to set instead of
// meeting an auth error on the first request.
func checkDeployment(v catalog.Vendor, cfg llm.Config) error {
	if v.API != llm.APIAnthropicVertex {
		return nil
	}
	if llm.ConfigNativeOf[llm.VertexConfig](cfg).Project == "" {
		return &MissingKeyError{
			Vendor:  v.ID,
			EnvVars: []string{v.DeploymentEnv["project"]},
			Note:    "Vertex needs a Google Cloud project. Credentials themselves come from Application Default Credentials, not from a variable.",
		}
	}
	return nil
}

// Open resolves a model reference from the environment and opens a client for
// it. The driver for the model's protocol must be linked in — see
// llm/driver/all.
func Open(ref string, opts ...llm.Option) (*llm.Client, error) {
	cfg, err := Config(ref)
	if err != nil {
		return nil, err
	}
	return llm.Open(cfg, opts...)
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
