package auth

import (
	"cmp"
	"os"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// Available reports the vendors that have a usable credential in the
// environment, in display order. A vendor needing no credential — a local
// Ollama — counts as available.
func Available() []catalog.Vendor {
	var out []catalog.Vendor
	store, storeErr := resolveStore(nil)
	for _, v := range catalog.All() {
		// A vendor that signs in through a browser is available exactly when
		// somebody has signed in.
		if _, interactive := flows[v.ID]; interactive {
			if storeErr == nil {
				if cred, found, err := store.Load(v.ID); err == nil && found && cred.Access != "" {
					out = append(out, v)
				}
			}
			continue
		}
		if _, err := envConfig(v, ai.Model{}); err == nil {
			out = append(out, v)
		}
	}
	return out
}

// Config builds an ai.Config for a model reference, filling the credential
// and endpoint from the environment.
func Config(ref string) (ai.Config, error) {
	model, err := catalog.Model(ref)
	if err != nil {
		return ai.Config{}, err
	}
	v, ok := catalog.Find(model.Vendor)
	if !ok {
		// A hand-built or listing-derived model with no catalog vendor: there
		// is nothing to resolve, so pass it through as-is.
		return ai.Config{Model: model}, nil
	}
	return vendorConfig(v, model)
}

// vendorConfig resolves a catalog vendor's credential: from the sign-in for a
// vendor with an interactive flow, from the environment otherwise.
func vendorConfig(v catalog.Vendor, model ai.Model) (ai.Config, error) {
	if _, interactive := flows[v.ID]; interactive {
		return interactiveConfig(v, model)
	}
	return envConfig(v, model)
}

// envConfig resolves a key-based vendor's credential, base URL and deployment
// from the environment. The base URL is written onto the model, replacing the
// one it carries: an override in the environment is the caller redirecting it.
func envConfig(v catalog.Vendor, model ai.Model) (ai.Config, error) {
	key, err := v.KeyFromEnv(os.Getenv)
	if err != nil {
		return ai.Config{}, err
	}
	baseURL, err := v.BaseURLFromEnv(os.Getenv)
	if err != nil {
		return ai.Config{}, err
	}
	vertex, err := v.VertexFromEnv(os.Getenv)
	if err != nil {
		return ai.Config{}, err
	}
	if baseURL != "" {
		model.BaseURL = baseURL
	}
	return ai.Config{Model: model, APIKey: key, Vertex: vertex}, nil
}

// interactiveConfig builds a Config for a vendor that signs in through a
// browser. It performs no network I/O: the base URL was recorded at sign-in,
// and the token is minted lazily by the transport on the first request.
func interactiveConfig(v catalog.Vendor, model ai.Model) (ai.Config, error) {
	s, err := resolveStore(nil)
	if err != nil {
		return ai.Config{}, err
	}
	cred, found, err := s.Load(v.ID)
	if err != nil {
		return ai.Config{}, err
	}
	if !found || cred.Access == "" {
		return ai.Config{}, &NotSignedInError{Vendor: v.ID}
	}
	client, err := HTTPClient(v.ID, cred, s, nil)
	if err != nil {
		return ai.Config{}, err
	}
	// APIKey stays empty: the transport supplies a fresh token per request,
	// and a static one here would go stale mid-session.
	// The base URL the sign-in resolved to replaces the vendor's default the
	// model carries.
	if baseURL := cmp.Or(cred.BaseURL, v.BaseURL); baseURL != "" {
		model.BaseURL = baseURL
	}
	return ai.Config{Model: model, HTTPClient: client}, nil
}

// Client returns a client for a model reference, with the credential and
// endpoint resolved from the environment.
//
//	client, err := auth.Client("openai/gpt-4.1")
func Client(ref string, opts ...ai.Option) (*ai.Client, error) {
	cfg, err := Config(ref)
	if err != nil {
		return nil, err
	}
	return ai.New(cfg, opts...)
}
