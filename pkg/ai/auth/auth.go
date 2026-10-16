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
		if _, err := envConfig(v); err == nil {
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

	if _, interactive := flows[v.ID]; interactive {
		return interactiveConfig(v, model, nil)
	}

	cfg, err := envConfig(v)
	if err != nil {
		return ai.Config{}, err
	}
	cfg.Model = model
	return cfg, nil
}

// envConfig resolves a key-based vendor's credential, base URL and deployment
// from the environment.
func envConfig(v catalog.Vendor) (ai.Config, error) {
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
	return ai.Config{APIKey: key, BaseURL: baseURL, Vertex: vertex}, nil
}

// interactiveConfig builds a Config for a vendor that signs in through a
// browser. It performs no network I/O: the base URL was recorded at sign-in,
// and the token is minted lazily by the transport on the first request.
func interactiveConfig(v catalog.Vendor, model ai.Model, store Store) (ai.Config, error) {
	s, err := resolveStore(store)
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
	return ai.Config{Model: model, BaseURL: cmp.Or(cred.BaseURL, v.BaseURL), HTTPClient: client}, nil
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
