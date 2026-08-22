package auth

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// Resolving a model reference into something that can talk to a model.
//
// Every entry point here answers one question — which vendor is this, and what
// credential reaches it — and then hands off: to env.go for a key-based vendor,
// to the credential store for one that signs a person in.

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
		// A vendor with no default host is unusable until it has one, key or
		// no key: Azure and Bedrock endpoints name a tenant's own resource.
		if v.RequiresBaseURL && BaseURL(v) == "" {
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

// Config builds an ai.Config for a model reference, filling the credential
// and endpoint from the environment.
//
// It fails when the vendor needs a key and none of its variables are set,
// rather than sending an empty credential and surfacing the problem later as
// an opaque 401.
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

	if _, interactive := Interactive(v.ID); interactive {
		return interactiveConfig(v, model, nil)
	}

	key, _ := Key(v)
	if key == "" && len(v.KeyEnv) > 0 {
		return ai.Config{}, &MissingKeyError{Vendor: v.ID, EnvVars: v.KeyEnv, Note: v.Note}
	}
	cfg := ai.Config{Model: model, APIKey: key, BaseURL: BaseURL(v), ProtocolConfig: Deployment(v)}
	if err := checkBaseURL(v, cfg); err != nil {
		return ai.Config{}, err
	}
	if err := checkDeployment(v, cfg); err != nil {
		return ai.Config{}, err
	}
	return cfg, nil
}

// interactiveConfig builds a Config for a vendor that signs in through a
// browser. It performs no network I/O: the endpoint was recorded at sign-in,
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
	baseURL := cred.Endpoint
	if baseURL == "" {
		baseURL = v.BaseURL
	}
	// APIKey stays empty: the transport supplies a fresh token per request,
	// and a static one here would go stale mid-session.
	return ai.Config{Model: model, BaseURL: baseURL, HTTPClient: client}, nil
}

// Client returns a client for a model reference, with the credential and
// endpoint resolved from the environment.
//
//	client, err := auth.Client("openai/gpt-4.1")
//
// It is the one-line form of Config followed by ai.Open, and the driver for the
// model's protocol must be linked in — see ai/driver/all.
func Client(ref string, opts ...ai.Option) (*ai.Client, error) {
	cfg, err := Config(ref)
	if err != nil {
		return nil, err
	}
	return ai.Open(cfg, opts...)
}
