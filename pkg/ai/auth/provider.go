package auth

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"github.com/genai-io/sdk-go/pkg/ai/provider"
)

// Provider builds a live provider for a catalog vendor, with its credential
// and base URL resolved from the environment.
func Provider(vendorID string) (*provider.Provider, error) {
	v, ok := catalog.Find(vendorID)
	if !ok {
		return nil, &UnknownVendorError{Vendor: vendorID}
	}
	if _, interactive := Interactive(v.ID); interactive {
		cfg, err := interactiveConfig(v, ai.Model{}, nil)
		if err != nil {
			return nil, err
		}
		return v.Provider(provider.Config{
			BaseURL:    cfg.BaseURL,
			HTTPClient: cfg.HTTPClient,
		}), nil
	}

	key, _ := Key(v)
	if key == "" && len(v.KeyEnv) > 0 {
		return nil, &MissingKeyError{Vendor: v.ID, EnvVars: v.KeyEnv, Note: v.Note}
	}
	return v.Provider(provider.Config{
		APIKey:         key,
		BaseURL:        BaseURL(v),
		ProtocolConfig: Deployment(v),
	}), nil
}

// Providers builds a live provider for every vendor with a usable credential, in
// catalog display order.
func Providers() *provider.Set {
	out := provider.NewSet()
	for _, v := range Available() {
		if e, err := Provider(v.ID); err == nil {
			out.Set(e)
		}
	}
	return out
}
