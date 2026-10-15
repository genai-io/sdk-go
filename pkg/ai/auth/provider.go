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

	cfg, err := keyConfig(v)
	if err != nil {
		return nil, err
	}
	return v.Provider(provider.Config{
		APIKey:         cfg.APIKey,
		BaseURL:        cfg.BaseURL,
		ProtocolConfig: cfg.ProtocolConfig,
	}), nil
}
