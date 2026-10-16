package auth

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"
)

// Endpoint builds a live endpoint for a catalog vendor, with its credential
// and base URL resolved from the environment.
func Endpoint(vendorID string) (*endpoint.Endpoint, error) {
	v, ok := catalog.Find(vendorID)
	if !ok {
		return nil, &catalog.UnknownVendorError{Vendor: vendorID}
	}
	if _, interactive := flows[v.ID]; interactive {
		cfg, err := interactiveConfig(v, ai.Model{}, nil)
		if err != nil {
			return nil, err
		}
		return v.Endpoint(endpoint.Config{
			BaseURL:    cfg.BaseURL,
			HTTPClient: cfg.HTTPClient,
		}), nil
	}

	cfg, err := envConfig(v)
	if err != nil {
		return nil, err
	}
	return v.Endpoint(endpoint.Config{
		APIKey:  cfg.APIKey,
		BaseURL: cfg.BaseURL,
		Vertex:  cfg.Vertex,
	}), nil
}
