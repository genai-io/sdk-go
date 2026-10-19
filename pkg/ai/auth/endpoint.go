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
	cfg, err := vendorConfig(v, ai.Model{})
	if err != nil {
		return nil, err
	}
	return v.Endpoint(endpoint.Config{
		APIKey:     cfg.APIKey,
		BaseURL:    cfg.Model.BaseURL,
		HTTPClient: cfg.HTTPClient,
		Vertex:     cfg.Vertex,
	}), nil
}
