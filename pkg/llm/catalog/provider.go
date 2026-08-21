package catalog

import "github.com/genai-io/sdk-go/pkg/llm"

// Provider builds a runtime provider for this vendor, seeded with its catalog
// models as the static baseline.
//
// The caller supplies the credential and any transport settings; the vendor
// supplies its identity, protocol, endpoint and models. Fields the caller
// leaves unset fall back to the vendor's — so passing a zero ProviderConfig
// yields a provider that can already list and open models, just without a key.
func (v Vendor) Provider(cfg llm.ProviderConfig) *llm.Provider {
	cfg.ID = v.ID
	if cfg.Name == "" {
		cfg.Name = v.DisplayName
	}
	cfg.API = v.API
	if cfg.BaseURL == "" {
		cfg.BaseURL = v.BaseURL
	}
	if cfg.Models == nil {
		cfg.Models = v.ModelList()
	}
	if cfg.Headers == nil {
		cfg.Headers = v.Headers
	}
	return llm.NewProvider(cfg)
}
