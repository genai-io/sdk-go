package auth

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"github.com/genai-io/sdk-go/pkg/ai/provider"
)

// Provider builds a live endpoint for a catalog vendor, with its credential
// and endpoint resolved from the environment.
//
// It fails when the vendor needs a key and none of its variables are set,
// rather than handing back an endpoint that will 401 on first use.
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
		APIKey:  key,
		BaseURL: BaseURL(v),
		Native:  Deployment(v),
	}), nil
}

// Providers builds a live endpoint for every vendor with a usable credential, in
// catalog display order.
//
// Vendors with no credential are skipped rather than reported: this is the
// "what can I actually use right now" question, and a missing key is the
// normal answer for most of the catalog.
func Providers() *provider.Set {
	out := provider.NewSet()
	for _, v := range Available() {
		if e, err := Provider(v.ID); err == nil {
			out.Set(e)
		}
	}
	return out
}
