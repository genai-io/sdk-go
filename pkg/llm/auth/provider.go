package auth

import (
	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
)

// Provider builds a runtime provider for a catalog vendor, with its credential
// and endpoint resolved from the environment.
//
// It fails when the vendor needs a key and none of its variables are set,
// rather than handing back a provider that will 401 on first use.
func Provider(vendorID string) (*llm.Provider, error) {
	v, ok := catalog.Find(vendorID)
	if !ok {
		return nil, &UnknownVendorError{Vendor: vendorID}
	}
	if _, interactive := Interactive(v.ID); interactive {
		cfg, err := interactiveConfig(v, llm.Model{}, nil)
		if err != nil {
			return nil, err
		}
		return v.Provider(llm.ProviderConfig{
			BaseURL:    cfg.BaseURL,
			HTTPClient: cfg.HTTPClient,
		}), nil
	}

	key, _ := Key(v)
	if key == "" && len(v.KeyEnv) > 0 {
		return nil, &MissingKeyError{Vendor: v.ID, EnvVars: v.KeyEnv, Note: v.Note}
	}
	return v.Provider(llm.ProviderConfig{
		APIKey:  key,
		BaseURL: BaseURL(v),
		Native:  Deployment(v),
	}), nil
}

// Providers builds a provider for every vendor with a usable credential, in
// catalog display order.
//
// Vendors with no credential are skipped rather than reported: this is the
// "what can I actually use right now" question, and a missing key is the
// normal answer for most of the catalog.
func Providers() *llm.Providers {
	out := llm.NewProviders()
	for _, v := range Available() {
		if p, err := Provider(v.ID); err == nil {
			out.Set(p)
		}
	}
	return out
}

// UnknownVendorError reports a vendor ID the catalog does not carry.
type UnknownVendorError struct{ Vendor string }

func (e *UnknownVendorError) Error() string {
	return "auth: no catalog vendor " + e.Vendor
}
