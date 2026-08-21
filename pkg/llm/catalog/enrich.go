package catalog

import (
	"strings"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Enrich fills in what a live model listing leaves out.
//
// Drivers return what their endpoint actually publishes, which for most
// OpenAI-compatible vendors is an ID and nothing else — no context window, no
// pricing, no reasoning support. Merging the catalog over that listing gives a
// model picker something to show, while the endpoint stays authoritative about
// which models exist and about any figure it did report.
//
// Models the catalog does not list still come back, decorated with their
// vendor's defaults.
func Enrich(vendorID string, models []llm.Model) []llm.Model {
	v, ok := Find(vendorID)
	if !ok {
		return models
	}
	out := make([]llm.Model, len(models))
	for i, m := range models {
		known := v.Model(m.ID)
		// The endpoint wins on anything it stated; the catalog fills the rest.
		if m.Name == "" || m.Name == m.ID {
			m.Name = known.Name
		}
		if m.ContextWindow == 0 {
			m.ContextWindow = known.ContextWindow
		}
		if m.MaxOutput == 0 {
			m.MaxOutput = known.MaxOutput
		}
		if m.Reasoning == nil {
			m.Reasoning = known.Reasoning
		}
		if !m.Pricing.Known() {
			m.Pricing = known.Pricing
		}
		if m.Input == nil {
			m.Input = known.Input
		}
		m.Vendor = v.ID
		m.API = v.API
		m.BaseURL = known.BaseURL
		m.Compat = known.Compat
		m.SamplingParams = known.SamplingParams
		m.Headers = known.Headers
		out[i] = m
	}
	return out
}

// Missing returns the catalog entries a live listing did not include. A vendor
// whose endpoint lists nothing useful — several return an empty array to an
// unprivileged key — still has models worth offering.
func Missing(vendorID string, listed []llm.Model) []llm.Model {
	v, ok := Find(vendorID)
	if !ok {
		return nil
	}
	seen := make(map[string]bool, len(listed))
	for _, m := range listed {
		seen[strings.ToLower(m.ID)] = true
	}
	var out []llm.Model
	for _, m := range v.ModelList() {
		if !seen[strings.ToLower(m.ID)] {
			out = append(out, m)
		}
	}
	return out
}
