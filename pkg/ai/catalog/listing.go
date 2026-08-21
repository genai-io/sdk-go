package catalog

import (
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"
)

// Reconciling a live model listing against the catalog.
//
// An endpoint is authoritative about which models exist; the catalog knows
// what they cost and what they can do. Neither alone is enough to show a
// person a model picker, so these two put them together — Enrich fills in what
// a listing left out, Missing reports what it never mentioned.

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
func Enrich(vendorID string, models []ai.Model) []ai.Model {
	v, ok := Find(vendorID)
	if !ok {
		return models
	}
	out := make([]ai.Model, len(models))
	for i, m := range models {
		// The catalog entry is the baseline and the listing is what the
		// endpoint stated. ai.MergeListing owns that rule; a vendor's
		// protocol, endpoint and dialect ride along on the baseline, because
		// no listing publishes them.
		out[i] = endpoint.MergeListing(v.Model(m.ID), m)
	}
	return out
}

// Missing returns the catalog entries a live listing did not include. A vendor
// whose endpoint lists nothing useful — several return an empty array to an
// unprivileged key — still has models worth offering.
func Missing(vendorID string, listed []ai.Model) []ai.Model {
	v, ok := Find(vendorID)
	if !ok {
		return nil
	}
	seen := make(map[string]bool, len(listed))
	for _, m := range listed {
		seen[strings.ToLower(m.ID)] = true
	}
	var out []ai.Model
	for _, m := range v.ModelList() {
		if !seen[strings.ToLower(m.ID)] {
			out = append(out, m)
		}
	}
	return out
}
