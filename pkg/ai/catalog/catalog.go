// Package catalog is the vendor and model directory, as data.
//
// A vendor here is a row, not a package. What distinguishes DeepSeek from
// Moonshot from Ollama is a base URL, an environment variable, a reasoning
// dialect and a list of models — never Go code — because all three serve the
// OpenAI Chat Completions protocol that one driver already speaks. Adding an
// OpenAI-compatible endpoint is an entry in vendors.go.
//
// The entries are a starting point, not an authority. Vendors ship models
// faster than any vendored table is refreshed, so an unknown model ID still
// resolves: it inherits its vendor's protocol, endpoint and dialect, and the
// driver's live listing fills in what the endpoint publishes about it.
//
// Every vendor records the date its figures were last checked against that
// vendor's documentation, because the failure mode here is silent — a
// two-year-old context window reads exactly like a fresh one. Stale reports
// entries that have aged out. Where a vendor publishes no per-model limits at
// all, the entry says so in its Note and reports zero rather than carrying a
// number nobody checked.
//
//	model, err := catalog.Model("deepseek/deepseek-v4-pro")
//	model, err := catalog.Model("claude-opus-4-6")  // unambiguous, vendor inferred
//
// A vendor is a storefront, not a protocol. "anthropic-vertex" serves Claude
// through Google Cloud and speaks Anthropic Messages; "google" serves Gemini
// and speaks generateContent. Which driver handles a model is decided by its
// API field, never by whose cloud it runs in.
//
// Nothing in this package reads the environment. KeyEnv and BaseURLEnv name
// the variables a vendor conventionally uses; package ai/auth is what reads
// them, and only if you import it.
//
// # Where things live
//
//	vendors.go   the table — one entry per vendor, and the file to edit
//	presets.go   the ladders, dialects and shorthands an entry is written in
//	infer.go     filling in what the table does not state for a model
//	vendor.go    what an entry means, and what a model inherits from it
//	catalog.go   looking a vendor or a model reference up
//	listing.go   reconciling a live endpoint listing against the table
//	errors.go    what an unresolvable reference reports
package catalog

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Stale returns the vendors whose entries have not been verified against
// their vendor's documentation within the given age, oldest first. Pass the
// current time explicitly so a caller decides what "now" means — a build
// checking freshness in CI and a runtime warning want different clocks.
//
// A vendor with no Verified date, or an unparseable one, is always reported.
func Stale(now time.Time, age time.Duration) []Vendor {
	var out []Vendor
	for _, v := range All() {
		checked, err := time.Parse("2006-01-02", v.Verified)
		if err != nil || now.Sub(checked) > age {
			out = append(out, v)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Verified < out[j].Verified })
	return out
}

// All returns every vendor, in display order.
func All() []Vendor {
	out := make([]Vendor, len(vendors))
	for i, vendor := range vendors {
		out[i] = vendor.clone()
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Order < out[j].Order })
	return out
}

// Find returns the vendor with the given ID.
func Find(id string) (Vendor, bool) {
	for _, v := range vendors {
		if strings.EqualFold(v.ID, id) {
			return v.clone(), true
		}
	}
	return Vendor{}, false
}

// Model resolves a model reference.
//
// The canonical form is "vendor/id" — note that a model ID may itself contain
// a slash ("mimo/xiaomi/mimo-v2.5-pro"), so only the first segment is the
// vendor. A bare ID is resolved by searching every vendor's catalog; that
// succeeds only when exactly one vendor lists it. An ID served by several
// vendors (deepseek-v4-flash, direct and via SenseNova) is reported as
// ambiguous rather than resolved to whichever happened to be listed first.
func Model(ref string) (ai.Model, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return ai.Model{}, fmt.Errorf("catalog: empty model reference")
	}

	if vendorID, id, ok := strings.Cut(ref, "/"); ok {
		if v, found := Find(vendorID); found {
			if id == "" {
				return ai.Model{}, fmt.Errorf("catalog: reference %q names vendor %q with no model", ref, vendorID)
			}
			return v.Model(id), nil
		}
	}

	var matches []ai.Model
	var direct []ai.Model
	for _, v := range All() {
		for _, m := range v.Models {
			if !strings.EqualFold(m.ID, ref) {
				continue
			}
			matches = append(matches, v.decorate(m))
			if !v.NeedsDeployment() {
				direct = append(direct, v.decorate(m))
			}
		}
	}
	// A vendor that needs deployment configuration — a cloud project, a region
	// — is never what a bare model name means. Someone typing
	// "claude-opus-5" wants the first-party API; reaching it through Vertex or
	// a private cloud deployment is a deliberate choice, and naming the vendor
	// is how that choice is made. Without this rule, adding one alternate
	// hosting vendor would make every model it serves ambiguous.
	if len(direct) > 0 {
		matches = direct
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return ai.Model{}, &UnknownModelError{Ref: ref}
	default:
		refs := make([]string, len(matches))
		for i, m := range matches {
			refs[i] = m.String()
		}
		return ai.Model{}, &AmbiguousModelError{Ref: ref, Candidates: refs}
	}
}

// Models returns every known model across all vendors, in vendor display
// order, retired ones included. Filter with ai.Available for a picker.
func Models() []ai.Model {
	var out []ai.Model
	for _, v := range All() {
		out = append(out, v.ModelList()...)
	}
	return out
}
