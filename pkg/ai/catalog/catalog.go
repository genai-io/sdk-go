// Package catalog is the vendor directory, as data: how to reach each vendor
// and speak its protocol — endpoint, credential variables, protocol and
// quirks. Which models a vendor serves, their limits, prices, modalities and
// the reasoning efforts they offer are the application's data; the SDK
// derives the wire value of an effort from the protocol.
//
//	model, err := catalog.Model("deepseek/deepseek-v4-pro")
//
// # Where things live
//
//	vendors.go   the table — one entry per vendor, and the file to edit
//	presets.go   the dialects and deployments an entry is written in
//	vendor.go    what an entry means, and what a model inherits from it
//	catalog.go   looking a vendor or a model reference up
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

// Find returns the vendor with the given ID, or one it used to be called.
func Find(id string) (Vendor, bool) {
	id = strings.TrimSpace(id)
	if v, ok := row(id); ok {
		return v, true
	}
	// A spelling the table has since dropped. What it resolves to still reports
	// the current vendor ID, so the old name goes no further than this lookup.
	if to, ok := aliases[strings.ToLower(id)]; ok {
		return row(to)
	}
	return Vendor{}, false
}

// row is the table lookup itself, without the aliases, so an alias can only
// ever point at a real row and never at another alias.
func row(id string) (Vendor, bool) {
	for _, v := range vendors {
		if strings.EqualFold(v.ID, id) {
			return v.clone(), true
		}
	}
	return Vendor{}, false
}

// Model resolves a "vendor/model" reference to a model carrying that
// vendor's protocol facts. The catalog lists no models, so a bare ID is an
// UnknownModelError.
func Model(ref string) (ai.Model, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return ai.Model{}, fmt.Errorf("catalog: empty model reference")
	}
	vendorID, id, ok := strings.Cut(ref, "/")
	if !ok {
		return ai.Model{}, &UnknownModelError{Ref: ref}
	}
	v, found := Find(vendorID)
	if !found {
		return ai.Model{}, &UnknownModelError{Ref: ref}
	}
	if id == "" {
		return ai.Model{}, fmt.Errorf("catalog: reference %q names vendor %q with no model", ref, vendorID)
	}
	return v.Model(id), nil
}
