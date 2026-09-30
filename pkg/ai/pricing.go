package ai

import "cmp"

// Pricing is a published rate card, per million tokens.
type Pricing struct {
	Currency   string  `json:"currency,omitempty"`
	Input      float64 `json:"input,omitempty"`
	Output     float64 `json:"output,omitempty"`
	CacheWrite float64 `json:"cache_write,omitempty"`
	CacheRead  float64 `json:"cache_read,omitempty"`

	// Tiers are request-wide rate switches. The highest tier whose threshold
	// the prompt exceeds takes over for the whole request — MiniMax bills M3 at
	// double above 512k input tokens, and a flat card cannot say so.
	Tiers []PricingTier `json:"tiers,omitempty"`
}

// PricingTier is a rate card that takes over above a prompt-size threshold. It
// states only the rates it changes: one left at zero keeps the base card's, so
// a rate that genuinely becomes zero cannot be said this way. None is published.
type PricingTier struct {
	// AboveInputTokens is the total input token count this tier applies past.
	AboveInputTokens int     `json:"above_input_tokens"`
	Input            float64 `json:"input,omitempty"`
	Output           float64 `json:"output,omitempty"`
	CacheWrite       float64 `json:"cache_write,omitempty"`
	CacheRead        float64 `json:"cache_read,omitempty"`
}

// USD is the currency code the catalog prices in.
const USD = "USD"

// Cost is the money breakdown of one call, in Pricing's currency.
type Cost struct {
	Currency   string  `json:"currency,omitempty"`
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheWrite float64 `json:"cache_write"`
	CacheRead  float64 `json:"cache_read"`
	Total      float64 `json:"total"`
}

// Cost prices a usage record, applying the highest matching tier.
func (p Pricing) Cost(u Usage) Cost {
	const perMillion = 1_000_000.0

	input, output := p.Input, p.Output
	cacheWrite, cacheRead := p.CacheWrite, p.CacheRead
	// Tiers switch on the whole prompt, cached portion included: that is what
	// the vendors bill on.
	matched := -1
	for _, tier := range p.Tiers {
		if u.TotalInput() > tier.AboveInputTokens && tier.AboveInputTokens > matched {
			matched = tier.AboveInputTokens
			input, output = cmp.Or(tier.Input, p.Input), cmp.Or(tier.Output, p.Output)
			cacheWrite = cmp.Or(tier.CacheWrite, p.CacheWrite)
			cacheRead = cmp.Or(tier.CacheRead, p.CacheRead)
		}
	}

	// A card without a write rate is a vendor that bills writing the cache as
	// ordinary input; none bills it for less.
	if cacheWrite == 0 {
		cacheWrite = input
	}

	// A long-lifetime cache write is billed at twice the input rate, where a
	// short one costs the CacheWrite rate. Splitting them here is what keeps a
	// long-cache turn from being understated.
	long := min(max(u.CacheWrite1h, 0), u.CacheWrite)
	short := u.CacheWrite - long

	c := Cost{
		Currency:   p.Currency,
		Input:      float64(u.Input) * input / perMillion,
		Output:     float64(u.Output) * output / perMillion,
		CacheWrite: (float64(short)*cacheWrite + float64(long)*input*2) / perMillion,
		CacheRead:  float64(u.CacheRead) * cacheRead / perMillion,
	}
	c.Total = c.Input + c.Output + c.CacheWrite + c.CacheRead
	return c
}

// Known reports whether any rate is set.
func (p Pricing) Known() bool {
	return p.Input > 0 || p.Output > 0 || p.CacheWrite > 0 || p.CacheRead > 0
}
