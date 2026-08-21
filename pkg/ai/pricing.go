package ai

// Pricing is a published rate card, per million tokens.
//
// A zero field means the rate is unknown or not charged, not that it is free of
// consequence — Cost simply contributes nothing for it.
//
// CacheWrite and CacheRead are two different events, not two halves of one:
// writing a prefix into the provider's cache costs more than ordinary input
// (Anthropic bills 1.25x for a short entry, twice for a long one), and reading
// it back costs a small fraction. Both are input tokens — output is never
// cached — which is why they are named for what happened rather than for which
// side of the call they sat on.
//
// Currency is carried rather than assumed: several vendors publish in CNY, and
// silently summing those with USD figures would produce a number that looks
// authoritative and is meaningless.
type Pricing struct {
	Currency   string  `json:"currency,omitempty"`
	Input      float64 `json:"input,omitempty"`
	Output     float64 `json:"output,omitempty"`
	CacheWrite float64 `json:"cache_write,omitempty"`
	CacheRead  float64 `json:"cache_read,omitempty"`

	// Tiers are request-wide rate switches. The highest tier whose threshold
	// the prompt exceeds replaces the base rates for the whole request —
	// MiniMax bills M3 at double above 512k input tokens, and a flat card
	// cannot say so.
	Tiers []PricingTier `json:"tiers,omitempty"`
}

// PricingTier is a rate card that takes over above a prompt-size threshold.
type PricingTier struct {
	// AboveInputTokens is the total input token count this tier applies past.
	AboveInputTokens int     `json:"above_input_tokens"`
	Input            float64 `json:"input,omitempty"`
	Output           float64 `json:"output,omitempty"`
	CacheWrite       float64 `json:"cache_write,omitempty"`
	CacheRead        float64 `json:"cache_read,omitempty"`
}

// Currency codes used by the catalog.
const (
	USD = "USD"
	CNY = "CNY"
)

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
//
// It reports zero when no rates are known, which callers should render as
// "unknown" rather than as free.
//
// It is an estimate, and one vendor's card can be conditional in ways a rate
// card cannot express — DeepSeek bills half price outside two windows of the
// day, and several vendors publish a different card per region. A Pricing that
// carries such a condition says so in the Note of the catalog vendor it came
// from; read it before showing a figure to anyone as authoritative. The number
// here is what the published card says, not an invoice.
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
			input, output = tier.Input, tier.Output
			cacheWrite, cacheRead = tier.CacheWrite, tier.CacheRead
		}
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
