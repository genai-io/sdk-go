package ai

import (
	"math"
	"testing"
)

func near(got, want float64) bool { return math.Abs(got-want) < 1e-12 }

// A rate card is money, so every part of the breakdown has to be checked
// rather than only the total: two errors that cancel out still bill wrongly.
func TestCostPricesEachPartOfTheCall(t *testing.T) {
	card := Pricing{Currency: USD, Input: 3, Output: 15, CacheWrite: 3.75, CacheRead: 0.30}

	for name, tc := range map[string]struct {
		pricing Pricing
		usage   Usage
		want    Cost
	}{
		"a plain call": {
			pricing: card,
			usage:   Usage{Input: 1_000_000, Output: 100_000},
			want:    Cost{Input: 3, Output: 1.5},
		},
		"a cached call": {
			pricing: card,
			usage:   Usage{Input: 100_000, CacheWrite: 200_000, CacheRead: 700_000, Output: 10_000},
			want:    Cost{Input: 0.3, CacheWrite: 0.75, CacheRead: 0.21, Output: 0.15},
		},
		"a long-lived cache write costs twice the input rate": {
			// The two lifetimes are billed differently and arrive as one total
			// plus the long slice, so the split has to happen here.
			pricing: card,
			usage:   Usage{CacheWrite: 100_000, CacheWrite1h: 40_000},
			want:    Cost{CacheWrite: (60_000*3.75 + 40_000*3*2) / 1e6},
		},
		"a long slice larger than the whole is clamped": {
			pricing: card,
			usage:   Usage{CacheWrite: 10_000, CacheWrite1h: 99_000},
			want:    Cost{CacheWrite: 10_000 * 3 * 2 / 1e6},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := tc.pricing.Cost(tc.usage)
			tc.want.Currency = tc.pricing.Currency
			tc.want.Total = tc.want.Input + tc.want.Output + tc.want.CacheWrite + tc.want.CacheRead
			if got.Currency != tc.want.Currency ||
				!near(got.Input, tc.want.Input) || !near(got.Output, tc.want.Output) ||
				!near(got.CacheWrite, tc.want.CacheWrite) || !near(got.CacheRead, tc.want.CacheRead) ||
				!near(got.Total, tc.want.Total) {
				t.Errorf("Cost = %+v\nwant   %+v", got, tc.want)
			}
		})
	}
}

// A tier says what changes above its threshold. Reading it as a whole
// replacement card makes every rate it does not restate free, which is silent
// and wrong in the direction that costs the vendor rather than the caller.
func TestATierOverridesOnlyTheRatesItStates(t *testing.T) {
	partial := Pricing{
		Currency: USD, Input: 1, Output: 2, CacheWrite: 4, CacheRead: 8,
		Tiers: []PricingTier{{AboveInputTokens: 1_000_000, Input: 10}},
	}
	got := partial.Cost(Usage{Input: 2_000_000, Output: 1_000_000, CacheWrite: 1_000_000, CacheRead: 1_000_000})
	if !near(got.Input, 20) {
		t.Errorf("input = %v, want the tier's own rate (20)", got.Input)
	}
	if !near(got.Output, 2) || !near(got.CacheRead, 8) {
		t.Errorf("output/cache-read = %v/%v, want the base card's rates (2/8) inherited", got.Output, got.CacheRead)
	}
	// CacheWrite is the base 4 for the short slice, and twice the *tier's*
	// input rate for the long one, because that is the rate in force.
	if !near(got.CacheWrite, 4) {
		t.Errorf("cache write = %v, want the base rate (4)", got.CacheWrite)
	}

	// Below the threshold nothing switches.
	small := partial.Cost(Usage{Input: 500_000})
	if !near(small.Input, 0.5) {
		t.Errorf("input = %v, want the base rate below the threshold", small.Input)
	}
}

// Tiers are a list, not an ordering, so the highest matching one has to win
// wherever it sits.
func TestTheHighestMatchingTierWinsWhateverTheOrder(t *testing.T) {
	card := Pricing{
		Input: 1, Output: 1,
		Tiers: []PricingTier{
			{AboveInputTokens: 512_000, Input: 20, Output: 20},
			{AboveInputTokens: 128_000, Input: 10, Output: 10},
		},
	}
	for tokens, want := range map[int]float64{
		100_000: 1,
		200_000: 10,
		600_000: 20,
	} {
		got := card.Cost(Usage{Input: tokens})
		if wantCost := float64(tokens) * want / 1e6; !near(got.Input, wantCost) {
			t.Errorf("%d input tokens cost %v, want %v (the %v rate)", tokens, got.Input, wantCost, want)
		}
	}
}

// A card with no rates is a card nobody published, and pricing a call against
// it would report a confident zero.
func TestKnownSaysWhetherThereIsACardAtAll(t *testing.T) {
	if (Pricing{}).Known() {
		t.Error("an empty card claims to be known")
	}
	if !(Pricing{CacheRead: 0.01}).Known() {
		t.Error("a card with one rate set is not known")
	}
}
