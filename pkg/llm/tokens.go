package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"strings"
	"unicode"
)

// TokenCount is how large a prompt is, and how much that number can be
// trusted.
//
// The distinction is the point. An exact count comes from the provider's own
// tokenizer and can be compared against the context window directly. An
// estimate comes from a heuristic here, and a caller acting on one should
// leave headroom.
type TokenCount struct {
	// Tokens is the prompt size.
	Tokens int
	// Exact is true when the provider counted, false when this package
	// estimated.
	Exact bool
}

// TokenCounter is an optional Driver capability: an endpoint that will count a
// prompt without generating from it.
//
// Anthropic and Google publish one; the OpenAI-family protocols do not, and
// their drivers do not implement this.
type TokenCounter interface {
	// CountTokens reports the prompt's exact size, as the provider's own
	// tokenizer sees it.
	CountTokens(ctx context.Context, p *Prompt, opts Options) (int, error)
}

// CountTokens reports how large a prompt is before it is sent.
//
// Without this a caller learns a prompt was too large only by sending it: the
// request is spent, the latency is spent, and the error arrives instead of an
// answer. Knowing beforehand is what makes it possible to compact first.
//
// It uses the provider's own tokenizer where the protocol offers one, and
// falls back to EstimateTokens where it does not. The result says which
// happened; an estimate is not a licence to fill the window to the brim.
func (c *Client) CountTokens(ctx context.Context, p *Prompt, opts *Options) (TokenCount, error) {
	if counter, ok := c.driver.(TokenCounter); ok {
		n, err := counter.CountTokens(ctx, p, opts.merged(c.model, c.defaults))
		if err == nil {
			return TokenCount{Tokens: n, Exact: true}, nil
		}
		// A counting endpoint that is down should not stop the caller from
		// sizing the prompt at all — the estimate is worse, not useless.
		return TokenCount{Tokens: EstimateTokens(p)}, nil
	}
	return TokenCount{Tokens: EstimateTokens(p)}, nil
}

// Headroom reports how many tokens are left in the model's context window
// after the prompt, and whether the figure can be trusted.
//
// A zero window means the model's size is unknown, which is reported as no
// headroom rather than as infinite: acting on an unknown window is what
// proactive compaction must not do.
func (c *Client) Headroom(ctx context.Context, p *Prompt, opts *Options) (left int, count TokenCount, err error) {
	count, err = c.CountTokens(ctx, p, opts)
	if err != nil {
		return 0, count, err
	}
	window := c.model.ContextWindow
	if window <= 0 {
		return 0, count, nil
	}
	return max(window-count.Tokens, 0), count, nil
}

// Token-estimation constants. They lean towards over-counting: compacting a
// little early costs some context, while discovering the prompt was too large
// costs a whole request.
const (
	// bytesPerToken is the ratio for Latin-script text across the BPE
	// tokenizers in use. Four is the figure every vendor quotes as a rule of
	// thumb; real text runs a little under it.
	bytesPerToken = 4
	// tokensPerIdeograph is what a CJK character costs. Modern tokenizers
	// average a little below one; one is the safe side.
	tokensPerIdeograph = 1
	// pixelsPerImageToken is Anthropic's published ratio, and close enough to
	// the other vision models to be a fair estimate for all of them.
	pixelsPerImageToken = 750
	// unsizedImageTokens stands in for an image whose dimensions cannot be
	// read — an unsupported format, or a truncated payload.
	unsizedImageTokens = 1_200
	// messageOverhead covers the role marker and delimiters each message costs
	// on top of its text.
	messageOverhead = 4
)

// EstimateTokens approximates a prompt's size without asking the provider.
//
// It is a heuristic, and it is the fallback for the protocols that publish no
// counting endpoint. Text is measured by script — a CJK character costs about
// a token where four bytes of Latin text do — and images by their pixel count,
// read from the file header rather than guessed, because a thumbnail and a
// screenshot differ by two orders of magnitude.
//
// It leans towards over-counting. A caller comparing it against a context
// window should still leave headroom: being wrong in this direction wastes a
// little context, and being wrong in the other wastes a request.
func EstimateTokens(p *Prompt) int {
	if p == nil {
		return 0
	}
	total := estimateText(p.System)
	for _, m := range p.Messages {
		total += messageOverhead
		for _, part := range m.Content {
			switch part.Type {
			case PartText:
				total += estimateText(part.Text)
			case PartImage:
				total += estimateImage(part.Image)
			}
		}
		total += estimateText(m.Thinking)
		for _, tc := range m.ToolCalls {
			total += messageOverhead + estimateText(tc.Name) + estimateText(tc.Input)
		}
		for _, r := range m.ToolResults {
			total += messageOverhead + estimateText(r.Content)
		}
	}
	// Tool definitions are part of the prompt and are easy to forget: a dozen
	// schemas can outweigh the conversation.
	for _, t := range p.Tools {
		total += messageOverhead + estimateText(t.Name) + estimateText(t.Description)
		if t.Parameters != nil {
			if schema, err := json.Marshal(t.Parameters); err == nil {
				total += estimateText(string(schema))
			}
		}
	}
	return total
}

// estimateText sizes a string by script, so that a CJK prompt is not
// under-counted by the factor of four that Latin text assumes.
func estimateText(s string) int {
	if s == "" {
		return 0
	}
	var ideographs, otherBytes int
	for _, r := range s {
		if isIdeograph(r) {
			ideographs++
			continue
		}
		otherBytes += utf8Len(r)
	}
	return ideographs*tokensPerIdeograph + (otherBytes+bytesPerToken-1)/bytesPerToken
}

func isIdeograph(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

func utf8Len(r rune) int {
	switch {
	case r < 0x80:
		return 1
	case r < 0x800:
		return 2
	case r < 0x10000:
		return 3
	default:
		return 4
	}
}

// estimateImage reads the image header for its dimensions rather than guessing
// from the payload size — compression ratios vary by orders of magnitude, so
// bytes say almost nothing about how many tokens an image costs.
func estimateImage(img *Image) int {
	if img == nil {
		return 0
	}
	cfg, _, err := image.DecodeConfig(base64.NewDecoder(base64.StdEncoding, strings.NewReader(img.Data)))
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return unsizedImageTokens
	}
	return (cfg.Width*cfg.Height + pixelsPerImageToken - 1) / pixelsPerImageToken
}
