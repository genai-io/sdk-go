package ai

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
type TokenCount struct {
	// Tokens is the prompt size.
	Tokens int
	// Exact is true when the provider counted, false when this package
	// estimated.
	Exact bool
}

// CountTokens reports how large a prompt is before it is sent.
func (c *Client) CountTokens(ctx context.Context, messages []Message, opts ...Option) (TokenCount, error) {
	req, err := c.prepare(ctx, messages, opts)
	if err != nil {
		return TokenCount{}, err
	}
	if counter, ok := c.driver.(TokenCounter); ok {
		n, err := counter.CountTokens(ctx, req)
		if err == nil {
			return TokenCount{Tokens: n, Exact: true}, nil
		}
		// An unavailable or transiently down counting endpoint should not stop
		// the caller from sizing a prompt at all. Authentication, malformed
		// requests and cancellation are different: estimating would hide an
		// actionable failure that generation will hit too.
		if IsUnsupported(err) || IsRetryable(err) {
			return TokenCount{Tokens: req.EstimateTokens()}, nil
		}
		return TokenCount{}, err
	}
	return TokenCount{Tokens: req.EstimateTokens()}, nil
}

// Headroom reports how many tokens are left in the model's context window
// after the prompt, and whether the figure can be trusted.
func (c *Client) Headroom(ctx context.Context, messages []Message, opts ...Option) (left int, count TokenCount, err error) {
	count, err = c.CountTokens(ctx, messages, opts...)
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

// estimateContent sizes one sequence of blocks. A tool result holds a sequence
// of its own, so this recurses into it: an image a tool returned costs what an
// image costs anywhere else, and counting it as nothing is how a prompt that
// was measured as small arrives over the window.
func estimateContent(c Content) int {
	total := 0
	for _, block := range c {
		switch block.Type {
		case BlockText, BlockThinking:
			total += EstimateTokens(block.Text)
		case BlockImage:
			total += estimateImage(block.Image)
		case BlockToolCall:
			if block.ToolCall != nil {
				total += messageOverhead + EstimateTokens(block.ToolCall.Name) + EstimateTokens(block.ToolCall.Input)
			}
		case BlockToolResult:
			if block.ToolResult != nil {
				total += messageOverhead + estimateContent(block.ToolResult.Content)
			}
		case BlockReasoning:
			if block.Reasoning != nil {
				total += EstimateTokens(block.Reasoning.Summary)
			}
		}
	}
	return total
}

// EstimateTokens returns the estimated size of the whole prompt: the system
// prompt, every message, and the tool definitions, which are part of what is
// sent and are easy to forget — a dozen schemas can outweigh the conversation.
func (r *Request) EstimateTokens() int {
	if r == nil {
		return 0
	}
	total := EstimateTokens(r.System)
	for _, m := range r.Messages {
		total += messageOverhead + estimateContent(m.Content)
	}
	for _, t := range r.Tools {
		total += messageOverhead + EstimateTokens(t.Schema.Name) + EstimateTokens(t.Schema.Description)
		if t.Schema.Definition != nil {
			if schema, err := json.Marshal(t.Schema.Definition); err == nil {
				total += EstimateTokens(string(schema))
			}
		}
	}
	return total
}

// runeClass groups characters the way a BPE tokenizer's pre-tokenizer splits
// them: it breaks text into runs of letters, digits, punctuation and
// whitespace before merging anything, so a run never spans two classes.
type runeClass int

const (
	classLetter runeClass = iota
	classDigit
	classPunct
	classSpace
	// classWide is the scripts written without spaces between words — Han,
	// kana, Hangul. No word boundaries to merge on, so about a token each.
	classWide
	// classScript is the non-Latin alphabets — Cyrillic, Greek, Arabic,
	// Hebrew. Held apart from Latin only because no tokenizer represents them
	// as well, so the same word costs more. Accented Latin is Latin: a run
	// split at every accent is a word billed as its fragments.
	classScript
	// classSymbol is emoji and the other non-letter, non-digit runes above
	// ASCII. The expensive ones: an emoji is often several tokens.
	classSymbol
)

// classify groups one rune. ASCII first: a prompt is mostly ASCII and the
// unicode tables are not free.
func classify(r rune) runeClass {
	if r < 0x80 {
		switch {
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			return classSpace
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			return classLetter
		case r >= '0' && r <= '9':
			return classDigit
		default:
			return classPunct
		}
	}
	switch {
	case isWideScript(r):
		return classWide
	case unicode.Is(unicode.Latin, r):
		return classLetter
	case unicode.IsLetter(r):
		return classScript
	case unicode.IsDigit(r):
		return classDigit
	case unicode.IsSpace(r):
		return classSpace
	}
	return classSymbol
}

func isWideScript(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

// runTokens estimates how many tokens a run of n same-class characters becomes.
// The ratios approximate what BPE merging does within each class: words merge
// aggressively, digits merge in groups of about three, punctuation merges only
// in short common pairs, and every non-ASCII rune stands roughly on its own.
func runTokens(class runeClass, n int) int {
	switch class {
	case classLetter:
		return max((n+2)/4, 1)
	case classScript:
		// Three, not Latin's four: Arabic runs about 3.3 characters a token.
		return max((n+2)/3, 1)
	case classDigit, classPunct:
		// Both merge, but only in short groups: tokenizers chunk digits about
		// three at a time, and JSON and code are dominated by short punctuation
		// runs learned as single units — `":"`, `":{"`, `!=`, `:=`, `))`. So
		// each sits well above one-token-per-character and well below prose.
		// (The classes stay separate because they decide where runs break —
		// `12+34` is three runs, not one — only the ratio is shared.)
		return max((n+2)/3, 1)
	case classWide:
		return n
	case classSymbol:
		// A variation selector or a joiner in the sequence costs its own.
		return 2 * n
	default: // classSpace
		// A lone space is absorbed into the token that follows it — " the" is
		// one token, not two. Longer runs (indentation, blank lines) do cost.
		if n <= 1 {
			return 0
		}
		return max((n+3)/4, 1)
	}
}

// EstimateTokens approximates what a string costs in tokens without running a
// tokenizer. Use [Request.EstimateTokens] to size a whole prompt.
//
// It counts by pre-token run, because the familiar four-characters-per-token is
// the ratio for prose and most of an agent's prompt is not prose: a flat ratio
// reads a JSON tool result at 71% of its real size and English at 110%.
//
// The result runs above o200k_base everywhere, by 12% to 45%. That is
// deliberate — o200k is the most token-efficient tokenizer this SDK talks to,
// so landing on it exactly would land under the others.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}

	// runLength == 0 means no run has opened yet, so the seeded class is
	// never used to close one.
	tokens, runLength, runClass := 0, 0, classLetter
	for _, r := range s {
		class := classify(r)
		if runLength > 0 && class != runClass {
			tokens += runTokens(runClass, runLength)
			runLength = 0
		}
		runClass, runLength = class, runLength+1
	}
	tokens += runTokens(runClass, runLength)

	return max(tokens, 1)
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

// SplitPromptTokens separates a combined prompt count into its fresh and
// cached halves.
func SplitPromptTokens(promptTokens, cachedTokens int) (fresh, cached int) {
	promptTokens = max(promptTokens, 0)
	cached = min(max(cachedTokens, 0), promptTokens)
	return promptTokens - cached, cached
}
