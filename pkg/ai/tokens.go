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
		// A counting endpoint that is down should not stop a caller sizing a
		// prompt. Auth, malformed requests and cancellation are different:
		// estimating would hide a failure generation will hit too.
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

// Token-estimation constants, leaning towards over-counting: compacting early
// costs some context, discovering the prompt was too large costs a request.
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

// estimateContent sizes one sequence of blocks, recursing into a tool result:
// an image a tool returned costs what an image costs anywhere else.
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

// EstimateTokens is the whole prompt: system, messages, and the tool
// definitions, a dozen of which can outweigh the conversation.
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

// runeClass is how a BPE pre-tokenizer splits text before merging anything, so
// a run never spans two classes.
type runeClass int

const (
	classLetter runeClass = iota
	classDigit
	classPunct
	classSpace
	// classWide is written without spaces between words, so about a token
	// a character: nothing for a tokenizer to merge on.
	classWide
	// classScript is the non-Latin alphabets. Held apart from Latin only
	// because no tokenizer represents them as well. Accented Latin stays
	// Latin: a run split at every accent is a word billed as its fragments.
	classScript
	// classSymbol is emoji and the other non-letter runes above ASCII.
	classSymbol
)

// classify groups one rune. ASCII first: the unicode tables are not free.
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

// runTokens is what a run of n same-class characters costs.
func runTokens(class runeClass, n int) int {
	switch class {
	case classLetter:
		return max((n+2)/4, 1)
	case classScript:
		// Three, not Latin's four: Arabic runs about 3.3 characters a token.
		return max((n+2)/3, 1)
	case classDigit, classPunct:
		// Digits chunk about three at a time; JSON and code are dominated by
		// short punctuation runs learned as single units. Separate classes
		// because they break runs differently — `12+34` is three runs.
		return max((n+2)/3, 1)
	case classWide:
		return n
	case classSymbol:
		// A variation selector or joiner in the sequence costs its own.
		return 2 * n
	default: // classSpace
		// A lone space is absorbed into the token after it; indentation and
		// blank lines are not.
		if n <= 1 {
			return 0
		}
		return max((n+3)/4, 1)
	}
}

// EstimateTokens approximates what a string costs in tokens. Use
// [Request.EstimateTokens] for a whole prompt.
//
// It counts by pre-token run: four-characters-per-token is the ratio for prose,
// and most of an agent's prompt is not prose. The result runs 12% to 45% above
// o200k_base, deliberately — o200k is the most token-efficient tokenizer this
// SDK talks to, so landing on it exactly would land under the others.
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
