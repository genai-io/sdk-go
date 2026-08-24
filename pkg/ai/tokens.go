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
	"unicode/utf8"
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
			return TokenCount{Tokens: EstimateTokens(req)}, nil
		}
		return TokenCount{}, err
	}
	return TokenCount{Tokens: EstimateTokens(req)}, nil
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
func EstimateTokens(req *Request) int {
	if req == nil {
		return 0
	}
	total := estimateText(req.System)
	for _, m := range req.Messages {
		total += messageOverhead
		for _, block := range m.Content {
			switch block.Type {
			case BlockText, BlockThinking:
				total += estimateText(block.Text)
			case BlockImage:
				total += estimateImage(block.Image)
			case BlockToolCall:
				if block.ToolCall != nil {
					total += messageOverhead + estimateText(block.ToolCall.Name) + estimateText(block.ToolCall.Input)
				}
			case BlockToolResult:
				if block.ToolResult != nil {
					total += messageOverhead + estimateText(block.ToolResult.Content)
				}
			case BlockReasoning:
				if block.Reasoning != nil {
					total += estimateText(block.Reasoning.Summary)
				}
			}
		}
	}
	// Tool definitions are part of the prompt and are easy to forget: a dozen
	// schemas can outweigh the conversation.
	for _, t := range req.Tools {
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
		otherBytes += utf8.RuneLen(r)
	}
	return ideographs*tokensPerIdeograph + (otherBytes+bytesPerToken-1)/bytesPerToken
}

func isIdeograph(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
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

// IsOverflow reports whether a turn failed because the prompt exceeded the
// model's context window.
func IsOverflow(resp *Response, model Model) bool {
	if resp == nil {
		return false
	}
	if resp.Err != nil && IsContextExceeded(resp.Err) {
		return true
	}
	window := model.ContextWindow
	if window <= 0 {
		return false
	}
	prompt := resp.Usage.TotalInput()
	if prompt == 0 {
		return false
	}
	// Accepted silently: billed for more prompt than the window holds.
	if prompt > window {
		return true
	}
	// Truncated to fit, then no room left to answer.
	return resp.StopReason == StopMaxTokens && resp.Usage.Output == 0 && prompt >= window
}
