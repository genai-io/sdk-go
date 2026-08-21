package llm_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/llmtest"
)

// Without a way to size a prompt, a caller learns it was too large only by
// sending it — the request is spent and an error arrives instead of an answer.
func TestEstimateCoversEveryPartOfThePrompt(t *testing.T) {
	base := &llm.Prompt{Messages: []llm.Message{llm.User("hello there")}}
	baseN := llm.EstimateTokens(base)
	if baseN == 0 {
		t.Fatal("a prompt with text estimated to nothing")
	}

	// Each of these is part of what the model is billed for, and each is easy
	// to forget.
	for _, tc := range []struct {
		name   string
		prompt *llm.Prompt
	}{
		{"system prompt", &llm.Prompt{System: strings.Repeat("instructions ", 50), Messages: base.Messages}},
		{"tool schemas", &llm.Prompt{Messages: base.Messages, Tools: []llm.Tool{{
			Name: "search", Description: "search the web",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "what to look for"},
			}},
		}}}},
		{"thinking", &llm.Prompt{Messages: []llm.Message{
			llm.User("hello there"),
			{Role: llm.RoleAssistant, Thinking: strings.Repeat("reasoning ", 50), Content: llm.Text("x")},
		}}},
		{"tool results", &llm.Prompt{Messages: append(base.Messages,
			llm.ToolResultsMessage(llm.ToolResult{ToolCallID: "1", Content: strings.Repeat("output ", 50)}))}},
	} {
		if got := llm.EstimateTokens(tc.prompt); got <= baseN {
			t.Errorf("%s contributed nothing: %d vs base %d", tc.name, got, baseN)
		}
	}
}

// Four bytes per token is a Latin-script rule of thumb. A CJK prompt sized
// that way is under-counted about fourfold, which is the direction that costs
// a failed request.
func TestEstimateIsScriptAware(t *testing.T) {
	const chars = 200
	latin := llm.EstimateTokens(&llm.Prompt{Messages: []llm.Message{llm.User(strings.Repeat("a", chars))}})
	chinese := llm.EstimateTokens(&llm.Prompt{Messages: []llm.Message{llm.User(strings.Repeat("字", chars))}})

	if chinese <= latin {
		t.Errorf("CJK=%d latin=%d — the same character count should cost more in CJK", chinese, latin)
	}
	// A CJK character is about a token; the same count of Latin letters is
	// about a quarter of that.
	if chinese < chars || chinese > chars+llm.EstimateTokens(&llm.Prompt{})+20 {
		t.Errorf("CJK estimate = %d, want about %d", chinese, chars)
	}
}

// Compression ratios vary by orders of magnitude, so payload size says almost
// nothing about what an image costs. The dimensions are read from the header.
func TestEstimateSizesImagesByPixels(t *testing.T) {
	small := pngOf(t, 64, 64)
	large := pngOf(t, 1024, 1024)

	smallN := llm.EstimateTokens(&llm.Prompt{Messages: []llm.Message{llm.User("look", small)}})
	largeN := llm.EstimateTokens(&llm.Prompt{Messages: []llm.Message{llm.User("look", large)}})

	if largeN <= smallN*10 {
		t.Errorf("small=%d large=%d — a 256x larger image should cost far more", smallN, largeN)
	}
	// 1024x1024 at ~750 pixels per token is around 1400.
	if largeN < 1_000 || largeN > 2_000 {
		t.Errorf("1024x1024 estimated at %d tokens, want roughly 1400", largeN)
	}

	// An image whose header cannot be read still costs something rather than
	// silently costing nothing.
	broken := llm.Image{MediaType: "image/webp", Data: base64.StdEncoding.EncodeToString([]byte("not an image"))}
	if n := llm.EstimateTokens(&llm.Prompt{Messages: []llm.Message{llm.User("x", broken)}}); n < 100 {
		t.Errorf("an unreadable image estimated at %d tokens", n)
	}
}

func pngOf(t *testing.T, w, h int) llm.Image {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.White)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return llm.Image{MediaType: "image/png", Data: base64.StdEncoding.EncodeToString(buf.Bytes())}
}

// A driver that cannot count says so, so a caller knows to leave headroom
// rather than filling the window to the brim.
func TestCountTokensReportsWhetherItIsExact(t *testing.T) {
	client := llmtest.Client(llmtest.Text("ok")) // the fake counts nothing
	count, err := client.CountTokens(context.Background(),
		&llm.Prompt{Messages: []llm.Message{llm.User("hello")}}, nil)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if count.Exact {
		t.Error("a driver with no counting endpoint must not claim an exact count")
	}
	if count.Tokens == 0 {
		t.Error("estimate = 0")
	}
}

// An unknown window reports no headroom rather than infinite headroom: acting
// on a window nobody knows is what proactive compaction must not do.
func TestHeadroom(t *testing.T) {
	prompt := &llm.Prompt{Messages: []llm.Message{llm.User(strings.Repeat("word ", 100))}}

	sized := llmtest.Client(llmtest.Text("ok"))
	left, count, err := sized.Headroom(context.Background(), prompt, nil)
	if err != nil {
		t.Fatalf("Headroom: %v", err)
	}
	if left <= 0 || left >= llmtest.Model.ContextWindow {
		t.Errorf("headroom = %d of %d", left, llmtest.Model.ContextWindow)
	}
	if left != llmtest.Model.ContextWindow-count.Tokens {
		t.Errorf("headroom %d does not match window minus count %d", left, count.Tokens)
	}

	unsized := llmtest.Model
	unsized.ContextWindow = 0
	left, _, err = llm.New(llmtest.Text("ok"), unsized).Headroom(context.Background(), prompt, nil)
	if err != nil {
		t.Fatalf("Headroom: %v", err)
	}
	if left != 0 {
		t.Errorf("headroom = %d for an unknown window, want 0", left)
	}
}
