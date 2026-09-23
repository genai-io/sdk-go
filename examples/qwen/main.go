// Command qwen talks to Alibaba's Qwen models through DashScope.
//
//	export DASHSCOPE_API_KEY=...
//	go run ./examples/qwen "用两句话解释 goroutine 泄漏"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
)

func main() {
	model := flag.String("model", "alibaba/qwen3.7-plus", "model reference")
	flag.Parse()

	if err := run(*model, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref, question string) error {
	if question == "" {
		question = "用两句话解释什么是 goroutine 泄漏。"
	}

	// What the catalog knows about this vendor, before any network call.
	m, err := catalog.Model(ref)
	if err != nil {
		return err
	}
	fmt.Printf("\033[2m%s · %s · %s\033[0m\n", m.Vendor, m.API, m.BaseURL)
	// The catalog states no window: limits are the caller's to set on the
	// model, and a zero one means "cannot size this prompt".

	client, err := auth.Client(ref)
	if err != nil {
		return err
	}

	// Qwen's reasoning switch is neither OpenAI's reasoning_effort nor
	// Anthropic's budget: DashScope wants enable_thinking plus a token budget.
	// The catalog records that dialect, so this stays the normalized rung.
	resp, err := client.Complete(context.Background(), []ai.Message{ai.UserMessage(question)},
		ai.WithSystem("你是一个简洁的 Go 专家。"),
		ai.WithEffort(ai.EffortMedium))
	if err != nil {
		return err
	}

	if t := resp.Thinking(); t != "" {
		fmt.Printf("\n\033[2m%s\033[0m\n", t)
	}
	fmt.Printf("\n%s\n", resp.Text())
	fmt.Printf("\n\033[2m%d in / %d out\033[0m\n", resp.Usage.TotalInput(), resp.Usage.Output)
	return nil
}

// anyCompatibleEndpoint reaches an OpenAI-compatible server the catalog has
// never heard of — a self-hosted vLLM, an internal gateway, a new vendor.
func anyCompatibleEndpoint(baseURL, key, modelID string) (*ai.Client, error) {
	return ai.New(ai.Config{
		Model: ai.Model{
			ID:            modelID,
			API:           ai.APIOpenAIChat,
			ContextWindow: 32_000, // state it if you know it; leave it zero if you do not
		},
		APIKey:  key,
		BaseURL: baseURL,
	})
}

var _ = anyCompatibleEndpoint
