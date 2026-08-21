// Command ollama talks to a model running on your own machine.
//
// It is the case with no credential at all: Ollama serves an OpenAI-compatible
// API on localhost, so the same driver that reaches DeepSeek and Qwen reaches
// it. What differs is one row in the catalog — no key variable, and a base URL
// that people paste as a bare host and port.
//
//	ollama serve
//	ollama pull llama4
//	go run ./examples/ollama
//
//	OLLAMA_BASE_URL=http://box.local:11434 go run ./examples/ollama
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
)

func main() {
	modelID := flag.String("model", "llama4", "a model you have pulled")
	flag.Parse()

	if err := run(*modelID, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(modelID, question string) error {
	if question == "" {
		question = "Say hello in one short sentence."
	}

	// An Endpoint is the right shape for a local server: what it serves changes
	// as you pull and remove models, so ask it rather than trusting a table.
	p, err := auth.Endpoint("ollama")
	if err != nil {
		return err
	}
	fmt.Printf("\033[2m%s · no credential needed\033[0m\n", p.Name())

	// Models() answers immediately from the static baseline; Refresh is the
	// only thing that touches the network. A picker renders first and corrects
	// itself after — it never blocks on a server that might be down.
	if err := p.Refresh(context.Background()); err != nil {
		return fmt.Errorf("is ollama running? %w", err)
	}
	show(p)

	client, err := p.Open(modelID)
	if err != nil {
		return err
	}
	resp, err := client.Complete(context.Background(), []ai.Message{ai.UserMessage(question)})
	if err != nil {
		return err
	}
	fmt.Printf("\n%s\n", resp.Text())
	return nil
}

func show(p *endpoint.Endpoint) {
	models := p.Models()
	fmt.Printf("\033[2m%d model(s) pulled locally:", len(models))
	for i, m := range models {
		if i == 5 {
			fmt.Printf(" …")
			break
		}
		fmt.Printf(" %s", m.ID)
	}
	fmt.Println("\033[0m")
}
