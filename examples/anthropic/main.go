// Command anthropic shows what the Anthropic Messages protocol offers that the
// others do not: thinking you can read, and a prompt cache you can aim.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/anthropic "Why does this deadlock?"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
)

func main() {
	model := flag.String("model", "anthropic/claude-opus-5", "model reference")
	flag.Parse()

	if err := run(*model, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref, question string) error {
	if question == "" {
		question = "Why might a goroutine reading from an unbuffered channel deadlock?"
	}
	client, err := auth.Client(ref)
	if err != nil {
		return err
	}

	ctx := context.Background()
	thinking := false
	for event, err := range client.Stream(ctx, []ai.Message{ai.UserMessage(question)},
		// A long, stable system prompt is what a cache is for: it is the same
		// on every turn, so the provider can be asked to keep it.
		ai.WithSystem("You are a Go expert. Answer in at most three sentences."),

		// Claude's ladder is normalized: "high" here becomes whatever this
		// model's generation wants — a token budget on 4.5, an effort string
		// on 4.6 and later. The catalog carries that difference, not this file.
		ai.WithEffort(ai.EffortHigh),

		// Ask the provider to hold the prefix. Anthropic bills a long-lived
		// write at twice the input rate and a short one at 1.25x, so this is a
		// real cost decision, not a free switch.
		ai.WithCacheRetention(ai.CacheShort),
	) {
		if err != nil {
			return err
		}
		switch event.Type {
		case ai.EventBlockDelta:
			switch event.Block.Type {
			case ai.BlockThinking:
				// Thinking arrives as its own block kind, before the answer.
				// You may show it or hide it — but it must be replayed
				// unchanged, with its signature, in the next turn's history.
				if !thinking {
					fmt.Print("\033[2mthinking: ")
					thinking = true
				}
				fmt.Print(event.Block.Text)
			case ai.BlockText:
				if thinking {
					fmt.Print("\033[0m\n\n")
					thinking = false
				}
				fmt.Print(event.Block.Text)
			}

		case ai.EventDone:
			resp := event.Response
			fmt.Printf("\n\n\033[2m%d fresh in", resp.Usage.Input)
			if resp.Usage.CacheWrite > 0 {
				fmt.Printf(" · %d written to cache", resp.Usage.CacheWrite)
			}
			if resp.Usage.CacheRead > 0 {
				fmt.Printf(" · %d read from cache", resp.Usage.CacheRead)
			}
			fmt.Printf(" · %d out\033[0m\n", resp.Usage.Output)

			// Run this twice: the second run should report cache reads rather
			// than a second write, and cost less for the same prompt.
			if p := client.Model().Pricing; p.Known() {
				c := p.Cost(resp.Usage)
				fmt.Printf("\033[2m%.5f %s\033[0m\n", c.Total, c.Currency)
			}
		}
	}
	return nil
}
