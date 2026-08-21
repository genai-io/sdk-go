// Command openai shows what the Responses protocol offers that Chat
// Completions does not: a reasoning model's own working, carried across turns.
//
// Drop it and the model re-reasons from scratch every turn. Carry it and it
// resumes — which is the difference between a multi-step conversation that
// gets cheaper and one that does not.
//
//	export OPENAI_API_KEY=...
//	go run ./examples/openai
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)

func main() {
	model := flag.String("model", "openai/gpt-4.1", "model reference")
	flag.Parse()

	if err := run(*model); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref string) error {
	client, err := auth.Client(ref)
	if err != nil {
		return err
	}
	ctx := context.Background()

	// Two turns, where the second depends on the first.
	turns := []string{
		"Pick a number between 1 and 100 and keep it to yourself.",
		"Was the number you picked prime?",
	}

	var history []ai.Message
	for i, question := range turns {
		history = append(history, ai.UserMessage(question))

		resp, err := client.Complete(ctx, history, ai.WithEffort(ai.EffortMedium))
		if err != nil {
			return err
		}

		fmt.Printf("\033[1mturn %d\033[0m  %s\n", i+1, resp.Text())

		// This is the whole point of the example. resp.Message() carries every
		// block the model produced — including the opaque reasoning items you
		// cannot read — so appending it is what lets the next turn resume.
		//
		// Appending ai.AssistantMessage(resp.Text()) instead would look identical and
		// silently throw the reasoning away.
		history = append(history, resp.Message())

		if items := resp.ReasoningItems(); len(items) > 0 {
			fmt.Printf("  \033[2mcarrying %d reasoning item(s) forward", len(items))
			if items[0].Summary != "" {
				fmt.Printf(" — %q", items[0].Summary)
			}
			fmt.Println("\033[0m")
		}
		fmt.Printf("  \033[2m%d in / %d out", resp.Usage.TotalInput(), resp.Usage.Output)
		if resp.Usage.Reasoning > 0 {
			fmt.Printf(" (%d of it reasoning)", resp.Usage.Reasoning)
		}
		fmt.Println("\033[0m")
	}
	return nil
}
