// Command deepseek shows the one thing about DeepSeek that will cost you money
// if you do not know it: it reasons unless told not to.
//
//	Effort unset       → nothing sent                    (on: DeepSeek's own default, and billed)
//	Effort EffortOff   → thinking: {"type":"disabled"}   (off)
//	Effort EffortHigh  → reasoning_effort: "high"        (on)
//
//	export DEEPSEEK_API_KEY=...
//	go run ./examples/deepseek "What is 17 * 23?"
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
	ref := flag.String("model", "deepseek/deepseek-v4-pro", "model reference")
	flag.Parse()

	if err := run(*ref, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref, question string) error {
	if question == "" {
		question = "What is 17 * 23? Answer with the number only."
	}

	cfg, err := auth.Config(ref)
	if err != nil {
		return err
	}
	// The catalog knows the protocol, not the model: say which efforts it
	// offers, and the SDK spells each one the way the protocol wants.
	for _, e := range cfg.Model.WireEfforts() {
		cfg.Model.Reasoning = append(cfg.Model.Reasoning, ai.ReasoningLevel{Effort: e})
	}
	client, err := ai.New(cfg)
	if err != nil {
		return err
	}
	model := client.Model()

	// What rungs this model offers. Say nothing and nothing is sent, which
	// on DeepSeek leaves reasoning on.
	fmt.Printf("\033[2m%s offers %v\033[0m\n", model, model.Efforts())

	// The same question twice: once thinking, once not. Only the Effort
	// changes; the driver turns that into whichever field this endpoint wants.
	for _, effort := range []ai.Effort{ai.EffortOff, ai.EffortHigh} {
		resp, err := client.Complete(context.Background(),
			[]ai.Message{ai.UserMessage(question)}, ai.WithEffort(effort))
		if err != nil {
			return err
		}

		fmt.Printf("\n\033[1meffort %q\033[0m\n  %s\n", effort, strings.TrimSpace(resp.Text()))
		if t := resp.Thinking(); t != "" {
			fmt.Printf("  \033[2mit thought first: %d characters\033[0m\n", len(t))
		}
		fmt.Printf("  \033[2m%d in / %d out\033[0m\n", resp.Usage.TotalInput(), resp.Usage.Output)
	}

	if v, ok := catalog.Find(model.Vendor); ok && v.Note != "" {
		fmt.Printf("\n\033[2mnote: %s\033[0m\n", v.Note)
	}
	return nil
}
