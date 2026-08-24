// Command deepseek shows the one thing about DeepSeek that will cost you money
// if you do not know it: it reasons unless told not to.
//
//	Effort unset       → reasoning_effort: "high"        (on, and billed)
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

	client, err := auth.Client(ref)
	if err != nil {
		return err
	}
	model := client.Model()

	// What rungs this model offers, and which one it uses when you say
	// nothing. Reading it beats assuming: the answer differs per vendor.
	fmt.Printf("\033[2m%s offers %v\033[0m\n", model, model.Efforts())
	if def, ok := model.DefaultLevel(); ok {
		fmt.Printf("\033[2mwith nothing set it uses %q — reasoning is on\033[0m\n", def.Effort)
	}

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
		fmt.Printf("  \033[2m%d in / %d out%s\033[0m\n",
			resp.Usage.TotalInput(), resp.Usage.Output, price(model, resp.Usage))
	}

	// The catalog records what a rate card cannot express.
	if v, ok := catalog.Find(model.Vendor); ok && v.Note != "" {
		fmt.Printf("\n\033[2mnote: %s\033[0m\n", v.Note)
	}
	return nil
}

func price(m ai.Model, u ai.Usage) string {
	if !m.Pricing.Known() {
		return ""
	}
	c := m.Pricing.Cost(u)
	return fmt.Sprintf(" · %.5f %s", c.Total, c.Currency)
}
