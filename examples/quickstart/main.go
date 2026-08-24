// Command quickstart streams one answer from any model the catalog can name.
//
//	export OPENAI_API_KEY=...
//	go run ./examples/quickstart "why is my goroutine leaking?"
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/quickstart -model anthropic/claude-opus-5 "why is my goroutine leaking?"
//
//	export DEEPSEEK_API_KEY=...
//	go run ./examples/quickstart -model deepseek/deepseek-v4-pro -effort high "prove it"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"

	// Registers all four protocol drivers. Import just the one you need to
	// keep the other vendor SDKs out of your binary.
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

func main() {
	model := flag.String("model", "openai/gpt-4.1", "model reference, vendor/id")
	system := flag.String("system", "You are concise.", "system prompt")
	effort := flag.String("effort", "",
		"reasoning effort: off, minimal, low, medium, high, xhigh or max "+
			"(a model that does not offer the rung asked for is snapped onto the nearest it does)")
	list := flag.Bool("list", false, "list vendors with a credential in the environment")
	flag.Parse()

	if *list {
		listVendors()
		return
	}

	prompt := strings.Join(flag.Args(), " ")
	if prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: quickstart [flags] <prompt>")
		os.Exit(2)
	}

	// Ctrl-C cancels the request mid-stream rather than killing the process
	// with the connection open.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *model, *system, ai.Effort(*effort), prompt); err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, model, system string, effort ai.Effort, prompt string) error {
	client, err := auth.Client(model)
	if err != nil {
		return err
	}

	messages := []ai.Message{ai.UserMessage(prompt)}

	thinking := false
	for event, err := range client.Stream(ctx, messages,
		ai.WithSystem(system), ai.WithEffort(effort)) {
		if err != nil {
			return explain(err)
		}
		switch event.Type {
		case ai.EventBlockDelta:
			switch event.Block.Type {
			case ai.BlockThinking:
				if !thinking {
					fmt.Print("\033[2m")
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
			if thinking {
				fmt.Print("\033[0m")
				thinking = false
			}
			fmt.Printf("\n\n— %s · %d in / %d out",
				event.Response.Model, event.Response.Usage.TotalInput(), event.Response.Usage.Output)
			if p := client.Model().Pricing; p.Known() {
				cost := p.Cost(event.Response.Usage)
				fmt.Printf(" · %.4f %s", cost.Total, cost.Currency)
			}
			fmt.Println()
		}
	}
	return nil
}

// explain turns a classified failure into advice, which is the point of
// classifying them.
func explain(err error) error {
	switch {
	case ai.IsAuth(err):
		return fmt.Errorf("%w\n(check the vendor's API key variable)", err)
	case ai.IsContextExceeded(err):
		return fmt.Errorf("%w\n(the prompt is larger than the model's context window)", err)
	case ai.IsRetryable(err):
		if after := ai.RetryAfter(err); after > 0 {
			return fmt.Errorf("%w\n(transient — retry in %s)", err, after)
		}
		return fmt.Errorf("%w\n(transient — worth retrying)", err)
	default:
		return err
	}
}

func listVendors() {
	available := auth.Available()
	for _, v := range available {
		models := ""
		if known := v.ModelList(); len(known) > 0 {
			models = " — e.g. " + known[0].String()
		}
		fmt.Printf("%-12s %-20s %s%s\n", v.ID, v.DisplayName, v.API, models)
	}
	if len(available) == 0 {
		fmt.Println("no credentials found; set one of:")
		for _, v := range catalog.All() {
			if len(v.KeyEnv) > 0 {
				fmt.Printf("  %-12s %s\n", v.ID, strings.Join(v.KeyEnv, ", "))
			}
		}
	}
}
