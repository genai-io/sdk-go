// Command tools runs a tool-calling loop, which is four rules and a for loop.
//
// The model does not run your tools. It asks you to, you answer, and it
// continues — so the loop is yours to write, and the rules that make it work
// are the ones this file exists to show:
//
//   - The schema comes from the Go type the arguments decode into, so the
//     shape the model is told about cannot drift from the shape you parse.
//   - Check the arguments before running anything. A model's mistake handed
//     back as a tool result is something it can correct; the same mistake run
//     is whatever your tool does with a missing field.
//   - Answer every call in the turn that immediately follows. Every protocol
//     here rejects a conversation where one is left hanging.
//   - Append response.Message(), not ai.AssistantMessage(response.Text()).
//     The first carries the model's thinking and reasoning state forward; the
//     second silently drops it and a reasoning model starts over each turn.
//
// ai.Handle and ai.RunTools do the middle two, and close a join that a
// hand-written dispatch leaves open. With more than one tool, only call.Name
// says which was meant, so a switch on it has to remember which argument type
// goes with which string — rename one and it still compiles. Handle takes the
// type from the function that receives it, so the name, the type and the code
// are written once, together.
//
// The question below needs both tools and several lookups, so the loop really
// does go round more than once.
//
//	export OPENAI_API_KEY=...
//	go run ./examples/tools
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/tools -model anthropic/claude-opus-5
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

// Each Args type is both a tool's schema and the struct its arguments decode
// into. The tags are what the model reads: a field name alone does not say
// that "city" means one of a fixed few, and enum does.
type (
	PopulationArgs struct {
		City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
		Year int    `json:"year" description:"census year" enum:"2000|2010|2020"`
	}
	AreaArgs struct {
		City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
	}
)

// The whole of the tools' world, so the example needs no network beyond the
// model itself and always gives the same answer.
var (
	census = map[string]map[int]float64{
		"Tokyo":     {2000: 34.5, 2010: 36.9, 2020: 37.4},
		"Delhi":     {2000: 15.7, 2010: 21.9, 2020: 31.0},
		"Shanghai":  {2000: 14.2, 2010: 19.9, 2020: 27.8},
		"São Paulo": {2000: 17.0, 2010: 19.7, 2020: 22.0},
	}
	areaKm2 = map[string]int{
		"Tokyo": 2194, "Delhi": 1484, "Shanghai": 6341, "São Paulo": 1521,
	}
)

func population(_ context.Context, args PopulationArgs) (string, error) {
	millions := census[args.City][args.Year]
	fmt.Printf("  \033[2m→ population(%s, %d) = %.1fM\033[0m\n", args.City, args.Year, millions)
	return fmt.Sprintf("%.1f million", millions), nil
}

func area(_ context.Context, args AreaArgs) (string, error) {
	km2 := areaKm2[args.City]
	fmt.Printf("  \033[2m→ area(%s) = %d km²\033[0m\n", args.City, km2)
	return fmt.Sprintf("%d square kilometres", km2), nil
}

func main() {
	model := flag.String("model", "openai/gpt-4.1", "model reference, vendor/id")
	flag.Parse()

	question := strings.Join(flag.Args(), " ")
	if question == "" {
		question = "Which was denser in 2020, Delhi or Tokyo? Show the figures you used."
	}
	if err := run(*model, question); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref, question string) error {
	client, err := auth.Client(ref)
	if err != nil {
		return err
	}
	ctx := context.Background()

	// The argument type is never named here. It comes from each function's own
	// parameter, which is the only place it appears.
	tools := []ai.Tool{
		ai.Handle("population", "Population of a city in millions, for one census year.", population),
		ai.Handle("area", "Area of a city in square kilometres.", area),
	}

	messages := []ai.Message{ai.UserMessage(question)}

	// A turn limit rather than for{}: a model that keeps calling tools is a
	// bug somewhere, and an example that spins forever is a bad example.
	for turn := 1; turn <= 8; turn++ {
		response, err := client.Complete(ctx, messages, ai.WithTools(tools...))
		if err != nil {
			return err
		}

		calls := response.ToolCalls()
		if len(calls) == 0 {
			fmt.Printf("\n%s\n", response.Text())
			fmt.Printf("\n\033[2m%d turns · %d in / %d out\033[0m\n",
				turn, response.Usage.TotalInput(), response.Usage.Output)
			return nil
		}

		// RunTools validates, decodes and dispatches each call, and turns
		// anything that goes wrong into a result the model can act on rather
		// than an error that ends the conversation.
		results := ai.RunTools(ctx, tools, calls)
		for _, result := range results {
			if result.IsError {
				fmt.Printf("  \033[31m✗\033[0m %s → %s\n", result.ToolName, result.Content)
			}
		}

		// response.Message() carries every block the model produced, thinking
		// included; the results answer every call in the turn that follows.
		messages = append(messages, response.Message(), ai.ToolResultsMessage(results...))
	}
	return fmt.Errorf("the model was still calling tools after 8 turns")
}
