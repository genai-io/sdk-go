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
// The question below needs two lookups before it can be answered, so the loop
// really does go round more than once.
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

// PopulationArgs is both the tool's schema and the struct its arguments decode
// into. The tags are what the model reads: a field name alone does not say
// that "city" means one of a fixed few, and enum does.
type PopulationArgs struct {
	City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
	Year int    `json:"year" description:"census year" enum:"2000|2010|2020"`
}

// The whole of the tool's world, so the example needs no network beyond the
// model itself and always gives the same answer.
var census = map[string]map[int]float64{
	"Tokyo":     {2000: 34.5, 2010: 36.9, 2020: 37.4},
	"Delhi":     {2000: 15.7, 2010: 21.9, 2020: 31.0},
	"Shanghai":  {2000: 14.2, 2010: 19.9, 2020: 27.8},
	"São Paulo": {2000: 17.0, 2010: 19.7, 2020: 22.0},
}

func main() {
	model := flag.String("model", "openai/gpt-4.1", "model reference, vendor/id")
	flag.Parse()

	question := strings.Join(flag.Args(), " ")
	if question == "" {
		question = "Which grew faster between 2000 and 2020, Delhi or Tokyo? Give the figures."
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

	population := ai.ToolFor[PopulationArgs]("population",
		"Population of a city in millions, for one census year.")

	messages := []ai.Message{ai.UserMessage(question)}

	// A turn limit rather than for{}: a model that keeps calling tools is a
	// bug somewhere, and an example that spins forever is a bad example.
	for turn := 1; turn <= 6; turn++ {
		response, err := client.Complete(ctx, messages, ai.WithTools(population))
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

		// Carries every block the model produced, thinking included.
		messages = append(messages, response.Message())

		results := make([]ai.ToolResult, len(calls))
		for i, call := range calls {
			results[i] = answer(population, call)
		}
		// One user turn answering all of this turn's calls, which is what the
		// protocols expect — not one message per call.
		messages = append(messages, ai.ToolResultsMessage(results...))
	}
	return fmt.Errorf("the model was still calling tools after 6 turns")
}

// answer runs one call, or explains why it could not. Either way the model
// gets a result: a tool that fails silently leaves it waiting on nothing.
func answer(tool ai.Tool, call ai.ToolCall) ai.ToolResult {
	result := ai.ToolResult{ToolCallID: call.ID, ToolName: call.Name}

	// Before decoding, and before running: the schema the model was given is
	// the same one its arguments are measured against.
	if err := tool.ValidateArgs(call.Input); err != nil {
		fmt.Printf("  \033[31m✗\033[0m %s%s → %v\n", call.Name, call.Input, err)
		result.Content, result.IsError = err.Error(), true
		return result
	}
	args, err := ai.UnmarshalArgs[PopulationArgs](call)
	if err != nil {
		result.Content, result.IsError = err.Error(), true
		return result
	}

	millions := census[args.City][args.Year]
	fmt.Printf("  \033[2m→ %s(%s, %d) = %.1fM\033[0m\n", call.Name, args.City, args.Year, millions)
	result.Content = fmt.Sprintf("%.1f million", millions)
	return result
}
