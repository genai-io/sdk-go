// Command tools runs a tool-calling loop, which is four rules and a for loop.
//
// The model does not run your tools. It asks you to, you answer, and it
// continues — so the loop is yours to write, and the rules that make it work
// are the ones this file exists to show:
//
//   - Say what every parameter means. The schema is prompt text, and a model
//     given nothing but a field name is left guessing what to put in it.
//   - Check the arguments before running anything. A model's mistake handed
//     back as a tool result is something it can correct; the same mistake run
//     is whatever your tool does with a missing field.
//   - Answer every call in the turn that immediately follows. Every protocol
//     here rejects a conversation where one is left hanging.
//   - Append response.Message(), not ai.AssistantMessage(response.Text()).
//     The first carries the model's thinking and reasoning state forward; the
//     second silently drops it and a reasoning model starts over each turn.
//
// A tool here is a struct and a function. The struct is everything the model
// is told — its name, what it does, and every argument it may send — and the
// function is what happens when it calls. Nothing has to be kept in step by
// hand, which a switch on call.Name with a matching UnmarshalArgs in each arm
// quietly requires.
//
// Both tools below pin their cities to an enum, which is why the wrong city
// comes back as a mistake the model can correct rather than as a lookup that
// silently returns zero.
//
// Client.Run does the middle two rules for you, and answers a bad call rather
// than ending the conversation over it.
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

// One struct per tool, holding everything the model is told about it: the
// ai.Doc line is the tool itself, the fields are the arguments it may send.
// Every word of it is prompt text.
type Population struct {
	_ ai.Doc `name:"population" description:"Population of a city in millions, for one census year."`

	City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
	Year int    `json:"year" description:"census year" enum:"2000|2010|2020"`
}

type Area struct {
	_ ai.Doc `name:"area" description:"Area of a city in square kilometres."`

	City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
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

	// The whole of the tools' world, so the example needs no network beyond the
	// model itself and always gives the same answer. In a real program these
	// would be a database handle or an HTTP client — closed over exactly like
	// this, which is all a tool's dependencies ever are.
	census := map[string]map[int]float64{
		"Tokyo":     {2000: 34.5, 2010: 36.9, 2020: 37.4},
		"Delhi":     {2000: 15.7, 2010: 21.9, 2020: 31.0},
		"Shanghai":  {2000: 14.2, 2010: 19.9, 2020: 27.8},
		"São Paulo": {2000: 17.0, 2010: 19.7, 2020: 22.0},
	}
	areaKm2 := map[string]int{
		"Tokyo": 2194, "Delhi": 1484, "Shanghai": 6341, "São Paulo": 1521,
	}

	tools := []ai.Tool{
		ai.ToolFunc(func(_ context.Context, a Population) (string, error) {
			millions := census[a.City][a.Year]
			fmt.Printf("  \033[2m→ population(%s, %d) = %.1fM\033[0m\n", a.City, a.Year, millions)
			return fmt.Sprintf("%.1f million", millions), nil
		}),
		ai.ToolFunc(func(_ context.Context, a Area) (string, error) {
			km2 := areaKm2[a.City]
			fmt.Printf("  \033[2m→ area(%s) = %d km²\033[0m\n", a.City, km2)
			return fmt.Sprintf("%d square kilometres", km2), nil
		}),
	}

	// Run is the loop: complete, answer whatever the model asked for, repeat
	// until it stops asking. history is the whole conversation, so a follow-up
	// question continues from it.
	response, history, err := client.Run(context.Background(),
		[]ai.Message{ai.UserMessage(question)}, tools)
	if err != nil {
		return err
	}

	// What the model got wrong along the way, and was told about. RunTools
	// hands each of these back as a tool result rather than failing the turn,
	// which is why the conversation above still reached an answer.
	for _, message := range history {
		for _, result := range message.ToolResults() {
			if result.IsError {
				fmt.Printf("  \033[31m✗\033[0m %s → %s\n", result.ToolName, result.Content)
			}
		}
	}

	fmt.Printf("\n%s\n", response.Text())
	fmt.Printf("\n\033[2m%d messages · %d in / %d out\033[0m\n",
		len(history), response.Usage.TotalInput(), response.Usage.Output)
	return nil
}
