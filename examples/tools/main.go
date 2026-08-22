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
// A tool here is one Go type. Its fields are the arguments, its tags say what
// each one means, and its Tool and Run methods say what it is called and what
// it does — so the string the model calls sits next to the fields it will fill
// in, and next to the code that reads them. Nothing has to be kept in step by
// hand, which a switch on call.Name with a matching UnmarshalArgs in each arm
// quietly requires.
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

// A type per tool, carrying everything the model is told about it: the
// arguments as fields, what each one means as tags, and the name and purpose
// as its Tool method. The tags matter — a field name alone does not say that
// "city" means one of a fixed few, and enum does.
type Population struct {
	City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
	Year int    `json:"year" description:"census year" enum:"2000|2010|2020"`

	census map[string]map[int]float64 // unexported: ours, never the model's
}

func (Population) Tool() ai.ToolInfo {
	return ai.ToolInfo{
		Name:        "population",
		Description: "Population of a city in millions, for one census year.",
	}
}

func (p Population) Run(context.Context) (string, error) {
	millions := p.census[p.City][p.Year]
	fmt.Printf("  \033[2m→ population(%s, %d) = %.1fM\033[0m\n", p.City, p.Year, millions)
	return fmt.Sprintf("%.1f million", millions), nil
}

type Area struct {
	City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`

	km2 map[string]int
}

func (Area) Tool() ai.ToolInfo {
	return ai.ToolInfo{Name: "area", Description: "Area of a city in square kilometres."}
}

func (a Area) Run(context.Context) (string, error) {
	km2 := a.km2[a.City]
	fmt.Printf("  \033[2m→ area(%s) = %d km²\033[0m\n", a.City, km2)
	return fmt.Sprintf("%d square kilometres", km2), nil
}

// The whole of the tools' world, so the example needs no network beyond the
// model itself and always gives the same answer. In a real program these would
// be a database handle or an HTTP client, handed to the tool the same way.
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

	// One line per tool, and everything about it is on its own type. What is
	// passed here is the tool's dependencies; the model fills in the rest.
	tools := []ai.Tool{
		ai.ToolOf(Population{census: census}),
		ai.ToolOf(Area{km2: areaKm2}),
	}

	// Run is the loop: complete, answer whatever the model asked for, repeat
	// until it stops asking. history is the whole conversation, so a follow-up
	// question continues from it.
	response, history, err := client.Run(context.Background(),
		[]ai.Message{ai.UserMessage(question)}, ai.WithTools(tools...))
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
