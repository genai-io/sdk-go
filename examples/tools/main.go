// Command tools runs a tool-calling loop, which is four rules and a for loop.
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

// One struct per tool, holding exactly what the model may send it. Every word
// of it is prompt text.
type PopulationArgs struct {
	City string `json:"city" description:"the city to look up" enum:"Tokyo|Delhi|Shanghai|São Paulo"`
	Year int    `json:"year" description:"census year" enum:"2000|2010|2020"`
}

type AreaArgs struct {
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
		ai.ToolFunc("population", "Population of a city in millions, for one census year.",
			func(_ context.Context, a PopulationArgs) (string, error) {
				millions := census[a.City][a.Year]
				fmt.Printf("  \033[2m→ population(%s, %d) = %.1fM\033[0m\n", a.City, a.Year, millions)
				return fmt.Sprintf("%.1f million", millions), nil
			}),
		ai.ToolFunc("area", "Area of a city in square kilometres.",
			func(_ context.Context, a AreaArgs) (string, error) {
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
