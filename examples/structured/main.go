// Command structured asks for an answer shaped like a Go type and gets one
// back, decoded, in a single call.
//
//	meeting, err := ai.CompleteAs[Meeting](ctx, client, messages)
//
//	export OPENAI_API_KEY=...
//	go run ./examples/structured
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/structured -model anthropic/claude-opus-5
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"
	"github.com/genai-io/sdk-go/pkg/ai/jsonschema"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

// Meeting is the shape of the answer, and the only place that shape is written
// down. Every tag here is prompt text.
type Meeting struct {
	Title      string   `json:"title" description:"one line, under ten words"`
	Attendees  []string `json:"attendees" description:"names only, as written in the notes"`
	Decisions  []string `json:"decisions" description:"each decision as a full sentence" maxItems:"5"`
	Urgency    string   `json:"urgency" description:"how soon this needs follow-up" enum:"none|week|today"`
	Confidence float64  `json:"confidence" description:"how clear the notes were" minimum:"0" maximum:"1"`
}

const notes = `
Weds standup, 20 min. Priya, Marcus, and Chen (dialled in, dropped twice).
Marcus walked through the latency numbers — p99 is up 40% since the cache
change last Thursday. Agreed to roll that back today rather than debug it in
prod. Priya will own the rollback. Chen raised that the same change is queued
for the EU region on Friday; we're pulling it from that release too. Parking
the redesign discussion until next week, nobody had read the doc.
`

func main() {
	model := flag.String("model", "openai/gpt-4.1", "model reference, vendor/id")
	showSchema := flag.Bool("schema", false, "print the derived schema and exit")
	flag.Parse()

	if *showSchema {
		out, _ := json.MarshalIndent(jsonschema.For[Meeting](), "", "  ")
		fmt.Println(string(out))
		return
	}
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

	meeting, err := ai.CompleteAs[Meeting](context.Background(), client,
		[]ai.Message{ai.UserMessage("Summarise these notes:\n" + notes)},
		ai.WithSystem("You extract structure from meeting notes. Be literal; do not infer."))
	if err != nil {
		return err
	}

	fmt.Printf("\033[1m%s\033[0m\n", meeting.Title)
	fmt.Printf("attendees   %s\n", strings.Join(meeting.Attendees, ", "))
	for i, decision := range meeting.Decisions {
		if i == 0 {
			fmt.Printf("decisions   %s\n", decision)
			continue
		}
		fmt.Printf("            %s\n", decision)
	}
	fmt.Printf("urgency     %s\n", meeting.Urgency)
	fmt.Printf("confidence  %.2f\n", meeting.Confidence)

	// Urgency can only ever hold one of three values, because the enum went to
	// the model as part of the schema rather than as a hopeful sentence in the
	// prompt. Switching on it needs no default-for-garbage branch.
	fmt.Println()
	switch meeting.Urgency {
	case "today":
		fmt.Println("→ page someone")
	case "week":
		fmt.Println("→ put it on the sprint")
	case "none":
		fmt.Println("→ nothing to do")
	}
	fmt.Println("\n\033[2mRun with -schema to see what the model was actually sent.\033[0m")
	return nil
}
