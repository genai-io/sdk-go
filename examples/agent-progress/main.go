// Command agent-progress runs a tool that takes a while and shows its work.
//
// Two seams that a short example usually skips:
//
//   - agent.Report sends a partial result from inside a running tool. It
//     arrives as ToolUpdate, so an interface can show a scan progressing
//     instead of freezing until it finishes. It reaches the tool through the
//     context, so a tool with nothing to report declares nothing.
//   - Result.Content goes to the model; Result.Details goes to you. The model
//     gets the count it asked for, this program gets the file list it needs to
//     draw, and neither pays for the other's formatting on every later turn.
//
// The model is asked for two counts at once, so the batch runs in parallel and
// the two tools report over each other — which is the point of ToolUpdate
// carrying the call's ID.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/agent-progress "how many Go files are here, and how many Markdown ones?"
package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

type countArgs struct {
	Extension string `json:"extension" description:"file extension to count, with the dot, such as .go"`
}

// found is what the scan collected. It rides on Result.Details, so it reaches
// the interface as a Go value rather than as text something has to parse back.
type found struct {
	Extension string
	Paths     []string
}

func main() {
	model := flag.String("model", "anthropic/claude-sonnet-5", "model reference, vendor/id")
	root := flag.String("dir", ".", "directory to scan")
	flag.Parse()

	prompt := strings.Join(flag.Args(), " ")
	if prompt == "" {
		prompt = "How many Go files are here, and how many Markdown ones?"
	}

	client, err := auth.Client(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	count := agent.ToolFunc("count_files", "Count files with a given extension, recursively.",
		func(ctx context.Context, args countArgs) (agent.Result, error) {
			var paths []string
			err := filepath.WalkDir(*root, func(path string, d fs.DirEntry, err error) error {
				if err != nil || d.IsDir() {
					return nil //nolint:nilerr // an unreadable directory is not worth failing the count
				}
				if !strings.HasSuffix(path, args.Extension) {
					return nil
				}
				paths = append(paths, path)

				// Slow enough to see. Report replaces rather than appends, so
				// this sends the running total, not the delta.
				time.Sleep(40 * time.Millisecond)
				agent.Report(ctx, agent.TextResult(
					fmt.Sprintf("%d %s files so far", len(paths), args.Extension)))
				return ctx.Err()
			})
			if err != nil {
				return agent.Result{}, err
			}
			return agent.Result{
				Content: ai.TextContent(fmt.Sprintf("%d", len(paths))),
				Details: found{Extension: args.Extension, Paths: paths},
			}, nil
		})

	a, err := agent.New(client,
		agent.WithSystem("You are terse. Use the tools rather than guessing, and answer in one sentence."),
		agent.WithTools(count),
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	ctx := context.Background()
	for e, err := range a.Run(ctx, ai.UserMessage(prompt)) {
		if err != nil {
			fmt.Fprintf(os.Stderr, "\n%v\n", err)
			os.Exit(1)
		}
		render(e)
	}
}

func render(e agent.Event) {
	switch v := e.(type) {
	case agent.MessageUpdate:
		fmt.Print(v.Text())
	case agent.ToolStart:
		fmt.Printf("\n\033[2m%s(%s)\033[0m\n", v.Name, strings.TrimSpace(v.Args))
	case agent.ToolUpdate:
		// Two tools run at once here, so the ID is what says which one this
		// is. Overwriting a single line would blend them together.
		fmt.Printf("\033[2m  %s · %s\033[0m\n", short(v.ID), v.Partial.Text())
	case agent.ToolEnd:
		if v.Err != nil {
			fmt.Printf("  failed: %v\n", v.Err)
			return
		}
		// Details never went to the model; it is here for the interface.
		if f, ok := v.Result.Details.(found); ok {
			fmt.Printf("  \033[2m%s → %s files", f.Extension, v.Result.Text())
			if len(f.Paths) > 0 {
				fmt.Printf(", first is %s", f.Paths[0])
			}
			fmt.Print("\033[0m\n")
		}
	case agent.TurnEnd:
		// A turn that failed says so here and nowhere else.
		if v.Err != nil {
			fmt.Fprintf(os.Stderr, "\n%v\n", v.Err)
		}
		fmt.Printf("\n\n\033[2m— %s · %d tokens\033[0m\n", v.StopReason, v.Usage.Total())
	}
}

// short trims a provider's call ID to something that fits beside the output.
func short(id string) string {
	if len(id) > 8 {
		return id[len(id)-8:]
	}
	return id
}
