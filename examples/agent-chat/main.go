// Command agent-chat holds a conversation, one exchange per line you type.
//
// It exists to show the decision people ask about first: Run advances the
// conversation by exactly one exchange and then returns. Repeating it is your
// loop, not a method on the agent — a CLI reads stdin, a server reads
// requests, and neither shape belongs in a library. The agent keeps the
// history between exchanges; nothing here threads it through.
//
// Ctrl-C ends the exchange in flight and hands the prompt back, instead of
// killing the process.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/agent-chat
//
//	export OPENAI_API_KEY=...
//	go run ./examples/agent-chat -model openai/gpt-4.1
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

type globArgs struct {
	Pattern string `json:"pattern" description:"a glob such as *.go or docs/*.md"`
}

func main() {
	model := flag.String("model", "anthropic/claude-sonnet-5", "model reference, vendor/id")
	flag.Parse()

	client, err := auth.Client(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	files := agent.ToolFunc("glob", "List files in the working directory matching a glob.",
		func(_ context.Context, args globArgs) (agent.Result, error) {
			matches, err := filepath.Glob(args.Pattern)
			if err != nil {
				return agent.Result{}, err
			}
			if len(matches) == 0 {
				return agent.TextResult("no files match " + args.Pattern), nil
			}
			return agent.TextResult(strings.Join(matches, "\n")), nil
		})

	a, err := agent.New(client,
		agent.WithSystem("You are a terse assistant. Answer in one or two sentences."),
		agent.WithTools(files),
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Ctrl-C interrupts the exchange, not the program. Between exchanges the
	// agent has nothing to interrupt and this does nothing.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt)
	go func() {
		for range sigs {
			a.Interrupt()
		}
	}()

	fmt.Printf("%s — ask something, or Ctrl-D to leave.\n", *model)

	ctx := context.Background()
	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n\033[1m>\033[0m ")
		if !in.Scan() {
			fmt.Println()
			return
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}

		// One exchange. The history it leaves behind is the starting point of
		// the next one, which is why nothing is passed back in.
		for e, err := range a.Run(ctx, ai.UserMessage(line)) {
			if err != nil {
				fmt.Fprintf(os.Stderr, "\n%v\n", err)
				break
			}
			render(e)
		}
	}
}

func render(e agent.Event) {
	switch v := e.(type) {
	case agent.MessageUpdate:
		fmt.Print(v.Text())
	case agent.ToolStart:
		fmt.Printf("\033[2m[%s %s]\033[0m ", v.Name, strings.TrimSpace(v.Args))
	case agent.TurnEnd:
		// An interrupted exchange still reports what it did; say so rather
		// than leaving the reader looking at half an answer.
		if v.StopReason != agent.StopEndTurn {
			fmt.Printf("\n\033[2m(%s)\033[0m", v.StopReason)
		}
		fmt.Println()
	}
}
