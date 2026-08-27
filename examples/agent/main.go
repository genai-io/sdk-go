// Command agent is a coding-assistant loop in about eighty lines: a model, one
// tool, a system prompt, and a session that survives the process.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/agent "how many Go files are in this directory?"
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/agent/session/jsonl"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
)

type shellArgs struct {
	Command string `json:"command" description:"the shell command to run"`
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: agent <prompt>")
	}
	ctx := context.Background()

	client, err := auth.Client("anthropic/claude-sonnet-5")
	if err != nil {
		log.Fatal(err)
	}

	cwd, _ := os.Getwd()

	// Assembling the prompt is this program's business; the agent takes the
	// result. Rebuild it and call SetSystem when something it mentions changes.
	system := fmt.Sprintf(
		"You are a terse assistant working in a Go repository. Answer in one or two sentences.\n\n"+
			"Working directory: %s\nToday: %s", cwd, time.Now().Format("2006-01-02"))

	shell := agent.ToolFunc("shell", "Run a read-only shell command and return its output.",
		func(ctx context.Context, args shellArgs) (agent.Result, error) {
			out, err := exec.CommandContext(ctx, "sh", "-c", args.Command).CombinedOutput()
			if err != nil {
				// Returning the error lets the model read what went wrong and
				// try again; failing the turn would just end the conversation.
				return agent.Result{}, fmt.Errorf("%s: %w", strings.TrimSpace(string(out)), err)
			}
			return agent.TextResult(string(out)), nil
		})

	a, err := agent.New(client,
		agent.WithID("example"),
		agent.WithSystem(system),
		agent.WithTools(shell),
		agent.WithMaxSteps(12),
		// The gate is where an application says no. This one refuses anything
		// that writes.
		agent.WithHooks(agent.Hook{
			PreTool: func(_ context.Context, c agent.PreToolContext) (agent.Decision, error) {
				for _, banned := range []string{"rm ", "mv ", ">", "curl", "git push"} {
					if strings.Contains(c.Call.Input, banned) {
						return agent.Decision{Block: true,
							Reason: "this session is read-only; " + banned + " is not allowed"}, nil
					}
				}
				return agent.Decision{}, nil
			},
		}))
	if err != nil {
		log.Fatal(err)
	}

	store, err := jsonl.Open(filepath.Join(os.TempDir(), "agent-example-sessions"))
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	// Resume the most recent session, or start one. What comes back is the
	// history to seed the agent with — the agent itself knows nothing about
	// storage.
	var resume string
	if list, err := store.List(ctx); err == nil && len(list) > 0 {
		resume = list[0].ID
	}
	rec, history, err := session.Open(ctx, store, resume)
	if err != nil {
		log.Fatal(err)
	}
	a.SetMessages(history)

	// One loop, and it owns the order: record first, then paint. A longer
	// programme would run this on its own goroutine and forward to a buffer
	// of its own; a one-shot has nothing else to do.
	var last agent.TurnEnd
	steps := 0
	for e, err := range a.Run(ctx, ai.UserMessage(strings.Join(os.Args[1:], " "))) {
		if err != nil {
			log.Fatalf("\n%v", err)
		}
		rec.Handle(ctx, e)
		render(e)
		switch v := e.(type) {
		case agent.MessageStart:
			if v.Attempt == 1 {
				steps++
			}
		case agent.TurnEnd:
			last = v
		}
	}

	fmt.Printf("\n\n— %s · %d steps · %d tokens · session %s\n",
		last.StopReason, steps, last.Usage.Total(), rec.ID())
	if err := rec.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "session was not fully written: %v\n", err)
	}
}

func render(e agent.Event) {
	switch v := e.(type) {
	case agent.MessageUpdate:
		if v.Delta.Type == ai.EventBlockDelta && v.Delta.Block.Type == ai.BlockText {
			fmt.Print(v.Delta.Block.Text)
		}
	case agent.ToolStart:
		fmt.Printf("\n[%s] %s\n", v.Name, strings.TrimSpace(v.Args))
	case agent.ToolEnd:
		if v.Err != nil {
			fmt.Printf("[failed] %v\n", v.Err)
		}
	}
}
