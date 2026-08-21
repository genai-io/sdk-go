// Command tools runs a small agent with two tools, printing the turn as it
// happens.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/tools -model anthropic/claude-opus-5 "how big is go.mod, and what time is it?"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/auth"
	"github.com/genai-io/sdk-go/pkg/san"

	_ "github.com/genai-io/sdk-go/pkg/llm/driver/all"
)

func main() {
	model := flag.String("model", "anthropic/claude-opus-5", "model reference, vendor/id")
	flag.Parse()

	prompt := strings.Join(flag.Args(), " ")
	if prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: tools [flags] <prompt>")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *model, prompt); err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, model, prompt string) error {
	client, err := auth.Open(model)
	if err != nil {
		return err
	}

	tools := san.NewToolSet()
	tools.Add(clockTool{})
	tools.Add(fileSizeTool{})

	agent, err := san.New(
		san.WithModel(client),
		san.WithSystem("You are a terse assistant. Use the tools when they answer the question."),
		san.WithTools(tools),
		san.WithMaxSteps(10),
	)
	if err != nil {
		return err
	}
	defer agent.Close()

	agent.SetMessages([]llm.Message{llm.User(prompt)})

	// Watch the event stream on the side while ThinkAct runs the turn.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for evt := range agent.Outbox() {
			switch evt.Type {
			case san.OnChunk:
				if chunk, ok := evt.Chunk(); ok && chunk.Type == llm.EventTextDelta {
					fmt.Print(chunk.Text)
				}
			case san.PreTool:
				if call, ok := evt.ToolCall(); ok {
					fmt.Printf("\n\033[2m→ %s(%s)\033[0m\n", call.Name, call.Input)
				}
			case san.PostTool:
				if res, ok := evt.ToolResult(); ok {
					fmt.Printf("\033[2m← %s\033[0m\n", firstLine(res.Content))
				}
			}
		}
	}()

	result, err := agent.ThinkAct(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("\n\n%s\n", result.Content)
	fmt.Printf("— %d steps, %d tool calls, %d in / %d out\n",
		result.Steps, result.ToolUses, result.Usage.TotalInput(), result.Usage.Output)
	return nil
}

// clockTool reports the current time.
type clockTool struct{}

func (clockTool) Name() string        { return "now" }
func (clockTool) Description() string { return "Return the current local time." }
func (clockTool) Schema() llm.Tool {
	return llm.Tool{
		Name:        "now",
		Description: "Return the current local time.",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
	}
}
func (clockTool) Execute(context.Context, map[string]any) (string, error) {
	return time.Now().Format(time.RFC1123), nil
}

// fileSizeTool reports the size of a file relative to the working directory.
type fileSizeTool struct{}

func (fileSizeTool) Name() string        { return "file_size" }
func (fileSizeTool) Description() string { return "Return the size in bytes of a file." }
func (fileSizeTool) Schema() llm.Tool {
	return llm.Tool{
		Name:        "file_size",
		Description: "Return the size in bytes of a file, relative to the working directory.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "description": "relative file path"},
			},
			"required": []string{"path"},
		},
	}
}

func (fileSizeTool) Execute(_ context.Context, input map[string]any) (string, error) {
	path, _ := input["path"].(string)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	// A tool reachable by a model needs its own boundary: refuse anything that
	// climbs out of the working directory.
	clean := filepath.Clean(path)
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") {
		return "", fmt.Errorf("path must stay inside the working directory")
	}
	info, err := os.Stat(clean)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%d bytes", info.Size()), nil
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i] + "…"
	}
	return s
}
