// Command mcp gives an agent the tools an MCP server advertises.
//
// The server here is npm's filesystem server, run as a child process. Anything
// that speaks the protocol works the same way, over stdio or over HTTP.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/mcp -dir . "what is in this directory, and how big is the largest file?"
//
//	go run ./examples/mcp -url http://localhost:3000/mcp "..."
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/mcp"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

func main() {
	model := flag.String("model", "anthropic/claude-opus-5", "model reference, vendor/id")
	dir := flag.String("dir", ".", "directory to give the filesystem server")
	url := flag.String("url", "", "reach an MCP server over HTTP instead of running one")
	flag.Parse()

	question := strings.Join(flag.Args(), " ")
	if question == "" {
		question = "What is in this directory, and which file is the largest?"
	}
	if err := run(*model, *dir, *url, question); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref, dir, url, question string) (err error) {
	ctx := context.Background()

	// Naming the server namespaces what it advertises — "files__read_file"
	// rather than "read_file" — which is what keeps two servers that both
	// advertise "search" from shadowing each other.
	server := mcp.Server{Name: "files", URL: url, Stderr: os.Stderr}
	if url == "" {
		server.Command = "npx"
		server.Args = []string{"-y", "@modelcontextprotocol/server-filesystem", dir}
	}

	// Closing the client is what stops the child process, so a failure to
	// close is a process left behind and worth reporting.
	c, err := mcp.Connect(ctx, server)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, c.Close()) }()

	tools, err := c.Tools(ctx)
	if err != nil {
		return err
	}
	for _, tool := range tools {
		fmt.Printf("  \033[2m· %s\033[0m\n", tool.Schema().Name)
	}

	client, err := auth.Client(ref)
	if err != nil {
		return err
	}

	// They are agent.Tool values, so there is nothing left to adapt: an MCP
	// tool and a Go function reach the loop the same way.
	a, err := agent.New(client, agent.WithTools(tools...), agent.WithMaxSteps(12))
	if err != nil {
		return err
	}

	for event, err := range a.Run(ctx, ai.UserMessage(question)) {
		if err != nil {
			return err
		}
		switch e := event.(type) {
		case agent.ToolStart:
			fmt.Printf("  \033[2m→ %s %s\033[0m\n", e.Name, e.Args)
		case agent.MessageUpdate:
			fmt.Print(e.Text())
		case agent.TurnEnd:
			fmt.Printf("\n\n\033[2m%s · %d in / %d out\033[0m\n",
				e.StopReason, e.Usage.TotalInput(), e.Usage.Output)
		}
	}
	return nil
}
