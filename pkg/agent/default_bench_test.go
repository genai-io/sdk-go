package agent_test

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// No network or disk: these benchmarks isolate the unchanged, no-callback paths.
type defaultBenchDriver struct{ tools int }

func (defaultBenchDriver) Name() string { return "benchmark" }
func (d defaultBenchDriver) Stream(_ context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		if d.tools > 0 && len(req.Messages[len(req.Messages)-1].Content.ToolResults()) == 0 {
			for i := range d.tools {
				if !yield(ai.Delta{Block: ai.ToolCallBlock(ai.ToolCall{ID: fmt.Sprint(i), Name: "noop", Input: `{}`})}, nil) {
					return
				}
			}
			return
		}
		yield(ai.Delta{Block: ai.TextBlock("done"), StopReason: ai.StopEndTurn}, nil)
	}
}

type defaultBenchTool struct{}

func (defaultBenchTool) Schema() ai.Schema { return ai.Schema{Name: "noop"} }
func (defaultBenchTool) Run(context.Context, ai.ToolCall) (agent.Result, error) {
	return agent.TextResult("done"), nil
}

func BenchmarkDefaultStream(b *testing.B) {
	for _, count := range []int{1, 100} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			client := ai.NewClientWithDriver(defaultBenchDriver{}, ai.Model{ID: "stub", API: "stub"})
			messages := make([]ai.Message, count)
			for i := range messages {
				messages[i] = ai.UserMessage(strings.Repeat("x", 512))
			}
			b.ReportAllocs()
			for b.Loop() {
				for _, err := range client.Stream(context.Background(), messages) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

func BenchmarkDefaultAgentRun(b *testing.B) {
	for _, tools := range []int{0, 1, 8} {
		b.Run(fmt.Sprint(tools), func(b *testing.B) {
			client := ai.NewClientWithDriver(defaultBenchDriver{tools: tools}, ai.Model{ID: "stub", API: "stub"})
			a, err := agent.New(client, agent.WithTools(defaultBenchTool{}))
			if err != nil {
				b.Fatal(err)
			}
			input := ai.UserMessage("hello")
			b.ReportAllocs()
			for b.Loop() {
				a.SetMessages(nil)
				for event, err := range a.Run(context.Background(), input) {
					if err != nil {
						b.Fatal(err)
					}
					if end, ok := event.(agent.TurnEnd); ok && end.Err != nil {
						b.Fatal(end.Err)
					}
				}
			}
		})
	}
}
