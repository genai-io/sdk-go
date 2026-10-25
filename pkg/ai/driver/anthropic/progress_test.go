package anthropic

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestToolArgumentProgress(t *testing.T) {
	s := sse(t, done[0], toolStart(0, "a", "Write"), toolStart(1, "b", "Read"), toolInput(0, "{"), toolInput(0, ""), toolInput(1, "{}"), toolInput(0, "}"), blockStop(0), blockStop(1), done[1])
	ds, err := collect(t, context.Background(), driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}), &ai.Request{})
	if err != nil {
		t.Fatal(err)
	}
	want := []ai.Progress{{ToolCallID: "a", ToolName: "Write", InputBytes: 1}, {ToolCallID: "b", ToolName: "Read", InputBytes: 2}, {ToolCallID: "a", ToolName: "Write", InputBytes: 2}}
	n := 0
	for _, d := range ds {
		if p := d.Progress; p != nil {
			if n >= len(want) || *p != want[n] || d.Block.Type != "" {
				t.Fatalf("progress=%+v", d)
			}
			n++
		}
	}
	if n != len(want) {
		t.Fatalf("progress events=%d", n)
	}
	calls := toolCalls(ds)
	if len(calls) != 2 || calls[0].Input != "{}" || calls[1].Input != "{}" {
		t.Fatalf("calls=%+v", calls)
	}
}
