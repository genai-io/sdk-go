package chat

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestToolArgumentProgress(t *testing.T) {
	s := sse(t,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"Write","arguments":"{"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"Write","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, "[DONE]")
	ds, err := collect(t, context.Background(), driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, d := range ds {
		if p := d.Progress; p != nil {
			n++
			if p.InputBytes != n || p.ToolCallID != "c" || p.ToolName != "Write" || d.Block.Type != "" {
				t.Fatalf("progress=%+v", d)
			}
		}
	}
	if n != 2 {
		t.Fatalf("progress events=%d", n)
	}
	calls := toolCalls(ds)
	if len(calls) != 1 || calls[0].Input != "{}" {
		t.Fatalf("calls=%+v", calls)
	}
}
