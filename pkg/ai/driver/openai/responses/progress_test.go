package responses

import (
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestToolArgumentProgress(t *testing.T) {
	s := sse(t,
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc","call_id":"c","name":"Write"}}`,
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc","call_id":"c","name":"Write"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":""}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"{\"text\":\""}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"中\"}"}`,
		`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc","call_id":"c","name":"Write","arguments":"{\"text\":\"中\"}"}}`, completed)
	ds, err := collect(t, driverFor(t, ai.Config{Model: ai.Model{BaseURL: s.URL}}))
	if err != nil {
		t.Fatal(err)
	}
	want := []int{len(`{"text":"`), len(`{"text":"中"}`)}
	n := 0
	for _, d := range ds {
		if p := d.Progress; p != nil {
			if n >= len(want) || p.InputBytes != want[n] || p.ToolCallID != "c" || p.ToolName != "Write" || d.Block.Type != "" {
				t.Fatalf("progress=%+v", d)
			}
			n++
		}
	}
	if n != len(want) {
		t.Fatalf("progress events=%d", n)
	}
	calls := toolCalls(ds)
	if len(calls) != 1 || calls[0].Input != `{"text":"中"}` {
		t.Fatalf("calls=%+v", calls)
	}
}
