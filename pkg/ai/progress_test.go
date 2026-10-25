package ai

import (
	"reflect"
	"testing"
)

func TestProgressDoesNotChangeResponseOrBlockLifecycle(t *testing.T) {
	base := []Delta{{Block: TextBlock("a")}, {Block: TextBlock("b")}, {Block: ToolCallBlock(ToolCall{ID: "c", Name: "Write", Input: "{}"})}}
	wantEvents, want, err := collectEvents(t, drive(script{deltas: base}))
	if err != nil {
		t.Fatal(err)
	}
	with := []Delta{base[0], {Progress: &Progress{ToolCallID: "c", ToolName: "Write", InputBytes: 1}}, base[1], base[2]}
	// A progress-bearing delta still processes all of its other fields.
	with[2].Progress = &Progress{ToolCallID: "c", ToolName: "Write", InputBytes: 2}
	with[2].Usage = &Usage{Output: 7}
	with[2].ID = "response"
	var gotEvents []seen
	var got *Response
	progress := 0
	c := drive(script{deltas: with})
	for e, err := range c.Stream(t.Context(), []Message{UserMessage("hi")}) {
		if err != nil {
			t.Fatal(err)
		}
		switch e.Type {
		case EventProgress:
			progress++
			if e.Block.Type != "" || e.Response != nil || e.Progress.InputBytes != progress {
				t.Fatalf("bad progress: %+v", e)
			}
		case EventDone:
			got = e.Response
		default:
			gotEvents = append(gotEvents, seen{e.Type, e.Index, e.Block.Type, e.Block.Text})
		}
	}
	want.Usage.Output = 7
	want.ID = "response"
	if progress != 2 || !reflect.DeepEqual(gotEvents, wantEvents) || !reflect.DeepEqual(got, want) {
		t.Fatalf("progress=%d events=%+v response=%+v; want events=%+v response=%+v", progress, gotEvents, got, wantEvents, want)
	}
	completed, err := c.Complete(t.Context(), []Message{UserMessage("hi")})
	if err != nil || !reflect.DeepEqual(completed, want) {
		t.Fatalf("Complete=%+v, %v", completed, err)
	}
}

func TestConsumerCanBreakOnProgress(t *testing.T) {
	c := drive(script{deltas: []Delta{{Progress: &Progress{InputBytes: 1}}, {Block: TextBlock("must not yield after break")}}})
	for e, err := range c.Stream(t.Context(), []Message{UserMessage("hi")}) {
		if err != nil || e.Type != EventProgress {
			t.Fatalf("%+v %v", e, err)
		}
		break
	}
}
