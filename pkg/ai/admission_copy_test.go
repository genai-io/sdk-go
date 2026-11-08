package ai

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestAdmissionCopiesDetachAllPayloads(t *testing.T) {
	definition := map[string]any{"nested": []any{map[string]any{"n": int64(9007199254740993)}}}
	temperature := 0.5
	result := &ToolResult{Content: Content{ImageBlock(Image{Data: "original"})}}
	req := &Request{
		Messages: []Message{{Content: Content{
			ImageBlock(Image{Data: "original"}),
			ToolCallBlock(ToolCall{Signature: []byte{1, 2}}),
			{Type: BlockToolResult, ToolResult: result},
			{Type: BlockToolResult, ToolResult: result},
			ReasoningBlock(ReasoningItem{Summary: "original"}),
		}}},
		Tools: []Schema{{Definition: definition}}, Schema: &Schema{Definition: definition},
		SamplingParams: map[string]any{"alias": definition}, Temperature: &temperature,
		StopSequences: []string{"original"}, ToolChoice: ToolChoiceRequired,
	}
	var snapshot *Request
	frozen, err := admitRequest(context.Background(), req, func(_ context.Context, got *Request) error {
		snapshot = got
		if n := got.Tools[0].Definition.(map[string]any)["nested"].([]any)[0].(map[string]any)["n"]; n != json.Number("9007199254740993") {
			t.Fatalf("number = %v", n)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	before, _ := json.Marshal(frozen)
	mutate := func(r *Request) {
		c := r.Messages[0].Content
		c[0].Image.Data = "changed"
		c[1].ToolCall.Signature[0] = 9
		c[2].ToolResult.Content[0].Image.Data = "changed"
		c[4].Reasoning.Summary = "changed"
		r.Tools[0].Definition.(map[string]any)["nested"].([]any)[0].(map[string]any)["n"] = 0
		r.Schema.Definition.(map[string]any)["other"] = true
		r.SamplingParams["alias"].(map[string]any)["other"] = true
		*r.Temperature = 9
		r.StopSequences[0] = "changed"
		r.ToolChoice = ToolChoiceNone
	}
	mutate(snapshot)
	// Repeated input aliases must also be detached within the snapshot.
	if snapshot.Messages[0].Content[3].ToolResult.Content[0].Image.Data != "original" {
		t.Fatal("snapshot retains content alias")
	}
	mutate(req)
	after, _ := json.Marshal(frozen)
	if string(before) != string(after) || frozen.ToolChoice != ToolChoiceRequired {
		t.Fatalf("dispatch mutated: %s", after)
	}
}

func TestAdmissionCopiesRefuseUnsupportedJSON(t *testing.T) {
	cycle := map[string]any{}
	cycle["self"] = cycle
	result := &ToolResult{}
	result.Content = Content{{ToolResult: result}}
	for name, req := range map[string]*Request{
		"schema_channel":  {Schema: &Schema{Definition: make(chan int)}},
		"tool_function":   {Tools: []Schema{{Definition: func() {}}}},
		"sampling_cycle":  {SamplingParams: cycle},
		"sampling_nan":    {SamplingParams: map[string]any{"n": math.NaN()}},
		"temperature_inf": {Temperature: newFloatForAdmission(math.Inf(1))},
		"content_cycle":   {Messages: []Message{{Content: result.Content}}},
	} {
		t.Run(name, func(t *testing.T) {
			called := false
			_, err := admitRequest(context.Background(), req, func(context.Context, *Request) error { called = true; return nil })
			var admission *AdmissionError
			if !errors.As(err, &admission) || called {
				t.Fatalf("err=%v called=%v", err, called)
			}
		})
	}
}

func newFloatForAdmission(v float64) *float64 { return &v }

func TestAdmissionCopiesNormalizeTypedJSON(t *testing.T) {
	type definition struct {
		Values []int64 `json:"values"`
	}
	values := []int64{9007199254740993}
	req := &Request{Schema: &Schema{Definition: &definition{Values: values}}, SamplingParams: map[string]any{"raw": json.RawMessage(`{"n":9007199254740993}`)}}
	frozen, snapshot, err := admissionCopies(req)
	if err != nil {
		t.Fatal(err)
	}
	values[0] = 0
	snapshot.Schema.Definition.(map[string]any)["values"].([]any)[0] = 0
	snapshot.SamplingParams["raw"].(map[string]any)["n"] = 0
	if got := frozen.Schema.Definition.(map[string]any)["values"].([]any)[0]; got != json.Number("9007199254740993") {
		t.Fatalf("typed definition number = %v", got)
	}
	if got := frozen.SamplingParams["raw"].(map[string]any)["n"]; got != json.Number("9007199254740993") {
		t.Fatalf("raw JSON number = %v", got)
	}
}

func TestAdmissionCopiesPreserveNativeValues(t *testing.T) {
	req := &Request{System: "raw\xff", Messages: []Message{{Content: Content{}}}, Tools: []Schema{}, StopSequences: []string{}}
	frozen, snapshot, err := admissionCopies(req)
	if err != nil || !reflect.DeepEqual(req, frozen) || !reflect.DeepEqual(req, snapshot) {
		t.Fatalf("copies=%+v / %+v err=%v", frozen, snapshot, err)
	}
}
