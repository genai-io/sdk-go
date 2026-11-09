package ai

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolChoiceJSON(t *testing.T) {
	for _, tc := range []struct {
		choice ToolChoice
		wire   string
	}{
		{ToolChoiceAuto, `"auto"`}, {ToolChoiceNone, `"none"`}, {ToolChoiceRequired, `"required"`}, {ToolChoiceNamed("lookup"), `{"tool":"lookup"}`},
	} {
		b, err := json.Marshal(tc.choice)
		if err != nil || string(b) != tc.wire {
			t.Fatalf("marshal = %s, %v", b, err)
		}
		var got ToolChoice
		if err := got.UnmarshalJSON(b); err != nil || got != tc.choice {
			t.Fatalf("roundtrip = %+v, %v", got, err)
		}
	}
	for _, wire := range []string{`null`, `"other"`, `""`, `{}`, `{"tool":""}`, `{"tool":null}`, `{"Tool":"x"}`, `{"tool":"x","extra":1}`, `[]`, `true`, `1`, `{"tool":`, `"auto" false`, `{"tool":"x"} {}`} {
		got := ToolChoiceRequired
		if err := got.UnmarshalJSON([]byte(wire)); err == nil || got != ToolChoiceRequired {
			t.Errorf("accepted or mutated on %s", wire)
		}
	}
	if _, err := json.Marshal(ToolChoiceNamed("")); err == nil {
		t.Error("marshaled empty name")
	}
}

func TestRequestJSONRoundTrip(t *testing.T) {
	temp := 0.0
	for _, choice := range []ToolChoice{ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceNamed("lookup")} {
		schema := Schema{Name: "lookup", Description: "lookup", Definition: map[string]any{"type": "object", "maxProperties": json.Number("9007199254740993")}}
		want := Request{System: "system", Messages: []Message{UserMessage("hello")}, Tools: []Schema{schema}, MaxTokens: 42, Temperature: &temp, Effort: EffortHigh, ToolChoice: choice, StopSequences: []string{"stop"}, CacheRetention: CacheLong, Schema: &schema, SamplingParams: map[string]any{"seed": json.Number("9007199254740993")}}
		b, err := json.Marshal(want)
		if err != nil {
			t.Fatal(err)
		}
		var got Request
		dec := json.NewDecoder(bytes.NewReader(b))
		dec.UseNumber()
		if err := dec.Decode(&got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("roundtrip mismatch: got %+v want %+v", got, want)
		}
	}
}
