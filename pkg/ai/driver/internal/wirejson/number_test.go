package wirejson

import (
	"encoding/json"
	"reflect"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	openai "github.com/openai/openai-go/v3"
)

func TestObject(t *testing.T) {
	src := map[string]any{"nullSlice": []any(nil), "nullMap": map[string]any(nil), "nested": []any{map[string]any{"n": json.Number("9007199254740993")}, json.Number("1.234567890123456789"), "8", "001", "1.25"}}
	before, err := json.Marshal(src)
	if err != nil {
		t.Fatal(err)
	}
	out := Object(src)
	for name, marshal := range map[string]func(map[string]any) ([]byte, error){
		"standard": func(m map[string]any) ([]byte, error) { return json.Marshal(m) },
		"openai": func(m map[string]any) ([]byte, error) {
			p := openai.ChatCompletionNewParams{}
			p.SetExtraFields(m)
			return p.MarshalJSON()
		},
		"anthropic": func(m map[string]any) ([]byte, error) {
			p := anthropic.MessageNewParams{}
			p.SetExtraFields(m)
			return p.MarshalJSON()
		},
	} {
		t.Run(name, func(t *testing.T) {
			raw, err := marshal(out)
			if err != nil {
				t.Fatal(err)
			}
			var got map[string]json.RawMessage
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatal(err)
			}
			var want map[string]json.RawMessage
			_ = json.Unmarshal(before, &want)
			for key, expected := range want {
				if !reflect.DeepEqual(got[key], expected) {
					t.Fatalf("got %s, want %s in %s", raw, key, before)
				}
			}
			if _, err := marshal(Object(map[string]any{"bad": json.Number(`1,"injected":true`)})); err == nil {
				t.Fatal("invalid Number accepted")
			}
		})
	}
	out["nested"].([]any)[0].(map[string]any)["n"] = "changed"
	after, _ := json.Marshal(src)
	if string(after) != string(before) {
		t.Fatal("source was mutated")
	}
	if Object(nil) != nil {
		t.Fatal("nil map changed")
	}
}
