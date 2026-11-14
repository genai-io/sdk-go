package ai

import (
	"encoding/json"
	"testing"
)

func TestDefinitionMapTypedNumbers(t *testing.T) {
	schema := Schema{Definition: struct {
		Const int64 `json:"const"`
		Enum  []any `json:"enum"`
	}{9007199254740993, []any{json.Number("1.234567890123456789"), "8", "001", "1.25"}}}
	got, err := json.Marshal(schema.DefinitionMap())
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"const":9007199254740993,"enum":[1.234567890123456789,"8","001","1.25"]}`
	if string(got) != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}
