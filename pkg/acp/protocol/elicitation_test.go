package protocol

import (
	"encoding/json"
	"testing"
)

func TestElicitationPrimitiveValues(t *testing.T) {
	for _, input := range []string{`"text"`, `42`, `1.5`, `true`, `["a","b"]`} {
		var value ElicitationContentValue
		if err := json.Unmarshal([]byte(input), &value); err != nil {
			t.Fatalf("%s: %v", input, err)
		}
		got, err := json.Marshal(value)
		if err != nil || string(got) != input {
			t.Fatalf("%s: %s %v", input, got, err)
		}
	}
	for _, input := range []string{`null`, `42`, `"request"`} {
		var value RequestID
		if err := json.Unmarshal([]byte(input), &value); err != nil {
			t.Fatalf("%s: %v", input, err)
		}
		got, err := json.Marshal(value)
		if err != nil || string(got) != input {
			t.Fatalf("%s: %s %v", input, got, err)
		}
	}
}

func TestElicitationFormScope(t *testing.T) {
	for _, scope := range []string{`"sessionId":"session","toolCallId":"tool"`, `"requestId":"request"`} {
		input := `{"mode":"form","message":"Input","requestedSchema":{},` + scope + `}`
		var req CreateElicitationRequest
		if err := json.Unmarshal([]byte(input), &req); err != nil {
			t.Fatal(err)
		}
		v := req.Value.(CreateElicitationRequestFormVariant)
		if !v.SessionID.Present && !v.RequestID.Present {
			t.Fatal("scope is not typed")
		}
		if len(v.Extra) != 0 {
			t.Fatal("scope hidden in extras")
		}
	}
}
