package protocol

import (
	"encoding/json"
	"testing"
)

func TestNumericBoundaryContract(t *testing.T) {
	for _, tc := range []struct {
		def, data string
		ok        bool
	}{
		{"InitializeRequest", `{"protocolVersion":65535}`, true},
		{"InitializeRequest", `{"protocolVersion":65536}`, false},
		{"InitializeRequest", `{"protocolVersion":1.000000000000000000001}`, false},
		{"InitializeRequest", `{"protocolVersion":9007199254740993}`, false},
		{"UsageUpdate", `{"used":18446744073709551615,"size":18446744073709551615}`, true},
		{"Cost", `{"amount":0.1,"currency":"USD"}`, true},
		// Arbitrary values do not undergo constrained numeric conversion.
		{"InitializeRequest", `{"protocolVersion":1,"_meta":{"x":18446744073709551616,"huge":1e10000,"tiny":1e-10000}}`, true},
	} {
		err := ValidateJSON(tc.def, []byte(tc.data))
		if (err == nil) != tc.ok {
			t.Fatalf("%s: %v", tc.data, err)
		}
	}
	var v UsageUpdate
	if err := json.Unmarshal([]byte(`{"used":18446744073709551615,"size":9007199254740993}`), &v); err != nil {
		t.Fatal(err)
	}
	if v.Used != ^uint64(0) || v.Size != 9007199254740993 {
		t.Fatal("integer precision lost")
	}
}

func TestTaggedTransportDoesNotFallBack(t *testing.T) {
	// This is schema-valid as a stdio server with unknown properties. A recognized
	// HTTP discriminator must nevertheless select HTTP and reject missing fields.
	var v MCPServer
	if err := json.Unmarshal([]byte(`{"type":"http","name":"m","command":"cmd","args":[],"env":[]}`), &v); err == nil {
		t.Fatal("HTTP incorrectly decoded as stdio")
	}
	// If both structural branches match, the explicit discriminator wins.
	if err := json.Unmarshal([]byte(`{"type":"http","name":"m","command":"cmd","args":[],"env":[],"url":"https://example.test","headers":[]}`), &v); err != nil {
		t.Fatal(err)
	}
	if _, ok := v.Value.(MCPServerHTTPVariant); !ok {
		t.Fatalf("got %T", v.Value)
	}
}

func TestExtensionRecoveryHintsAreStrict(t *testing.T) {
	for _, b := range []string{`{"protocolVersion":1,"clientCapabilities":"invalid"}`, `{"protocolVersion":1,"clientInfo":false}`} {
		var v InitializeRequest
		if err := json.Unmarshal([]byte(b), &v); err == nil {
			t.Fatal("recovery hints must not silently repair invalid input")
		}
	}
}

func TestPermissionOutcomeConstruction(t *testing.T) {
	for _, v := range []RequestPermissionResponse{
		{Outcome: RequestPermissionOutcome{Value: RequestPermissionOutcomeCancelledVariant{Outcome: "cancelled"}}},
		{Outcome: RequestPermissionOutcome{Value: RequestPermissionOutcomeSelectedVariant{Outcome: "selected", OptionID: "allow"}}},
	} {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		var got RequestPermissionResponse
		if err = json.Unmarshal(b, &got); err != nil {
			t.Fatal(err)
		}
	}
}
