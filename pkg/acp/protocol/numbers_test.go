package protocol

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

func TestArbitraryNumbersAreOpaque(t *testing.T) {
	for _, token := range []string{"18446744073709551616", "1e10000", "1e-10000", "1e1000000", "1e-1000000"} {
		b := []byte(`{"protocolVersion":1,"_meta":{"n":` + token + `},"future":{"n":` + token + `}}`)
		if err := ValidateJSON("InitializeRequest", b); err != nil {
			t.Fatal(err)
		}
		var v InitializeRequest
		if err := json.Unmarshal(b, &v); err != nil {
			t.Fatal(err)
		}
		out, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(out, []byte(`"_meta":{"n":`+token+`}`)) || !bytes.Contains(out, []byte(`"future":{"n":`+token+`}`)) {
			t.Fatalf("arbitrary number changed: %s", out)
		}
		// Raw tool values are unconstrained even when they are scalar numbers.
		tool := []byte(`{"toolCallId":"t","rawInput":` + token + `,"rawOutput":{"n":` + token + `}}`)
		var call ToolCallUpdate
		if err := json.Unmarshal(tool, &call); err != nil {
			t.Fatal(err)
		}
		out, err = json.Marshal(call)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(out, []byte(`"rawInput":`+token)) {
			t.Fatal(string(out))
		}
	}
}

func TestUnionUnknownNumericProperties(t *testing.T) {
	b := []byte(`{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hello"},"used":1e10000,"cost":{"amount":1e10000}}`)
	if err := ValidateJSON("SessionUpdate", b); err != nil {
		t.Fatal(err)
	}
	var v SessionUpdate
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"used":1e10000`)) {
		t.Fatal(string(out))
	}
	for _, definition := range []string{"StopReason", "SessionId", "ContentBlock"} {
		if err := ValidateJSON(definition, []byte("1e10000")); err == nil {
			t.Fatalf("numeric token accepted for %s", definition)
		}
	}
}

func TestIntegralSpellings(t *testing.T) {
	for _, token := range []string{"1", "1.0", "1e0", "10e-1", "0.1e1", "100000e-5", "1.000e+0"} {
		t.Run(token, func(t *testing.T) {
			var version ProtocolVersion
			if err := json.Unmarshal([]byte(token), &version); err != nil {
				t.Fatal(err)
			}
			if version != 1 {
				t.Fatal(version)
			}
			b := []byte(`{"protocolVersion":` + token + `}`)
			if err := ValidateJSON("InitializeRequest", b); err != nil {
				t.Fatal(err)
			}
			var init InitializeRequest
			if err := json.Unmarshal(b, &init); err != nil {
				t.Fatal(err)
			}
			if init.ProtocolVersion != 1 {
				t.Fatal(init)
			}
			var usage UsageUpdate
			if err := json.Unmarshal([]byte(`{"used":`+token+`,"size":`+token+`}`), &usage); err != nil {
				t.Fatal(err)
			}
			if usage.Used != 1 || usage.Size != 1 {
				t.Fatal(usage)
			}
			var optional Optional[uint64]
			if err := json.Unmarshal([]byte(token), &optional); err != nil {
				t.Fatal(err)
			}
			if !optional.Present || optional.Null || optional.Value != 1 {
				t.Fatal(optional)
			}
			var versions Optional[[]ProtocolVersion]
			if err := json.Unmarshal([]byte("["+token+"]"), &versions); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(versions.Value, []ProtocolVersion{1}) {
				t.Fatal(versions)
			}
			var nested Optional[map[string][]int64]
			if err := json.Unmarshal([]byte(`{"n":[`+token+`]}`), &nested); err != nil {
				t.Fatal(err)
			}
			if nested.Value["n"][0] != 1 {
				t.Fatal(nested)
			}
		})
	}
	for _, token := range []string{"65535.0", "655350e-1"} {
		var v ProtocolVersion
		if err := json.Unmarshal([]byte(token), &v); err != nil || v != 65535 {
			t.Fatalf("%s: %v", token, err)
		}
	}
	for _, token := range []string{"18446744073709551615.0", "184467440737095516150e-1"} {
		var v UsageUpdate
		if err := json.Unmarshal([]byte(`{"used":`+token+`,"size":0e10000}`), &v); err != nil || v.Used != ^uint64(0) {
			t.Fatalf("%s: %v", token, err)
		}
	}
	for _, token := range []string{"-0.0", "0e10000", "0e-10000"} {
		var v ProtocolVersion
		if err := json.Unmarshal([]byte(token), &v); err != nil || v != 0 {
			t.Fatalf("%s: %v", token, err)
		}
	}
}

func TestConstrainedNumbersRejectWithoutRounding(t *testing.T) {
	for _, token := range []string{"1e-10000", "1e10000", "1.0000000000000000001", "65536.0", "\"1e0\"", "true", "null"} {
		if err := ValidateJSON("ProtocolVersion", []byte(token)); err == nil {
			t.Fatalf("validated invalid integer %s", token)
		}
		var v ProtocolVersion
		if err := json.Unmarshal([]byte(token), &v); err == nil {
			t.Fatalf("decoded invalid integer %s", token)
		}
	}
	for _, token := range []string{"-1e0", "18446744073709551616.0", "1e-10000", "1e10000", "\"1\""} {
		b := []byte(`{"used":` + token + `,"size":1}`)
		if err := ValidateJSON("UsageUpdate", b); err == nil {
			t.Fatalf("validated invalid uint64 %s", token)
		}
		var v UsageUpdate
		if err := json.Unmarshal(b, &v); err == nil {
			t.Fatalf("decoded invalid uint64 %s", token)
		}
	}
	var optional Optional[ProtocolVersion]
	if err := json.Unmarshal([]byte("65536e0"), &optional); err == nil {
		t.Fatal("optional alias bypassed bound")
	}
	if optional.Present {
		t.Fatal("failed optional decode mutated receiver")
	}
	if err := json.Unmarshal([]byte("null"), &optional); err != nil {
		t.Fatal(err)
	}
	if !optional.Present || !optional.Null {
		t.Fatal(optional)
	}
}
