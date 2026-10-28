package protocol

import (
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
)

func TestSchemaPin(t *testing.T) {
	h := sha1.New()
	_, _ = fmt.Fprintf(h, "blob %d\x00", len(schemaJSON))
	_, _ = h.Write(schemaJSON)
	if got := hex.EncodeToString(h.Sum(nil)); got != SchemaBlob {
		t.Fatal(got)
	}
	if got := len(supportedDefinitions); got != 108 {
		t.Fatalf("generated closure has %d definitions, want 108", got)
	}
}
func TestOptionalAndUnknown(t *testing.T) {
	for _, input := range []string{`{"protocolVersion":1}`, `{"protocolVersion":1,"clientInfo":null}`, `{"protocolVersion":1,"clientInfo":{"name":"test","version":"1"}}`, `{"protocolVersion":1,"_meta":{"large":9007199254740993123},"future":{"n":9007199254740993123}}`} {
		var v InitializeRequest
		if err := json.Unmarshal([]byte(input), &v); err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(input, "9007199254740993123") && !strings.Contains(string(b), "9007199254740993123") {
			t.Fatal(string(b))
		}
		var want, got map[string]json.RawMessage
		if err := json.Unmarshal([]byte(input), &want); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatal(err)
		}
		if len(want) != len(got) {
			t.Fatalf("%s -> %s", input, b)
		}
		if _, ok := want["clientInfo"]; ok != v.ClientInfo.Present {
			t.Fatal("presence")
		}
	}
	if _, err := json.Marshal(InitializeRequest{ProtocolVersion: 1, ClientCapabilities: Null[ClientCapabilities]()}); err == nil {
		t.Fatal("optional non-null accepted null")
	}
}
func TestValidation(t *testing.T) {
	for _, tc := range []struct {
		def, data string
		valid     bool
	}{
		{"InitializeRequest", `{}`, false},
		{"InitializeRequest", `{"protocolVersion":1,"clientInfo":null}`, true},
		{"InitializeRequest", `{"protocolVersion":1,"clientCapabilities":null}`, false},
		{"NewSessionRequest", `{"cwd":"/","mcpServers":[]}`, true},
		{"NewSessionRequest", `{"cwd":"/","mcpServers":null}`, false},
		{"NewSessionRequest", `{"cwd":"/"}`, false},
		{"PromptRequest", `{"sessionId":"s","prompt":[]}`, true},
		{"PromptRequest", `{"sessionId":"s","prompt":null}`, false},
		{"PromptRequest", `{"sessionId":"s","prompt":[{"type":"text"}]}`, false},
		{"SessionConfigOptionCategory", `"future_category"`, true},
	} {
		t.Run(tc.def+tc.data, func(t *testing.T) {
			err := ValidateJSON(tc.def, []byte(tc.data))
			if (err == nil) != tc.valid {
				t.Fatalf("valid=%v: %v", tc.valid, err)
			}
		})
	}
	if _, err := json.Marshal(NewSessionRequest{CWD: "/"}); err == nil {
		t.Fatal("nil required array")
	}
	if _, err := json.Marshal(NewSessionRequest{CWD: "/", MCPServers: []MCPServer{}}); err != nil {
		t.Fatal(err)
	}
}
func TestUnions(t *testing.T) {
	cases := []struct {
		input string
		dst   any
		want  any
	}{
		{`{"type":"text","text":"hello"}`, &ContentBlock{}, ContentBlockTextVariant{}},
		{`{"type":"image","data":"AA==","mimeType":"image/png"}`, &ContentBlock{}, ContentBlockImageVariant{}},
		{`{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hi"}}`, &SessionUpdate{}, SessionUpdateAgentMessageChunkVariant{}},
		{`{"name":"m","command":"cmd","args":[],"env":[]}`, &MCPServer{}, MCPServerStdioVariant{}},
		{`{"type":"http","name":"m","url":"https://example.test","headers":[]}`, &MCPServer{}, MCPServerHTTPVariant{}},
		{`{"type":"sse","name":"m","url":"https://example.test","headers":[]}`, &MCPServer{}, MCPServerSSEVariant{}},
		{`{"uri":"file:///x","text":"hello"}`, &EmbeddedResourceResource{}, nil},
		{`{"uri":"file:///x","blob":"AA=="}`, &EmbeddedResourceResource{}, nil},
		{`[{"value":"v","name":"Value"}]`, &SessionConfigSelectOptions{}, SessionConfigSelectOptionsUngroupedVariant{}},
		{`[{"group":"g","name":"Group","options":[]}]`, &SessionConfigSelectOptions{}, SessionConfigSelectOptionsGroupedVariant{}},
		{`[]`, &SessionConfigSelectOptions{}, SessionConfigSelectOptionsUngroupedVariant{}},
		{`{"id":"x","name":"X","type":"boolean","currentValue":true}`, &SessionConfigOption{}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			if err := json.Unmarshal([]byte(tc.input), tc.dst); err != nil {
				t.Fatal(err)
			}
			if tc.want != nil {
				var value any
				switch v := tc.dst.(type) {
				case *ContentBlock:
					value = v.Value
				case *SessionUpdate:
					value = v.Value
				case *MCPServer:
					value = v.Value
				case *SessionConfigSelectOptions:
					value = v.Value
				}
				if fmt.Sprintf("%T", value) != fmt.Sprintf("%T", tc.want) {
					t.Fatalf("got %T want %T", value, tc.want)
				}
			}
			b, err := json.Marshal(tc.dst)
			if err != nil {
				t.Fatal(err)
			}
			if !json.Valid(b) {
				t.Fatal(string(b))
			}
		})
	}
	var v ContentBlock
	if err := json.Unmarshal([]byte(`{"type":"future","text":"hi"}`), &v); err == nil {
		t.Fatal("unknown tag")
	}
	if err := json.Unmarshal([]byte(`{"type":"text","data":"AA==","mimeType":"image/png"}`), &v); err == nil {
		t.Fatal("mismatched tag")
	}
	var r EmbeddedResourceResource
	if err := json.Unmarshal([]byte(`{"uri":"x","text":"a","blob":"b"}`), &r); err == nil {
		t.Fatal("ambiguous resource")
	}
}
func TestConstruction(t *testing.T) {
	v := PromptRequest{SessionID: "s", Prompt: []ContentBlock{{Value: ContentBlockTextVariant{Type: "text", Text: "hello"}}}}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"type":"text"`)) {
		t.Fatal(string(b))
	}
}
func TestConcurrentValidation(t *testing.T) {
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 5 {
				if err := ValidateJSON("InitializeRequest", []byte(`{"protocolVersion":1}`)); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}
