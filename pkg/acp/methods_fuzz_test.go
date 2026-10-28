package acp_test

import (
	"encoding/json"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func FuzzProtocolRoundTrip(f *testing.F) {
	constructors := []struct {
		name string
		new  func() any
	}{
		{"InitializeRequest", func() any { return new(protocol.InitializeRequest) }},
		{"InitializeResponse", func() any { return new(protocol.InitializeResponse) }},
		{"NewSessionRequest", func() any { return new(protocol.NewSessionRequest) }},
		{"NewSessionResponse", func() any { return new(protocol.NewSessionResponse) }},
		{"PromptRequest", func() any { return new(protocol.PromptRequest) }},
		{"PromptResponse", func() any { return new(protocol.PromptResponse) }},
		{"CancelNotification", func() any { return new(protocol.CancelNotification) }},
		{"SessionNotification", func() any { return new(protocol.SessionNotification) }},
		{"RequestPermissionRequest", func() any { return new(protocol.RequestPermissionRequest) }},
		{"RequestPermissionResponse", func() any { return new(protocol.RequestPermissionResponse) }},
	}
	for i, seed := range []string{
		`{"protocolVersion":1}`,
		`{"protocolVersion":1,"authMethods":[]}`,
		`{"cwd":"/fixture","mcpServers":[]}`,
		`{"sessionId":"fixture"}`,
		`{"sessionId":"fixture","prompt":[{"type":"text","text":"hello"}]}`,
		`{"stopReason":"end_turn"}`,
		`{"sessionId":"fixture"}`,
		`{"sessionId":"fixture","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hello"}}}`,
		`{"sessionId":"fixture","toolCall":{"toolCallId":"t"},"options":[]}`,
		`{"outcome":{"outcome":"cancelled"}}`,
	} {
		f.Add(uint8(i), []byte(seed))
	}
	f.Add(uint8(0), []byte(`{"protocolVersion":1e0,"_meta":{"large":18446744073709551616}}`))
	f.Add(uint8(0), []byte(`{"protocolVersion":1,"_meta":{"exponent":1e1000000}}`))
	f.Fuzz(func(t *testing.T, index uint8, data []byte) {
		definition := constructors[int(index)%len(constructors)]
		value := definition.new()
		if err := json.Unmarshal(data, value); err != nil {
			return
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatalf("%s decoded but could not encode: %v", definition.name, err)
		}
		if err := protocol.ValidateJSON(definition.name, encoded); err != nil {
			t.Fatalf("%s encoded invalid wire value: %v", definition.name, err)
		}
		if err := json.Unmarshal(encoded, definition.new()); err != nil {
			t.Fatalf("%s could not decode its own encoding: %v", definition.name, err)
		}
	})
}
