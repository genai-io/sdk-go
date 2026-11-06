package acp_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func TestElicitationModeCapabilities(t *testing.T) {
	for _, tc := range []struct {
		caps, mode string
		allowed    bool
	}{
		{`{}`, "url", false},
		{`{"elicitation":{"form":{}}}`, "url", false},
		{`{"elicitation":{"url":{}}}`, "form", false},
		{`{"elicitation":{"form":null}}`, "form", false},
		{`{"elicitation":{"url":{}}}`, "url", true},
	} {
		t.Run(tc.caps+tc.mode, func(t *testing.T) {
			ctx := testContext(t)
			h := agentHandlers()
			done := make(chan error, 1)
			req := formRequest(t)
			if tc.mode == "url" {
				req.Value = &protocol.CreateElicitationRequestURLVariant{Mode: "url", Message: "Authorize", ElicitationID: "flow", URL: "https://example.com/auth"}
			}
			h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
				_, err := turn.CreateElicitation(req)
				done <- err
				return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
			}
			_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
			p.send(t, ctx, agentRequest("init", "initialize", `{"protocolVersion":1,"clientCapabilities":`+tc.caps+`}`))
			equalJSON(t, p.read(t, ctx)["result"], `{"protocolVersion":1}`)
			p.send(t, ctx, agentRequest("new", "session/new", newParams))
			p.read(t, ctx)
			p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
			if tc.allowed {
				frame := p.read(t, ctx)
				var request protocol.CreateElicitationRequest
				if err := json.Unmarshal(frame["params"], &request); err != nil {
					t.Fatal(err)
				}
				if request.Value.(protocol.CreateElicitationRequestURLVariant).SessionID.Value != "s" {
					t.Fatal("URL scope not bound")
				}
				p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"result":{"action":"cancel"}}`+"\n")
			}
			equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
			err := receive(t, ctx, done)
			if tc.allowed && err != nil {
				t.Fatal(err)
			}
			if !tc.allowed && !errors.Is(err, acp.ErrElicitationUnsupported) {
				t.Fatal(err)
			}
		})
	}
}
