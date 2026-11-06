package acp_test

import (
	"context"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func TestAgentLoadReservationAndRetry(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	entered := make(chan struct{}, 1)
	h.LoadSession = func(ctx context.Context, turn *acp.Turn, req protocol.LoadSessionRequest) (protocol.LoadSessionResponse, error) {
		entered <- struct{}{}
		<-ctx.Done()
		return protocol.LoadSessionResponse{}, ctx.Err()
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{MaxSessions: 1})
	p.send(t, ctx, agentRequest("init", "initialize", initParams))
	p.read(t, ctx)
	params := `{"sessionId":"s","cwd":"/test","mcpServers":[]}`
	p.send(t, ctx, agentRequest("load", "session/load", params))
	<-entered
	p.send(t, ctx, agentRequest("duplicate", "session/load", params))
	assertRPCError(t, p.read(t, ctx), "duplicate", -32001)
	p.send(t, ctx, agentRequest("early", "session/prompt", promptParams))
	assertRPCError(t, p.read(t, ctx), "early", -32002)
	p.send(t, ctx, cancelFrame)
	p.read(t, ctx)
	// Failed loads release both identity and slot at the response barrier.
	p.send(t, ctx, agentRequest("new", "session/new", newParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"sessionId":"s"}`)
}

func TestAgentModeBoundaryDoesNotCancelChange(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	entered, release := make(chan context.Context, 1), make(chan struct{})
	h.SetMode = func(ctx context.Context, req protocol.SetSessionModeRequest) (protocol.SetSessionModeResponse, error) {
		entered <- ctx
		<-release
		return protocol.SetSessionModeResponse{}, ctx.Err()
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
	agentReady(t, ctx, p)
	params := `{"sessionId":"s","modeId":"normal"}`
	p.send(t, ctx, agentRequest("mode", "session/set_mode", params))
	changeCtx := <-entered
	p.send(t, ctx, cancelFrame+agentRequest("prompt", "session/prompt", promptParams))
	assertRPCError(t, p.read(t, ctx), "prompt", -32001)
	if changeCtx.Err() != nil {
		t.Fatal("session/cancel cancelled configuration")
	}
	p.send(t, ctx, agentRequest("second", "session/set_mode", params))
	assertRPCError(t, p.read(t, ctx), "second", -32001)
	close(release)
	equalJSON(t, p.read(t, ctx)["result"], `{}`)
	p.send(t, ctx, agentRequest("prompt2", "session/prompt", promptParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
}

func TestAgentSetConfigWireMethod(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	h.SetConfig = func(_ context.Context, req protocol.SetSessionConfigOptionRequest) (protocol.SetSessionConfigOptionResponse, error) {
		v, ok := req.Value.(protocol.SetSessionConfigOptionRequestValueIDVariant)
		if !ok || v.SessionID != "s" || v.ConfigID != "model" || v.Value != "other" {
			t.Errorf("unexpected configuration request: %#v", req)
		}
		return protocol.SetSessionConfigOptionResponse{ConfigOptions: []protocol.SessionConfigOption{}}, nil
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("config", "session/set_config_option", `{"sessionId":"s","configId":"model","value":"other"}`))
	equalJSON(t, p.read(t, ctx)["result"], `{"configOptions":[]}`)
}

func TestAgentLoadReplaysBeforeAdmission(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	h.LoadSession = func(ctx context.Context, turn *acp.Turn, req protocol.LoadSessionRequest) (protocol.LoadSessionResponse, error) {
		err := turn.SendUpdate(protocol.SessionNotification{SessionID: "wrong", Update: protocol.SessionUpdate{Value: protocol.SessionUpdateUserMessageChunkVariant{SessionUpdate: "user_message_chunk", Content: protocol.ContentBlock{Value: protocol.ContentBlockTextVariant{Type: "text", Text: "history"}}}}})
		return protocol.LoadSessionResponse{}, err
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
	p.send(t, ctx, agentRequest("init", "initialize", initParams))
	p.read(t, ctx)
	p.send(t, ctx, agentRequest("load", "session/load", `{"sessionId":"s","cwd":"/test","mcpServers":[]}`))
	update := p.read(t, ctx)
	equalJSON(t, update["method"], `"session/update"`)
	equalJSON(t, update["params"], `{"sessionId":"s","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"history"}}}`)
	equalJSON(t, p.read(t, ctx)["result"], `{}`)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
}
