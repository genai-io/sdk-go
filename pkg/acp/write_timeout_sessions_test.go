package acp_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func TestWriteTimeoutSlowSessionsDrainInOrder(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	next := 0 // Creations below are sequential request/response exchanges.
	h.CreateSession = func(context.Context, protocol.NewSessionRequest) (protocol.NewSessionResponse, error) {
		id := protocol.SessionID(fmt.Sprintf("s%d", next))
		next++
		return protocol.NewSessionResponse{SessionID: id}, nil
	}
	h.Prompt = func(ctx context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		if err := turn.SendUpdate(finalUpdate()); err != nil {
			return protocol.PromptResponse{}, err
		}
		return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, ctx.Err()
	}
	// The entire drain takes longer than the per-write budget. Waiting behind
	// other sessions must not start a frame's timer or drop its promised update.
	a, p := timeoutAgent(t, h, 150*time.Millisecond, nil)
	p.send(t, ctx, agentRequest("init", "initialize", initParams))
	p.read(t, ctx)
	for i := range 8 {
		p.send(t, ctx, agentRequest(fmt.Sprintf("new%d", i), "session/new", newParams))
		p.read(t, ctx)
	}
	for i := range 8 {
		p.send(t, ctx, agentRequest(fmt.Sprintf("s%d", i), "session/prompt", fmt.Sprintf(`{"sessionId":"s%d","prompt":[]}`, i)))
	}
	p.send(t, ctx, "{\"jsonrpc\":\"2.0\",\"method\":\"session/cancel\",\"params\":{\"sessionId\":\"s0\"}}\n")
	updates := make(map[string]bool)
	responses := make(map[string]bool)
	for range 16 {
		time.Sleep(20 * time.Millisecond)
		frame := p.read(t, ctx)
		if string(frame["method"]) == `"session/update"` {
			var params struct {
				SessionID string `json:"sessionId"`
			}
			if err := json.Unmarshal(frame["params"], &params); err != nil {
				t.Fatal(err)
			}
			if updates[params.SessionID] {
				t.Fatal("duplicate update")
			}
			updates[params.SessionID] = true
		} else {
			var id string
			if err := json.Unmarshal(frame["id"], &id); err != nil {
				t.Fatal(err)
			}
			if !updates[id] || responses[id] {
				t.Fatalf("response before update or duplicated: %s", id)
			}
			responses[id] = true
			reason := "end_turn"
			if id == "s0" {
				reason = "cancelled"
			}
			equalJSON(t, frame["result"], fmt.Sprintf(`{"stopReason":%q}`, reason))
		}
	}
	if len(responses) != 8 || a.Err() != nil {
		t.Fatalf("responses %d, connection %v", len(responses), a.Err())
	}
}
