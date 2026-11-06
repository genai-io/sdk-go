package acp_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

const formParams = `{"mode":"form","sessionId":"s","message":"Choose","requestedSchema":{"type":"object","properties":{"answer":{"type":"string","enum":["a","b"]}},"required":["answer"]}}`

func formRequest(t *testing.T) protocol.CreateElicitationRequest {
	t.Helper()
	var req protocol.CreateElicitationRequest
	if err := json.Unmarshal([]byte(formParams), &req); err != nil {
		t.Fatal(err)
	}
	return req
}
func formReady(t *testing.T, ctx context.Context, p *peer) {
	t.Helper()
	p.send(t, ctx, agentRequest("init", "initialize", `{"protocolVersion":1,"clientCapabilities":{"elicitation":{"form":{}}}}`))
	equalJSON(t, p.read(t, ctx)["result"], `{"protocolVersion":1}`)
	p.send(t, ctx, agentRequest("new", "session/new", newParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"sessionId":"s"}`)
}
func TestElicitationCalls(t *testing.T) {
	for _, result := range []string{`{"action":"accept","content":{"answer":"a"}}`, `{"action":"accept","content":{"integer":42,"number":1.5,"boolean":true,"array":["a","b"]}}`, `{"action":"decline"}`, `{"action":"cancel"}`} {
		t.Run(result, func(t *testing.T) {
			testProtocolCall(t, "elicitation/create", formParams, result, false, (*acp.Conn).CreateElicitation)
		})
	}
	t.Run("malformed", func(t *testing.T) {
		testProtocolCall(t, "elicitation/create", formParams, `{"content":{}}`, true, (*acp.Conn).CreateElicitation)
	})
	t.Run("request scope", func(t *testing.T) {
		testProtocolCall(t, "elicitation/create", `{"mode":"form","requestId":42,"message":"Input","requestedSchema":{}}`, `{"action":"cancel"}`, false, (*acp.Conn).CreateElicitation)
	})
	t.Run("url", func(t *testing.T) {
		testProtocolCall(t, "elicitation/create", `{"mode":"url","sessionId":"s","elicitationId":"flow","url":"https://example.com/auth","message":"Authorize"}`, `{"action":"accept"}`, false, (*acp.Conn).CreateElicitation)
	})
}
func TestElicitationNegotiationAndScope(t *testing.T) {
	for _, supported := range []bool{false, true} {
		t.Run(map[bool]string{false: "absent", true: "form"}[supported], func(t *testing.T) {
			ctx := testContext(t)
			h := agentHandlers()
			done := make(chan error, 1)
			retained := make(chan *acp.Turn, 1)
			req := formRequest(t)
			v := req.Value.(protocol.CreateElicitationRequestFormVariant)
			v.SessionID = protocol.Some(protocol.SessionID("wrong"))
			req.Value = v
			h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
				retained <- turn
				_, err := turn.CreateElicitation(req)
				done <- err
				return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
			}
			_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
			if supported {
				formReady(t, ctx, p)
			} else {
				agentReady(t, ctx, p)
			}
			p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
			if supported {
				frame := p.read(t, ctx)
				equalJSON(t, frame["params"], formParams)
				// Unsolicited response IDs cannot satisfy this request.
				p.send(t, ctx, `{"jsonrpc":"2.0","id":"not-offered","result":{"action":"accept","content":{"answer":"b"}}}`+"\n")
				p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"result":{"action":"cancel"}}`+"\n")
			}
			equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
			err := receive(t, ctx, done)
			if supported && err != nil {
				t.Fatal(err)
			}
			if !supported && !errors.Is(err, acp.ErrElicitationUnsupported) {
				t.Fatal(err)
			}
			if _, err := (<-retained).CreateElicitation(req); !errors.Is(err, acp.ErrTurnFinished) {
				t.Fatal(err)
			}
		})
	}
}
func TestElicitationCancelAndFinish(t *testing.T) {
	for _, finish := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "finish"}[finish], func(t *testing.T) {
			ctx := testContext(t)
			h := agentHandlers()
			started := make(chan struct{})
			done := make(chan error, 1)
			req := formRequest(t)
			h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
				if finish {
					go func() { _, err := turn.CreateElicitation(req); done <- err }()
					<-started
				} else {
					_, err := turn.CreateElicitation(req)
					done <- err
				}
				return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
			}
			_, p := rawAgentOutput(t, ctx, h, acp.AgentOptions{}, func(w io.WriteCloser) io.WriteCloser { return &thirdWrite{WriteCloser: w, started: started} })
			formReady(t, ctx, p)
			p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
			frame := p.read(t, ctx)
			equalJSON(t, frame["method"], `"elicitation/create"`)
			if !finish {
				p.send(t, ctx, cancelFrame)
			}
			result := `{"stopReason":"cancelled"}`
			if finish {
				result = `{"stopReason":"end_turn"}`
			}
			equalJSON(t, p.read(t, ctx)["result"], result)
			if err := receive(t, ctx, done); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"result":{"action":"accept","content":{"answer":"a"}}}`+"\n")
		})
	}
}
func TestElicitationHandler(t *testing.T) {
	calls := 0
	handler := acp.ElicitationHandler(func(context.Context, protocol.CreateElicitationRequest) (protocol.CreateElicitationResponse, error) {
		calls++
		return protocol.CreateElicitationResponse{Value: protocol.CreateElicitationResponseCancelVariant{Action: "cancel"}}, nil
	})
	for _, params := range []string{`{}`, `{"mode":"form","message":"missing scope","requestedSchema":{}}`} {
		if _, err := handler(context.Background(), nil, "elicitation/create", json.RawMessage(params)); err == nil {
			t.Fatal("accepted malformed request")
		}
	}
	if _, err := handler(context.Background(), nil, "elicitation/create", json.RawMessage(formParams)); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatal(calls)
	}
}
