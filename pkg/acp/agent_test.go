package acp_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func agentHandlers() acp.AgentHandlers {
	return acp.AgentHandlers{
		Initialize: func(context.Context, protocol.InitializeRequest) (protocol.InitializeResponse, error) {
			return protocol.InitializeResponse{ProtocolVersion: 1}, nil
		},
		CreateSession: func(context.Context, protocol.NewSessionRequest) (acp.NewSessionResult, error) {
			return acp.NewSessionResult{NewSessionResponse: protocol.NewSessionResponse{SessionID: "s"}}, nil
		},
		Prompt: func(context.Context, *acp.Turn, protocol.PromptRequest) (protocol.PromptResponse, error) {
			return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
		},
	}
}

func rawAgent(t *testing.T, ctx context.Context, h acp.AgentHandlers, options acp.AgentOptions) (*acp.Agent, *peer) {
	return rawAgentOutput(t, ctx, h, options, func(w io.WriteCloser) io.WriteCloser { return w })
}

func rawAgentOutput(t *testing.T, ctx context.Context, h acp.AgentHandlers, options acp.AgentOptions, wrap func(io.WriteCloser) io.WriteCloser) (*acp.Agent, *peer) {
	t.Helper()
	input, send := io.Pipe()
	recv, output := io.Pipe()
	a, err := acp.NewAgent(ctx, input, wrap(output), h, options)
	if err != nil {
		t.Fatal(err)
	}
	p := &peer{send, recv, bufio.NewScanner(recv)}
	t.Cleanup(func() {
		_ = a.Close()
		_ = send.Close()
		_ = recv.Close()
		if err := a.Wait(ctx); !errors.Is(err, acp.ErrClosed) {
			t.Errorf("agent cleanup: %v", err)
		}
	})
	return a, p
}

func agentRequest(id, method, params string) string {
	return `{"jsonrpc":"2.0","id":"` + id + `","method":"` + method + `","params":` + params + "}\n"
}

const initParams = `{"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":true}}}`
const newParams = `{"cwd":"/test","mcpServers":[]}`
const promptParams = `{"sessionId":"s","prompt":[]}`
const cancelFrame = "{\"jsonrpc\":\"2.0\",\"method\":\"session/cancel\",\"params\":{\"sessionId\":\"s\"}}\n"

func agentReady(t *testing.T, ctx context.Context, p *peer) {
	t.Helper()
	p.send(t, ctx, agentRequest("init", "initialize", initParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"protocolVersion":1}`)
	p.send(t, ctx, agentRequest("new", "session/new", newParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"sessionId":"s"}`)
}

func assertRPCError(t *testing.T, m map[string]json.RawMessage, id string, code int) {
	t.Helper()
	equalJSON(t, m["id"], `"`+id+`"`)
	var e acp.Error
	if err := json.Unmarshal(m["error"], &e); err != nil || e.Code != int64(code) {
		t.Fatalf("error = %s, want %d", m["error"], code)
	}
}

func TestAgentInitialCommands(t *testing.T) {
	for _, mode := range []string{"commands", "empty", "omitted", "prompt-error", "prompt-cancelled"} {
		t.Run(mode, func(t *testing.T) {
			ctx := testContext(t)
			h := agentHandlers()
			commands := []protocol.AvailableCommand{{Name: "check", Description: "Check project"}}
			if mode == "empty" {
				commands = []protocol.AvailableCommand{}
			} else if mode == "omitted" {
				commands = nil
			}
			h.CreateSession = func(context.Context, protocol.NewSessionRequest) (acp.NewSessionResult, error) {
				return acp.NewSessionResult{NewSessionResponse: protocol.NewSessionResponse{SessionID: "s"}, InitialCommands: commands}, nil
			}
			h.Prompt = func(ctx context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
				if mode == "prompt-error" {
					return protocol.PromptResponse{}, errors.New("provider failed")
				}
				if mode == "prompt-cancelled" {
					<-ctx.Done()
					return protocol.PromptResponse{}, ctx.Err()
				}
				return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, turn.SendUpdate(finalUpdate())
			}
			_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
			agentReady(t, ctx, p)
			prompt := agentRequest("prompt", "session/prompt", promptParams)
			if mode == "prompt-cancelled" {
				prompt += cancelFrame
			}
			p.send(t, ctx, prompt)
			if commands != nil {
				frame := p.read(t, ctx)
				equalJSON(t, frame["method"], `"session/update"`)
				var update protocol.SessionNotification
				if err := json.Unmarshal(frame["params"], &update); err != nil {
					t.Fatal(err)
				}
				catalog, ok := update.Update.Value.(protocol.SessionUpdateAvailableCommandsUpdateVariant)
				if !ok || update.SessionID != "s" || len(catalog.AvailableCommands) != len(commands) {
					t.Fatalf("initial catalog: %s", frame["params"])
				}
				if len(commands) != 0 && catalog.AvailableCommands[0].Name != "check" {
					t.Fatal("lost initial command")
				}
			}
			switch mode {
			case "prompt-error":
				assertRPCError(t, p.read(t, ctx), "prompt", -32603)
			case "prompt-cancelled":
				equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"cancelled"}`)
			default:
				equalJSON(t, p.read(t, ctx)["method"], `"session/update"`)
				equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
			}
		})
	}
}

func TestAgentInvalidInitialCommandsReleaseSlot(t *testing.T) {
	for _, oversized := range []bool{false, true} {
		t.Run(map[bool]string{false: "encoding", true: "frame-limit"}[oversized], func(t *testing.T) {
			ctx := testContext(t)
			h := agentHandlers()
			calls := 0
			h.CreateSession = func(context.Context, protocol.NewSessionRequest) (acp.NewSessionResult, error) {
				calls++
				result := acp.NewSessionResult{NewSessionResponse: protocol.NewSessionResponse{SessionID: "s"}}
				if calls == 1 {
					c := protocol.AvailableCommand{Name: "check", Description: "Check project"}
					if oversized {
						c.Description = strings.Repeat("x", 2048)
					} else {
						c.Extra = map[string]json.RawMessage{"invalid": json.RawMessage(`{`)}
					}
					result.InitialCommands = []protocol.AvailableCommand{c}
				}
				return result, nil
			}
			_, p := rawAgent(t, ctx, h, acp.AgentOptions{MaxSessions: 1, Limits: acp.Limits{FrameBytes: 1024}})
			p.send(t, ctx, agentRequest("init", "initialize", initParams))
			p.read(t, ctx)
			p.send(t, ctx, agentRequest("bad", "session/new", newParams))
			assertRPCError(t, p.read(t, ctx), "bad", -32603)
			p.send(t, ctx, agentRequest("retry", "session/new", newParams))
			equalJSON(t, p.read(t, ctx)["result"], `{"sessionId":"s"}`)
		})
	}
}

func TestAgentInitializationAndSessions(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	var initCalls atomic.Int32
	h.Initialize = func(context.Context, protocol.InitializeRequest) (protocol.InitializeResponse, error) {
		if initCalls.Add(1) == 1 {
			return protocol.InitializeResponse{}, errors.New("private failure")
		}
		return protocol.InitializeResponse{ProtocolVersion: 1}, nil
	}
	a, p := rawAgent(t, ctx, h, acp.AgentOptions{MaxSessions: 1})
	for _, tc := range []struct {
		id, method, params string
		code               int
	}{
		{"pre", "session/new", newParams, -32600},
		{"invalid", "initialize", `{}`, -32602},
		{"version", "initialize", `{"protocolVersion":2}`, -32602},
		{"failure", "initialize", initParams, -32603},
	} {
		p.send(t, ctx, agentRequest(tc.id, tc.method, tc.params))
		assertRPCError(t, p.read(t, ctx), tc.id, tc.code)
	}
	if _, _, ok := a.Initialization(); ok {
		t.Fatal("initialized on failure")
	}
	agentReady(t, ctx, p)
	req, res, ok := a.Initialization()
	if !ok || req.ProtocolVersion != 1 || res.ProtocolVersion != 1 {
		t.Fatal("missing initialization state")
	}
	req.Extra = map[string]json.RawMessage{"x": json.RawMessage(`true`)}
	req2, _, _ := a.Initialization()
	if len(req2.Extra) != 0 {
		t.Fatal("snapshot aliases state")
	}
	for _, tc := range []struct {
		id, method, params string
		code               int
	}{
		{"dup", "initialize", initParams, -32600},
		{"limit", "session/new", newParams, -32001},
		{"invalid", "session/prompt", `{"sessionId":"s"}`, -32602},
		{"unknown", "session/prompt", `{"sessionId":"missing","prompt":[]}`, -32002},
		{"optional", "session/load", `{}`, -32601},
	} {
		p.send(t, ctx, agentRequest(tc.id, tc.method, tc.params))
		assertRPCError(t, p.read(t, ctx), tc.id, tc.code)
	}
	if initCalls.Load() != 2 {
		t.Fatal("invalid or duplicate initialization called application")
	}
}

func TestAgentWireOrderCancellationAndSuccessor(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	entered := make(chan *acp.Turn, 2)
	cancelled := make(chan struct{}, 1)
	unwind := make(chan struct{})
	var calls atomic.Int32
	h.Prompt = func(ctx context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		entered <- turn
		if calls.Add(1) == 1 {
			<-ctx.Done()
			cancelled <- struct{}{}
			<-unwind
			var update protocol.SessionNotification
			_ = json.Unmarshal([]byte(`{"sessionId":"wrong","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"final"}}}`), &update)
			if err := turn.SendUpdate(update); err != nil {
				return protocol.PromptResponse{}, err
			}
			return protocol.PromptResponse{}, ctx.Err()
		}
		if ctx.Err() != nil {
			return protocol.PromptResponse{}, errors.New("successor cancelled")
		}
		return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
	agentReady(t, ctx, p)
	// A single write tests reader order independent of callback scheduling.
	p.send(t, ctx, agentRequest("first", "session/prompt", promptParams)+cancelFrame+agentRequest("overlap", "session/prompt", promptParams))
	old := receive(t, ctx, entered)
	receive(t, ctx, cancelled)
	// No prompt response can precede unwind; the only available response is the rejection.
	assertRPCError(t, p.read(t, ctx), "overlap", -32001)
	p.send(t, ctx, cancelFrame) // still targets the first turn
	close(unwind)
	update := p.read(t, ctx)
	equalJSON(t, update["method"], `"session/update"`)
	var params protocol.SessionNotification
	if err := json.Unmarshal(update["params"], &params); err != nil || params.SessionID != "s" {
		t.Fatalf("update = %s", update["params"])
	}
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"cancelled"}`)
	// Idle cancellation must not become cancellation of the next prompt.
	p.send(t, ctx, cancelFrame+agentRequest("first", "session/prompt", promptParams))
	receive(t, ctx, entered)
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
	if err := old.SendUpdate(protocol.SessionNotification{}); !errors.Is(err, acp.ErrTurnFinished) {
		t.Fatalf("late update: %v", err)
	}
	if _, err := old.RequestPermission(protocol.RequestPermissionRequest{}); !errors.Is(err, acp.ErrTurnFinished) {
		t.Fatalf("late permission: %v", err)
	}
}

func permissionRequest() protocol.RequestPermissionRequest {
	var req protocol.RequestPermissionRequest
	_ = json.Unmarshal([]byte(`{"sessionId":"wrong","toolCall":{"toolCallId":"tool"},"options":[{"optionId":"allow","name":"Allow","kind":"allow_once"}]}`), &req)
	return req
}

type thirdWrite struct {
	io.WriteCloser
	count   int
	started chan struct{}
}

func (w *thirdWrite) Write(b []byte) (int, error) {
	w.count++
	if w.count == 3 {
		close(w.started)
	}
	return w.WriteCloser.Write(b)
}

func TestAgentPermissionCancellationDuringWrite(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	entered := make(chan struct{}, 1)
	result := make(chan error, 1)
	h.Prompt = func(ctx context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		entered <- struct{}{}
		_, err := turn.RequestPermission(permissionRequest())
		result <- err
		return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, err
	}
	started := make(chan struct{})
	a, p := rawAgentOutput(t, ctx, h, acp.AgentOptions{}, func(w io.WriteCloser) io.WriteCloser { return &thirdWrite{WriteCloser: w, started: started} })
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	receive(t, ctx, entered)
	receive(t, ctx, started)
	// Do not read output: permission's pipe write cannot finish. Cancel must not
	// poison framing, even if its write is active rather than queued.
	p.send(t, ctx, cancelFrame)
	frame := p.read(t, ctx)
	equalJSON(t, frame["method"], `"session/request_permission"`)
	var req protocol.RequestPermissionRequest
	if err := json.Unmarshal(frame["params"], &req); err != nil || req.SessionID != "s" {
		t.Fatal("permission not bound to session")
	}
	if err := receive(t, ctx, result); !errors.Is(err, context.Canceled) {
		t.Fatalf("permission cancellation: %v", err)
	}
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"cancelled"}`)
	// A stale reply must not complete a later call or close the connection.
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"result":{"outcome":{"outcome":"selected","optionId":"allow"}}}`+"\n")
	if a.Err() != nil {
		t.Fatal(a.Err())
	}
	p.send(t, ctx, agentRequest("unknown", "test/alive", `{}`))
	assertRPCError(t, p.read(t, ctx), "unknown", -32601)
}

func TestAgentPermissionResponseAndShutdown(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	got := make(chan protocol.RequestPermissionResponse, 1)
	ended := make(chan struct{})
	h.Prompt = func(ctx context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		defer close(ended)
		res, err := turn.RequestPermission(permissionRequest())
		if err != nil {
			return protocol.PromptResponse{}, err
		}
		got <- res
		<-ctx.Done()
		return protocol.PromptResponse{}, ctx.Err()
	}
	a, p := rawAgent(t, ctx, h, acp.AgentOptions{})
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	frame := p.read(t, ctx)
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"result":{"outcome":{"outcome":"selected","optionId":"allow"}}}`+"\n")
	res := receive(t, ctx, got)
	if _, ok := res.Outcome.Value.(protocol.RequestPermissionOutcomeSelectedVariant); !ok {
		t.Fatalf("outcome = %#v", res)
	}
	_ = a.Close()
	if err := a.Wait(ctx); !errors.Is(err, acp.ErrClosed) {
		t.Fatal(err)
	}
	select {
	case <-ended:
	default:
		t.Fatal("Wait did not join prompt")
	}
}
