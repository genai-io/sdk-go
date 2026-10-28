package acp_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func TestAgentInitializationAdmission(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	release := make(chan struct{})
	h.Initialize = func(ctx context.Context, _ protocol.InitializeRequest) (protocol.InitializeResponse, error) {
		select {
		case <-release:
		case <-ctx.Done():
			return protocol.InitializeResponse{}, ctx.Err()
		}
		return protocol.InitializeResponse{ProtocolVersion: 1}, nil
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{})
	p.send(t, ctx, agentRequest("init", "initialize", initParams)+agentRequest("duplicate", "initialize", initParams)+agentRequest("early", "session/new", newParams))
	for range 2 {
		frame := p.read(t, ctx)
		switch string(frame["id"]) {
		case `"duplicate"`:
			assertRPCError(t, frame, "duplicate", -32600)
		case `"early"`:
			assertRPCError(t, frame, "early", -32600)
		default:
			t.Fatalf("unexpected response: %v", frame)
		}
	}
	close(release)
	equalJSON(t, p.read(t, ctx)["result"], `{"protocolVersion":1}`)
}

func TestAgentSessionReservationsAndFailures(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	release := make(chan struct{})
	started := make(chan struct{})
	var calls atomic.Int32
	h.CreateSession = func(ctx context.Context, _ protocol.NewSessionRequest) (protocol.NewSessionResponse, error) {
		switch calls.Add(1) {
		case 1:
			close(started)
			select {
			case <-release:
			case <-ctx.Done():
				return protocol.NewSessionResponse{}, ctx.Err()
			}
			return protocol.NewSessionResponse{}, errors.New("failed creation")
		case 2:
			panic("application panic")
		case 3:
			return protocol.NewSessionResponse{SessionID: "s"}, nil
		case 4:
			return protocol.NewSessionResponse{SessionID: "s"}, nil
		default:
			return protocol.NewSessionResponse{SessionID: "other"}, nil
		}
	}
	_, p := rawAgent(t, ctx, h, acp.AgentOptions{MaxSessions: 2})
	p.send(t, ctx, agentRequest("init", "initialize", initParams))
	p.read(t, ctx)
	p.send(t, ctx, agentRequest("blocked", "session/new", newParams))
	receive(t, ctx, started)
	// Second creation panics; its reservation is freed at error-response commit.
	p.send(t, ctx, agentRequest("panic", "session/new", newParams))
	assertRPCError(t, p.read(t, ctx), "panic", -32603)
	p.send(t, ctx, agentRequest("ok", "session/new", newParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"sessionId":"s"}`)
	p.send(t, ctx, agentRequest("full", "session/new", newParams))
	assertRPCError(t, p.read(t, ctx), "full", -32001)
	close(release)
	assertRPCError(t, p.read(t, ctx), "blocked", -32603)
	p.send(t, ctx, agentRequest("duplicate", "session/new", newParams))
	assertRPCError(t, p.read(t, ctx), "duplicate", -32603)
	p.send(t, ctx, agentRequest("other", "session/new", newParams))
	equalJSON(t, p.read(t, ctx)["result"], `{"sessionId":"other"}`)
}

func TestAgentFinishJoinsAdmittedUpdate(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	started := make(chan struct{})
	returned := make(chan *acp.Turn, 1)
	updateDone := make(chan error, 1)
	h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		var update protocol.SessionNotification
		_ = json.Unmarshal([]byte(`{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"final"}}}`), &update)
		go func() { updateDone <- turn.SendSessionUpdate(update) }()
		// The write has entered the transport, so the operation is admitted even
		// though this deliberately adversarial callback fails to join its goroutine.
		<-started
		returned <- turn
		return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
	}
	_, p := rawAgentOutput(t, ctx, h, acp.AgentOptions{}, func(w io.WriteCloser) io.WriteCloser { return &thirdWrite{WriteCloser: w, started: started} })
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	receive(t, ctx, returned)
	p.send(t, ctx, agentRequest("overlap", "session/prompt", promptParams))
	// A separate pipe write can be consumed only after dispatching overlap.
	p.send(t, ctx, "{\"jsonrpc\":\"2.0\",\"method\":\"test/barrier\"}\n")
	equalJSON(t, p.read(t, ctx)["method"], `"session/update"`)
	for range 2 {
		frame := p.read(t, ctx)
		if string(frame["id"]) == `"overlap"` {
			assertRPCError(t, frame, "overlap", -32001)
		} else {
			equalJSON(t, frame["id"], `"prompt"`)
			equalJSON(t, frame["result"], `{"stopReason":"end_turn"}`)
		}
	}
	if err := receive(t, ctx, updateDone); err != nil {
		t.Fatal(err)
	}
}

func TestAgentShutdownUnblocksPermissionWrite(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	started := make(chan struct{})
	ended := make(chan struct{})
	h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		defer close(ended)
		_, err := turn.RequestPermission(permissionRequest())
		return protocol.PromptResponse{}, err
	}
	a, p := rawAgentOutput(t, ctx, h, acp.AgentOptions{}, func(w io.WriteCloser) io.WriteCloser { return &thirdWrite{WriteCloser: w, started: started} })
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	receive(t, ctx, started)
	_ = a.Close()
	if err := a.Wait(ctx); !errors.Is(err, acp.ErrClosed) {
		t.Fatal(err)
	}
	select {
	case <-ended:
	default:
		t.Fatal("Wait returned before callback unwind")
	}
}

func TestAgentFinishCancelsPermissionWait(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	started := make(chan struct{})
	permissionDone := make(chan error, 1)
	h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		go func() {
			_, err := turn.RequestPermission(permissionRequest())
			permissionDone <- err
		}()
		<-started
		// Returning must cancel the outstanding response wait, but join the write.
		return protocol.PromptResponse{StopReason: protocol.StopReasonEndTurn}, nil
	}
	a, p := rawAgentOutput(t, ctx, h, acp.AgentOptions{}, func(w io.WriteCloser) io.WriteCloser { return &thirdWrite{WriteCloser: w, started: started} })
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	equalJSON(t, p.read(t, ctx)["method"], `"session/request_permission"`)
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"end_turn"}`)
	if err := receive(t, ctx, permissionDone); !errors.Is(err, context.Canceled) {
		t.Fatalf("finish permission: %v", err)
	}
	if a.Err() != nil {
		t.Fatal(a.Err())
	}
}
