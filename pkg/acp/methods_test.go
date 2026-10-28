package acp_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func equalJSON(t *testing.T, got []byte, want string) {
	t.Helper()
	var left, right any
	if err := json.Unmarshal(got, &left); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &right); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(left, right) {
		t.Fatalf("JSON = %s, want %s", got, want)
	}
}

func testProtocolCall[Request, Response any](t *testing.T, method, requestJSON, responseJSON string, wantInvalid bool, call func(*acp.Conn, context.Context, Request) (Response, error)) {
	t.Helper()
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	var request Request
	if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
		t.Fatal(err)
	}
	type result struct {
		response Response
		err      error
	}
	done := make(chan result, 1)
	go func() {
		response, err := call(c, ctx, request)
		done <- result{response, err}
	}()
	frame := p.read(t, ctx)
	equalJSON(t, frame["method"], `"`+method+`"`)
	equalJSON(t, frame["params"], requestJSON)
	if len(frame["id"]) == 0 {
		t.Fatal("typed call sent a notification")
	}
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"result":`+responseJSON+"}\n")
	got := receive(t, ctx, done)
	if wantInvalid {
		if got.err == nil {
			t.Fatal("accepted invalid typed response")
		}
		if c.Err() != nil {
			t.Fatalf("typed response validation closed connection: %v", c.Err())
		}
		return
	}
	if got.err != nil {
		t.Fatal(got.err)
	}
	encoded, err := json.Marshal(got.response)
	if err != nil {
		t.Fatal(err)
	}
	equalJSON(t, encoded, responseJSON)
}

func TestProtocolCalls(t *testing.T) {
	t.Run("initialize", func(t *testing.T) {
		testProtocolCall(t, "initialize", `{"protocolVersion":1}`, `{"protocolVersion":1,"agentCapabilities":{},"authMethods":[]}`, false, (*acp.Conn).Initialize)
	})
	t.Run("integral numeric spelling", func(t *testing.T) {
		testProtocolCall(t, "initialize", `{"protocolVersion":1.0}`, `{"protocolVersion":1e0}`, false, (*acp.Conn).Initialize)
	})
	t.Run("new", func(t *testing.T) {
		testProtocolCall(t, "session/new", `{"cwd":"/fixture","mcpServers":[]}`, `{"sessionId":"fixture"}`, false, (*acp.Conn).CreateSession)
	})
	t.Run("prompt", func(t *testing.T) {
		testProtocolCall(t, "session/prompt", `{"sessionId":"fixture","prompt":[{"type":"text","text":"hello"}]}`, `{"stopReason":"end_turn"}`, false, (*acp.Conn).Prompt)
	})
	t.Run("permission", func(t *testing.T) {
		testProtocolCall(t, "session/request_permission", `{"sessionId":"fixture","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"allow","name":"Allow once","kind":"allow_once"}]}`, `{"outcome":{"outcome":"selected","optionId":"allow"}}`, false, (*acp.Conn).RequestPermission)
	})
}

func TestProtocolResponseValidation(t *testing.T) {
	t.Run("version range", func(t *testing.T) {
		testProtocolCall(t, "initialize", `{"protocolVersion":1}`, `{"protocolVersion":65536,"agentCapabilities":{},"authMethods":[]}`, true, (*acp.Conn).Initialize)
	})
	t.Run("missing session ID", func(t *testing.T) {
		testProtocolCall(t, "session/new", `{"cwd":"/fixture","mcpServers":[]}`, `{}`, true, (*acp.Conn).CreateSession)
	})
	t.Run("unknown stop reason", func(t *testing.T) {
		testProtocolCall(t, "session/prompt", `{"sessionId":"fixture","prompt":[]}`, `{"stopReason":"invented"}`, true, (*acp.Conn).Prompt)
	})
	t.Run("missing selected option", func(t *testing.T) {
		testProtocolCall(t, "session/request_permission", `{"sessionId":"fixture","toolCall":{"toolCallId":"tool-1"},"options":[]}`, `{"outcome":{"outcome":"selected"}}`, true, (*acp.Conn).RequestPermission)
	})
}

func TestProtocolCancelledContext(t *testing.T) {
	ctx := testContext(t)
	c, _ := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := c.CreateSession(cancelled, protocol.NewSessionRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("CreateSession = %v, want context.Canceled", err)
	}
	if err := c.CancelPrompt(cancelled, protocol.CancelNotification{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("CancelPrompt = %v, want context.Canceled", err)
	}
	if c.Err() != nil {
		t.Fatal(c.Err())
	}
}

func TestProtocolRequestValidation(t *testing.T) {
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	if _, err := c.CreateSession(ctx, protocol.NewSessionRequest{}); err == nil {
		t.Fatal("nil required mcpServers array accepted")
	}
	if _, err := c.Prompt(ctx, protocol.PromptRequest{}); err == nil {
		t.Fatal("nil required prompt array accepted")
	}
	if _, err := c.RequestPermission(ctx, protocol.RequestPermissionRequest{}); err == nil {
		t.Fatal("nil required options array accepted")
	}
	if c.Err() != nil {
		t.Fatalf("local validation closed connection: %v", c.Err())
	}
	done := make(chan error, 1)
	go func() { done <- c.Notify(ctx, "test/still-open", nil) }()
	frame := p.read(t, ctx)
	equalJSON(t, frame["method"], `"test/still-open"`)
	if err := receive(t, ctx, done); err != nil {
		t.Fatal(err)
	}
}

func testProtocolNotification[Notification any](t *testing.T, method, params string, notify func(*acp.Conn, context.Context, Notification) error) {
	t.Helper()
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	var notification Notification
	if err := json.Unmarshal([]byte(params), &notification); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- notify(c, ctx, notification) }()
	frame := p.read(t, ctx)
	if _, present := frame["id"]; present {
		t.Fatal("typed notification sent as a request")
	}
	equalJSON(t, frame["method"], `"`+method+`"`)
	equalJSON(t, frame["params"], params)
	if err := receive(t, ctx, done); err != nil {
		t.Fatal(err)
	}
}

func TestProtocolNotifications(t *testing.T) {
	t.Run("cancel", func(t *testing.T) {
		testProtocolNotification(t, "session/cancel", `{"sessionId":"fixture"}`, (*acp.Conn).CancelPrompt)
	})
	t.Run("update", func(t *testing.T) {
		testProtocolNotification(t, "session/update", `{"sessionId":"fixture","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Hello\n世界"}}}`, (*acp.Conn).SendSessionUpdate)
	})
}

func TestProtocolPromptLocalCancellation(t *testing.T) {
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	var prompt protocol.PromptRequest
	if err := json.Unmarshal([]byte(`{"sessionId":"fixture","prompt":[]}`), &prompt); err != nil {
		t.Fatal(err)
	}
	turn, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := c.Prompt(turn, prompt)
		done <- err
	}()
	request := p.read(t, ctx)

	p.flushWrites(t, ctx, c)
	cancel()
	if err := receive(t, ctx, done); !errors.Is(err, context.Canceled) {
		t.Fatalf("Prompt = %v, want context.Canceled", err)
	}
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(request["id"])+`,"result":{"stopReason":"end_turn"}}`+"\n")

	var initialize protocol.InitializeRequest
	if err := json.Unmarshal([]byte(`{"protocolVersion":1}`), &initialize); err != nil {
		t.Fatal(err)
	}
	go func() {
		_, err := c.Initialize(ctx, initialize)
		done <- err
	}()
	next := p.read(t, ctx)
	// Any automatically emitted session/cancel would appear here instead.
	equalJSON(t, next["method"], `"initialize"`)
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(next["id"])+`,"result":{"protocolVersion":1}}`+"\n")
	if err := receive(t, ctx, done); err != nil {
		t.Fatal(err)
	}
	if c.Err() != nil {
		t.Fatal(c.Err())
	}
}

func TestProtocolCancelPromptKeepsWait(t *testing.T) {
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	var prompt protocol.PromptRequest
	if err := json.Unmarshal([]byte(`{"sessionId":"fixture","prompt":[]}`), &prompt); err != nil {
		t.Fatal(err)
	}
	var cancel protocol.CancelNotification
	if err := json.Unmarshal([]byte(`{"sessionId":"fixture"}`), &cancel); err != nil {
		t.Fatal(err)
	}
	type result struct {
		response protocol.PromptResponse
		err      error
	}
	promptDone := make(chan result, 1)
	go func() {
		response, err := c.Prompt(ctx, prompt)
		promptDone <- result{response, err}
	}()
	request := p.read(t, ctx)
	cancelDone := make(chan error, 1)
	go func() { cancelDone <- c.CancelPrompt(ctx, cancel) }()
	notification := p.read(t, ctx)
	equalJSON(t, notification["method"], `"session/cancel"`)
	if _, present := notification["id"]; present {
		t.Fatal("cancel must be a notification")
	}
	if err := receive(t, ctx, cancelDone); err != nil {
		t.Fatal(err)
	}
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(request["id"])+`,"result":{"stopReason":"cancelled"}}`+"\n")
	got := receive(t, ctx, promptDone)
	if got.err != nil {
		t.Fatal(got.err)
	}
	wire, err := json.Marshal(got.response)
	if err != nil {
		t.Fatal(err)
	}
	equalJSON(t, wire, `{"stopReason":"cancelled"}`)
}

func TestProtocolRPCError(t *testing.T) {
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
	var request protocol.InitializeRequest
	if err := json.Unmarshal([]byte(`{"protocolVersion":1}`), &request); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := c.Initialize(ctx, request)
		done <- err
	}()
	frame := p.read(t, ctx)
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(frame["id"])+`,"error":{"code":-32000,"message":"Authentication required"}}`+"\n")
	var failure *acp.Error
	if err := receive(t, ctx, done); !errors.As(err, &failure) || failure.Code != -32000 {
		t.Fatalf("typed method lost JSON-RPC error: %v", err)
	}
}
