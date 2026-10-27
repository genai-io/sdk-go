package integration_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
)

type acpPeer struct {
	input  *os.File
	output *bufio.Reader
}

func acpPipe(t *testing.T, handlers acp.Handlers) (context.Context, *acp.Conn, acpPeer) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	conn, peer := acpPipeContext(t, ctx, handlers)
	return ctx, conn, peer
}

func acpPipeContext(t *testing.T, ctx context.Context, handlers acp.Handlers) (*acp.Conn, acpPeer) {
	t.Helper()
	in, clientInput, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	clientOutput, out, err := os.Pipe()
	if err != nil {
		_ = in.Close()
		_ = clientInput.Close()
		t.Fatal(err)
	}
	conn, err := acp.New(ctx, in, out, handlers, acp.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	stop := context.AfterFunc(ctx, func() { _ = clientInput.Close(); _ = clientOutput.Close() })
	t.Cleanup(func() {
		stop()
		_ = conn.Close()
		_ = clientInput.Close()
		_ = clientOutput.Close()
		if err := conn.Wait(ctx); err != nil && !errors.Is(err, acp.ErrClosed) && !errors.Is(err, context.Canceled) {
			t.Errorf("ACP cleanup: %v", err)
		}
	})
	return conn, acpPeer{clientInput, bufio.NewReader(clientOutput)}
}

func (p acpPeer) send(t *testing.T, message jsonrpc.Message) {
	t.Helper()
	data, err := jsonrpc.EncodeMessage(message)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.input.Write(append(data, '\n')); err != nil {
		t.Fatal(err)
	}
}

func (p acpPeer) read(t *testing.T) jsonrpc.Message {
	t.Helper()
	frame, err := p.output.ReadBytes('\n')
	if err != nil {
		t.Fatalf("missing ACP frame: %v", err)
	}
	message, err := jsonrpc.DecodeMessage(frame)
	if err != nil {
		t.Fatal(err)
	}
	return message
}

func acpID(t *testing.T, value string) jsonrpc.ID {
	t.Helper()
	id, err := jsonrpc.MakeID(value)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestACPWireConversation(t *testing.T) {
	_, _, peer := acpPipe(t, acp.Handlers{Request: func(ctx context.Context, c *acp.Conn, method string, _ json.RawMessage) (any, error) {
		switch method {
		case "initialize":
			return map[string]any{"protocolVersion": 1, "agentCapabilities": map[string]any{}, "authMethods": []any{}}, nil
		case "session/new":
			return map[string]any{"sessionId": "fixture"}, nil
		case "session/prompt":
			var answer struct {
				Outcome struct {
					OptionID string `json:"optionId"`
				} `json:"outcome"`
			}
			if err := c.Call(ctx, "session/request_permission", map[string]any{
				"sessionId": "fixture", "toolCall": map[string]any{"toolCallId": "tool-1"},
				"options": []any{map[string]any{"optionId": "allow", "name": "Allow once", "kind": "allow_once"}},
			}, &answer); err != nil {
				return nil, err
			}
			if answer.Outcome.OptionID != "allow" {
				return nil, &acp.Error{Code: -32001, Message: "Permission denied"}
			}
			if err := c.Notify(ctx, "session/update", map[string]any{
				"sessionId": "fixture",
				"update":    map[string]any{"sessionUpdate": "agent_message_chunk", "content": map[string]any{"type": "text", "text": "Hello\n世界"}},
			}); err != nil {
				return nil, err
			}
			return map[string]any{"stopReason": "end_turn"}, nil
		default:
			return nil, &acp.Error{Code: -32601, Message: "Method not found"}
		}
	}})
	for _, step := range []struct{ method, params, result string }{
		{"initialize", `{"protocolVersion":1}`, `"protocolVersion":1`},
		{"session/new", `{"cwd":"/fixture","mcpServers":[]}`, `"sessionId":"fixture"`},
	} {
		peer.send(t, &jsonrpc.Request{ID: acpID(t, step.method), Method: step.method, Params: json.RawMessage(step.params)})
		response, ok := peer.read(t).(*jsonrpc.Response)
		if !ok || response.Error != nil || !strings.Contains(string(response.Result), step.result) {
			t.Fatalf("%s response: %+v", step.method, response)
		}
	}
	peer.send(t, &jsonrpc.Request{ID: acpID(t, "prompt"), Method: "session/prompt", Params: json.RawMessage(`{"sessionId":"fixture","prompt":[{"type":"text","text":"hello"}]}`)})
	permission, ok := peer.read(t).(*jsonrpc.Request)
	if !ok || permission.Method != "session/request_permission" || !permission.IsCall() {
		t.Fatalf("permission request: %+v", permission)
	}
	peer.send(t, &jsonrpc.Response{ID: permission.ID, Result: json.RawMessage(`{"outcome":{"outcome":"selected","optionId":"allow"}}`)})
	update, ok := peer.read(t).(*jsonrpc.Request)
	if !ok || update.IsCall() || update.Method != "session/update" {
		t.Fatalf("update: %+v", update)
	}
	response, ok := peer.read(t).(*jsonrpc.Response)
	if !ok || response.ID.Raw() != "prompt" || string(response.Result) != `{"stopReason":"end_turn"}` {
		t.Fatalf("prompt result: %+v", response)
	}
}

func TestACPCancelDuringPermission(t *testing.T) {
	var mu sync.Mutex
	var stop context.CancelFunc
	_, _, peer := acpPipe(t, acp.Handlers{
		Request: func(ctx context.Context, c *acp.Conn, _ string, _ json.RawMessage) (any, error) {
			turn, cancel := context.WithCancel(ctx)
			defer cancel()
			mu.Lock()
			stop = cancel
			mu.Unlock()
			if err := c.Call(turn, "session/request_permission", map[string]any{"sessionId": "fixture"}, nil); !errors.Is(err, context.Canceled) {
				return nil, errors.New("permission was not cancelled")
			}
			return map[string]any{"stopReason": "cancelled"}, nil
		},
		Notification: func(_ context.Context, _ *acp.Conn, method string, _ json.RawMessage) error {
			if method == "session/cancel" {
				mu.Lock()
				cancel := stop
				mu.Unlock()
				if cancel != nil {
					cancel()
				}
			}
			return nil
		},
	})
	peer.send(t, &jsonrpc.Request{ID: acpID(t, "prompt"), Method: "session/prompt", Params: json.RawMessage(`{"sessionId":"fixture","prompt":[]}`)})
	permission := peer.read(t).(*jsonrpc.Request)
	peer.send(t, &jsonrpc.Request{Method: "session/cancel", Params: json.RawMessage(`{"sessionId":"fixture"}`)})
	response, ok := peer.read(t).(*jsonrpc.Response)
	if !ok || response.Error != nil || string(response.Result) != `{"stopReason":"cancelled"}` {
		t.Fatalf("cancel result: %+v", response)
	}
	peer.send(t, &jsonrpc.Response{ID: permission.ID, Result: json.RawMessage(`{"outcome":{"outcome":"cancelled"}}`)})
}

func TestACPBlockedOSPipe(t *testing.T) {
	ctx, conn, peer := acpPipe(t, acp.Handlers{})
	writeCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- conn.Notify(writeCtx, "session/update", map[string]any{"text": strings.Repeat("x", 512<<10)})
	}()
	var first [1]byte
	if _, err := io.ReadFull(peer.output, first[:]); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("blocked write: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("blocked OS pipe prevented cancellation")
	}
	if err := conn.Wait(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("blocked write cleanup: %v", err)
	}
}
