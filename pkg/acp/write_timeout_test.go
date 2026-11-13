package acp_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

func timeoutAgent(t *testing.T, h acp.AgentHandlers, budget time.Duration, started chan struct{}) (*acp.Agent, *peer) {
	t.Helper()
	ctx := testContext(t)
	input, send := io.Pipe()
	recv, output := io.Pipe()
	var w io.WriteCloser = output
	if started != nil {
		w = &thirdWrite{WriteCloser: output, started: started}
	}
	a, err := acp.NewAgent(ctx, input, w, h, acp.AgentOptions{Limits: acp.Limits{WriteTimeout: budget}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = a.Close()
		_ = send.Close()
		_ = recv.Close()
		if err := a.Wait(ctx); !errors.Is(err, acp.ErrClosed) && !errors.Is(err, acp.ErrWriteTimeout) {
			t.Errorf("join: %v", err)
		}
	})
	return a, &peer{send, recv, bufio.NewScanner(recv)}
}

func finalUpdate() protocol.SessionNotification {
	var u protocol.SessionNotification
	_ = json.Unmarshal([]byte(`{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"final"}}}`), &u)
	return u
}

func TestInitialCommandsWriteTimeout(t *testing.T) {
	ctx := testContext(t)
	h := agentHandlers()
	h.CreateSession = func(context.Context, protocol.NewSessionRequest) (acp.NewSessionResult, error) {
		return acp.NewSessionResult{NewSessionResponse: protocol.NewSessionResponse{SessionID: "s"},
			InitialCommands: []protocol.AvailableCommand{{Name: "check", Description: "Check project"}}}, nil
	}
	started := make(chan struct{})
	a, p := timeoutAgent(t, h, 100*time.Millisecond, started)
	agentReady(t, ctx, p)
	receive(t, ctx, started)
	p.send(t, ctx, cancelFrame)
	if err := a.Wait(ctx); !errors.Is(err, acp.ErrWriteTimeout) {
		t.Fatalf("initial catalog write: %v", err)
	}
}

func TestAgentWriteTimeoutCancellation(t *testing.T) {
	for _, permission := range []bool{false, true} {
		for _, partial := range []bool{false, true} {
			name := "update"
			if permission {
				name = "permission"
			}
			if partial {
				name += "/partial"
			}
			t.Run(name, func(t *testing.T) {
				ctx := testContext(t)
				started, ended := make(chan struct{}), make(chan struct{})
				result := make(chan error, 1)
				h := agentHandlers()
				h.Prompt = func(_ context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
					defer close(ended)
					var err error
					if permission {
						_, err = turn.RequestPermission(permissionRequest())
					} else {
						err = turn.SendUpdate(finalUpdate())
					}
					result <- err
					return protocol.PromptResponse{}, err
				}
				a, p := timeoutAgent(t, h, 100*time.Millisecond, started)
				agentReady(t, ctx, p)
				p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
				receive(t, ctx, started)
				if partial {
					// io.Pipe.Write remains blocked after exposing a prefix. Close interrupts
					// that same Write; no goroutine is left holding a torn JSON frame.
					prefix := make([]byte, 5)
					if _, err := io.ReadFull(p.out, prefix); err != nil {
						t.Fatal(err)
					}
					if string(prefix) != `{"jso` {
						t.Fatalf("prefix %q", prefix)
					}
				}
				p.send(t, ctx, cancelFrame)
				if err := a.Wait(ctx); !errors.Is(err, acp.ErrWriteTimeout) {
					t.Fatalf("Wait: %v", err)
				}
				if err := receive(t, ctx, result); err == nil {
					t.Fatal("blocked frame reported success")
				}
				select {
				case <-ended:
				default:
					t.Fatal("Wait did not join callback")
				}
				if _, err := p.out.Read(make([]byte, 1)); err == nil {
					t.Fatal("connection reusable after partial frame")
				}
			})
		}
	}
}

func TestAgentWriteTimeoutHealthyCancel(t *testing.T) {
	ctx := testContext(t)
	started := make(chan struct{})
	h := agentHandlers()
	h.Prompt = func(ctx context.Context, turn *acp.Turn, _ protocol.PromptRequest) (protocol.PromptResponse, error) {
		err := turn.SendUpdate(finalUpdate())
		if err != nil {
			return protocol.PromptResponse{}, err
		}
		<-ctx.Done()
		return protocol.PromptResponse{}, ctx.Err()
	}
	a, p := timeoutAgent(t, h, time.Second, started)
	agentReady(t, ctx, p)
	p.send(t, ctx, agentRequest("prompt", "session/prompt", promptParams))
	receive(t, ctx, started)
	p.send(t, ctx, cancelFrame)
	equalJSON(t, p.read(t, ctx)["method"], `"session/update"`)
	equalJSON(t, p.read(t, ctx)["result"], `{"stopReason":"cancelled"}`)
	if a.Err() != nil {
		t.Fatal(a.Err())
	}
}

func TestWriteTimeoutCompletionRace(t *testing.T) {
	for range 30 {
		ctx := testContext(t)
		c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{WriteTimeout: time.Millisecond})
		done := make(chan error, 1)
		go func() { done <- c.Notify(ctx, "test", nil) }()
		time.Sleep(time.Millisecond)
		// Completion at the deadline may win or fail closed, never hang.
		_, _ = p.out.Read(make([]byte, 1024))
		err := receive(t, ctx, done)
		if err != nil && !errors.Is(err, acp.ErrWriteTimeout) {
			t.Fatal(err)
		}
		_ = c.Close()
		if err := c.Wait(ctx); !errors.Is(err, acp.ErrClosed) && !errors.Is(err, acp.ErrWriteTimeout) {
			t.Fatal(err)
		}
	}
}

func TestWriteTimeoutRejectsNegative(t *testing.T) {
	input, send := io.Pipe()
	recv, output := io.Pipe()
	t.Cleanup(func() {
		// These unused pipes only need resource cleanup.
		_ = input.Close()
		_ = send.Close()
		_ = recv.Close()
		_ = output.Close()
	})
	if _, err := acp.New(context.Background(), input, output, acp.Handlers{}, acp.Limits{WriteTimeout: -1}); !errors.Is(err, acp.ErrLimit) {
		t.Fatal(err)
	}
}
