package acp_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/acp"
)

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func receive[T any](t *testing.T, ctx context.Context, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-ctx.Done():
		t.Fatal("timed out waiting for test event")
		var zero T
		return zero
	}
}

type peer struct {
	in   *io.PipeWriter
	out  *io.PipeReader
	scan *bufio.Scanner
}

func rawConn(t *testing.T, ctx context.Context, h acp.Handlers, limits acp.Limits) (*acp.Conn, *peer) {
	t.Helper()
	input, send := io.Pipe()
	recv, output := io.Pipe()
	c, err := acp.New(ctx, input, output, h, limits)
	if err != nil {
		t.Fatal(err)
	}
	p := &peer{send, recv, bufio.NewScanner(recv)}
	t.Cleanup(func() { _ = c.Close(); _ = send.Close(); _ = recv.Close() })
	return c, p
}

func (p *peer) send(t *testing.T, ctx context.Context, frame string) {
	t.Helper()
	done := make(chan error, 1)
	go func() { _, err := io.WriteString(p.in, frame); done <- err }()
	if err := receive(t, ctx, done); err != nil {
		t.Fatal(err)
	}
}

func (p *peer) read(t *testing.T, ctx context.Context) map[string]json.RawMessage {
	t.Helper()
	type result struct {
		frame []byte
		err   error
	}
	done := make(chan result, 1)
	go func() {
		if !p.scan.Scan() {
			done <- result{err: p.scan.Err()}
			return
		}
		done <- result{frame: append([]byte(nil), p.scan.Bytes()...)}
	}()
	r := receive(t, ctx, done)
	if r.err != nil {
		t.Fatal(r.err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(r.frame, &m); err != nil {
		t.Fatalf("invalid response %q: %v", r.frame, err)
	}
	return m
}

func wantCause(t *testing.T, ctx context.Context, c *acp.Conn, want error) {
	t.Helper()
	if err := c.Wait(ctx); !errors.Is(err, want) {
		t.Fatalf("Wait = %v, want %v", err, want)
	}
	if !errors.Is(c.Err(), want) {
		t.Fatalf("Err = %v, want %v", c.Err(), want)
	}
}

func TestBidirectionalPromptAndPermission(t *testing.T) {
	ctx := testContext(t)
	ar, bw := io.Pipe()
	br, aw := io.Pipe()
	var notifications atomic.Int32
	client, err := acp.New(ctx, ar, aw, acp.Handlers{
		Request: func(_ context.Context, _ *acp.Conn, method string, _ json.RawMessage) (any, error) {
			if method != "permission" {
				return nil, errors.New("unexpected method")
			}
			return true, nil
		},
		Notification: func(ctx context.Context, c *acp.Conn, _ string, _ json.RawMessage) error {
			if err := c.Call(ctx, "forbidden", nil, nil); !errors.Is(err, acp.ErrNotificationCall) {
				return errors.New("notification Call was not rejected")
			}
			notifications.Add(1)
			return nil
		},
	}, acp.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	server, err := acp.New(ctx, br, bw, acp.Handlers{Request: func(ctx context.Context, c *acp.Conn, _ string, _ json.RawMessage) (any, error) {
		var allowed bool
		if err := c.Call(ctx, "permission", nil, &allowed); err != nil {
			return nil, err
		}
		if !allowed {
			return nil, errors.New("denied")
		}
		for i := 0; i < 3; i++ {
			if err := c.Notify(ctx, "update", []int{i}); err != nil {
				return nil, err
			}
		}
		return "complete", nil
	}}, acp.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Close() })
	var result string
	if err := client.Call(ctx, "prompt", nil, &result); err != nil {
		t.Fatal(err)
	}
	if result != "complete" || notifications.Load() != 3 {
		t.Fatalf("result=%q, notifications=%d", result, notifications.Load())
	}
}

func TestRequestIDsAndHandlerErrors(t *testing.T) {
	for _, id := range []string{`9223372036854775807`, `"quote\"slash\\newline\n"`, `null`} {
		t.Run(id, func(t *testing.T) {
			ctx := testContext(t)
			_, p := rawConn(t, ctx, acp.Handlers{Request: func(context.Context, *acp.Conn, string, json.RawMessage) (any, error) { return true, nil }}, acp.Limits{})
			p.send(t, ctx, `{"jsonrpc":"2.0","id":`+id+`,"method":"echo"}`+"\n")
			m := p.read(t, ctx)
			if string(m["id"]) != id {
				t.Fatalf("id=%s, want %s", m["id"], id)
			}
			if string(m["result"]) != "true" {
				t.Fatalf("response: %v", m)
			}
		})
	}
	for _, typed := range []bool{false, true} {
		name := "redacted"
		if typed {
			name = "typed"
		}
		t.Run(name, func(t *testing.T) {
			ctx := testContext(t)
			_, p := rawConn(t, ctx, acp.Handlers{Request: func(context.Context, *acp.Conn, string, json.RawMessage) (any, error) {
				if typed {
					return nil, &acp.Error{Code: -32042, Message: "permission denied"}
				}
				return nil, errors.New("secret-token-must-not-leak")
			}}, acp.Limits{})
			p.send(t, ctx, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"fail\"}\n")
			m := p.read(t, ctx)
			var e acp.Error
			if err := json.Unmarshal(m["error"], &e); err != nil {
				t.Fatal(err)
			}
			if typed {
				if e.Code != -32042 || e.Message != "permission denied" {
					t.Fatalf("typed error: %+v", e)
				}
			} else if e.Code != -32603 || strings.Contains(string(m["error"]), "secret-token") {
				t.Fatalf("unredacted error: %s", m["error"])
			}
		})
	}
}

func TestInvalidFrames(t *testing.T) {
	cases := []struct {
		name, frame string
		limit       int
		eof         bool
		want        error
	}{
		{"syntax", "{\n", 0, false, acp.ErrInvalidMessage},
		{"batch", "[]\n", 0, false, acp.ErrInvalidMessage},
		{"utf8", "{\"jsonrpc\":\"2.0\",\"method\":\"\xff\"}\n", 0, false, acp.ErrInvalidMessage},
		{"version", "{\"jsonrpc\":\"1.0\",\"method\":\"x\"}\n", 0, false, acp.ErrInvalidMessage},
		{"fractional-id", "{\"jsonrpc\":\"2.0\",\"id\":1.5,\"method\":\"x\"}\n", 0, false, acp.ErrInvalidMessage},
		{"both-response-fields", "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":null,\"error\":{\"code\":-1,\"message\":\"x\"}}\n", 0, false, acp.ErrInvalidMessage},
		{"missing-response-fields", "{\"jsonrpc\":\"2.0\",\"id\":1}\n", 0, false, acp.ErrInvalidMessage},
		{"truncated", "{\"jsonrpc\":\"2.0\",\"method\":\"x\"}", 0, true, acp.ErrInvalidMessage},
		{"oversize", strings.Repeat("x", 65) + "\n", 64, false, acp.ErrFrameTooLarge},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := testContext(t)
			c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{FrameBytes: tc.limit})
			// A terminating parser may close the pipe before the full write completes.
			go func() {
				_, _ = io.WriteString(p.in, tc.frame)
				if tc.eof {
					_ = p.in.Close()
				}
			}()
			wantCause(t, ctx, c, tc.want)
		})
	}
}

func TestBlockedOutputCancellation(t *testing.T) {
	for _, call := range []bool{false, true} {
		name := "notify"
		if call {
			name = "call"
		}
		t.Run(name, func(t *testing.T) {
			ctx := testContext(t)
			c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{})
			opctx, cancel := context.WithCancel(ctx)
			done := make(chan error, 1)
			go func() {
				if call {
					done <- c.Call(opctx, "blocked", nil, nil)
				} else {
					done <- c.Notify(opctx, "blocked", nil)
				}
			}()
			writing := make(chan error, 1)
			go func() { var first [1]byte; _, err := io.ReadFull(p.out, first[:]); writing <- err }()
			if err := receive(t, ctx, writing); err != nil {
				t.Fatal(err)
			}
			cancel()
			if err := receive(t, ctx, done); !errors.Is(err, context.Canceled) {
				t.Fatalf("operation = %v", err)
			}
			receive(t, ctx, c.Done())
		})
	}
}

func TestShutdownJoinsHandler(t *testing.T) {
	for _, mode := range []string{"close", "parent", "eof"} {
		t.Run(mode, func(t *testing.T) {
			ctx := testContext(t)
			parent, cancel := context.WithCancel(ctx)
			defer cancel()
			started, canceled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			defer close(release)
			c, p := rawConn(t, parent, acp.Handlers{Request: func(ctx context.Context, _ *acp.Conn, _ string, _ json.RawMessage) (any, error) {
				close(started)
				<-ctx.Done()
				close(canceled)
				<-release
				return nil, ctx.Err()
			}}, acp.Limits{})
			p.send(t, ctx, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"wait\"}\n")
			receive(t, ctx, started)
			want := acp.ErrClosed
			switch mode {
			case "close":
				_ = c.Close()
				_ = c.Close()
			case "parent":
				cancel()
				want = context.Canceled
			case "eof":
				_ = p.in.Close()
				want = io.EOF
			}
			receive(t, ctx, c.Done())
			receive(t, ctx, canceled)
			expired, stop := context.WithCancel(ctx)
			stop()
			if err := c.Wait(expired); !errors.Is(err, context.Canceled) {
				t.Fatalf("Wait before handler exit = %v", err)
			}
			// Release only after proving Wait does not treat Done as a join.
			// Use a separate gate relay so the deferred close remains the sole closer.
			joined := make(chan error, 1)
			go func() { joined <- c.Wait(ctx) }()
			// Cleanup cannot release until this function returns; finish joining in cleanup.
			t.Cleanup(func() {
				if err := receive(t, ctx, joined); !errors.Is(err, want) {
					t.Errorf("joined cause = %v, want %v", err, want)
				}
			})
		})
	}
}

func TestRequestAdmission(t *testing.T) {
	for _, duplicate := range []bool{false, true} {
		name := "limit"
		if duplicate {
			name = "duplicate"
		}
		t.Run(name, func(t *testing.T) {
			ctx := testContext(t)
			started := make(chan struct{})
			c, p := rawConn(t, ctx, acp.Handlers{Request: func(ctx context.Context, _ *acp.Conn, _ string, _ json.RawMessage) (any, error) {
				close(started)
				<-ctx.Done()
				return nil, ctx.Err()
			}}, acp.Limits{Requests: 1})
			p.send(t, ctx, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"hold\"}\n")
			receive(t, ctx, started)
			id := "2"
			want := acp.ErrLimit
			if duplicate {
				id = "1"
				want = acp.ErrInvalidMessage
			}
			p.send(t, ctx, `{"jsonrpc":"2.0","id":`+id+`,"method":"hold"}`+"\n")
			wantCause(t, ctx, c, want)
		})
	}
}

func TestPendingCallCapAndLateResponse(t *testing.T) {
	ctx := testContext(t)
	c, p := rawConn(t, ctx, acp.Handlers{}, acp.Limits{PendingCalls: 1})
	callctx, cancel := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- c.Call(callctx, "first", nil, nil) }()
	first := p.read(t, ctx)
	if err := c.Call(ctx, "overflow", nil, nil); !errors.Is(err, acp.ErrLimit) {
		t.Fatalf("overflow = %v", err)
	}
	select {
	case <-c.Done():
		t.Fatal("pending cap closed connection")
	default:
	}
	cancel()
	if err := receive(t, ctx, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(first["id"])+`,"result":null}`+"\n")
	go func() { done <- c.Call(ctx, "next", nil, nil) }()
	next := p.read(t, ctx)
	p.send(t, ctx, `{"jsonrpc":"2.0","id":`+string(next["id"])+`,"result":null}`+"\n")
	if err := receive(t, ctx, done); err != nil {
		t.Fatal(err)
	}
}

func TestNotificationOrderAndOverflow(t *testing.T) {
	t.Run("ordered", func(t *testing.T) {
		ctx := testContext(t)
		seen := make(chan int, 3)
		_, p := rawConn(t, ctx, acp.Handlers{Notification: func(_ context.Context, _ *acp.Conn, _ string, params json.RawMessage) error {
			var values []int
			if err := json.Unmarshal(params, &values); err != nil {
				return err
			}
			seen <- values[0]
			return nil
		}}, acp.Limits{})
		for _, n := range []string{"1", "2", "3"} {
			p.send(t, ctx, `{"jsonrpc":"2.0","method":"update","params":[`+n+`]}`+"\n")
		}
		for i := 1; i <= 3; i++ {
			if n := receive(t, ctx, seen); n != i {
				t.Fatalf("notification %d = %d", i, n)
			}
		}
	})
	t.Run("overflow", func(t *testing.T) {
		ctx := testContext(t)
		started := make(chan struct{})
		c, p := rawConn(t, ctx, acp.Handlers{Notification: func(ctx context.Context, _ *acp.Conn, _ string, _ json.RawMessage) error {
			close(started)
			<-ctx.Done()
			return ctx.Err()
		}}, acp.Limits{Notifications: 1})
		frame := "{\"jsonrpc\":\"2.0\",\"method\":\"update\"}\n"
		p.send(t, ctx, frame)
		receive(t, ctx, started)
		p.send(t, ctx, frame)
		p.send(t, ctx, frame)
		wantCause(t, ctx, c, acp.ErrLimit)
	})
}

// heldWriter makes the first frame visible before its Write can return.
type heldWriter struct {
	output  *io.PipeWriter
	started chan struct{}
	release chan struct{}
	closed  chan struct{}
	first   atomic.Bool
	once    sync.Once
}

func (w *heldWriter) Write(p []byte) (int, error) {
	first := w.first.CompareAndSwap(false, true)
	if first {
		close(w.started)
	}
	n, err := w.output.Write(p)
	if first && err == nil {
		select {
		case <-w.release:
		case <-w.closed:
			return n, io.ErrClosedPipe
		}
	}
	return n, err
}

func (w *heldWriter) Close() error {
	w.once.Do(func() { close(w.closed) })
	return w.output.Close()
}

func heldConn(t *testing.T, ctx context.Context, h acp.Handlers, limits acp.Limits) (*acp.Conn, *peer, *heldWriter) {
	t.Helper()
	input, send := io.Pipe()
	recv, output := io.Pipe()
	w := &heldWriter{output: output, started: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{})}
	c, err := acp.New(ctx, input, w, h, limits)
	if err != nil {
		_ = input.Close()
		_ = send.Close()
		_ = recv.Close()
		_ = w.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = c.Close()
		_ = send.Close()
		_ = recv.Close()
		join, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := c.Wait(join); errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("connection workers did not join: %v", err)
		}
	})
	return c, &peer{send, recv, bufio.NewScanner(recv)}, w
}

func TestRequestIDReusableBeforeResponseWriteReturns(t *testing.T) {
	ctx := testContext(t)
	entered := make(chan int32, 2)
	var calls atomic.Int32
	c, p, w := heldConn(t, ctx, acp.Handlers{Request: func(context.Context, *acp.Conn, string, json.RawMessage) (any, error) {
		n := calls.Add(1)
		entered <- n
		return n, nil
	}}, acp.Limits{Requests: 1})
	frame := "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"echo\"}\n"
	p.send(t, ctx, frame)
	if n := receive(t, ctx, entered); n != 1 {
		t.Fatalf("first handler = %d", n)
	}
	first := p.read(t, ctx)
	if string(first["result"]) != "1" {
		t.Fatalf("first response: %v", first)
	}
	// The peer has consumed the response, but the writer remains gated.
	p.send(t, ctx, frame)
	select {
	case n := <-entered:
		if n != 2 {
			t.Fatalf("second handler = %d", n)
		}
	case <-c.Done():
		t.Fatalf("reusing completed ID closed connection: %v", c.Err())
	case <-ctx.Done():
		t.Fatal("reused ID was not admitted while Write remained blocked")
	}
	close(w.release)
	second := p.read(t, ctx)
	if string(second["id"]) != "7" || string(second["result"]) != "2" {
		t.Fatalf("second response: %v", second)
	}
	_ = c.Close()
	wantCause(t, ctx, c, acp.ErrClosed)
}

func TestWriteQueueOverflowJoinsWorkers(t *testing.T) {
	ctx := testContext(t)
	c, _, w := heldConn(t, ctx, acp.Handlers{}, acp.Limits{Writes: 1})
	done := make(chan error, 3)
	start := func() { go func() { done <- c.Notify(ctx, "update", nil) }() }
	start()
	receive(t, ctx, w.started)
	// With the writer blocked, two more sends cannot both fit in one slot.
	start()
	start()
	t.Cleanup(func() {
		_ = c.Close()
		join, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		for i := 0; i < 3; i++ {
			if err := receive(t, join, done); err == nil {
				t.Error("blocked notification unexpectedly succeeded")
			}
		}
	})
	wantCause(t, ctx, c, acp.ErrLimit)
}

type trackedStream struct{ closed bool }

func (*trackedStream) Read([]byte) (int, error)    { return 0, io.EOF }
func (*trackedStream) Write(p []byte) (int, error) { return len(p), nil }
func (s *trackedStream) Close() error              { s.closed = true; return nil }

func TestInvalidLimitsRetainOwnership(t *testing.T) {
	for _, limits := range []acp.Limits{{FrameBytes: -1}, {Requests: -1}, {PendingCalls: -1}, {Notifications: -1}, {Writes: -1}} {
		in, out := new(trackedStream), new(trackedStream)
		c, err := acp.New(testContext(t), in, out, acp.Handlers{}, limits)
		if err == nil {
			if c != nil {
				_ = c.Close()
			}
			t.Fatal("negative limit accepted")
		}
		if in.closed || out.closed {
			t.Fatal("failed New took stream ownership")
		}
	}
}
