package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)

type progressTool struct {
	t        *testing.T
	finished *atomic.Bool
	runs     *atomic.Int32
}

func (p progressTool) Schema() ai.Schema {
	return ai.Schema{Name: "Write", Definition: json.RawMessage(`{"type":"object","properties":{"content":{"type":"string"}}}`)}
}
func (p progressTool) Run(_ context.Context, call ai.ToolCall) (agent.Result, error) {
	if !p.finished.Load() || call.Input != `{"content":"`+strings.Repeat("x", 12)+`"}` {
		p.t.Errorf("tool ran before successful inference or with incomplete input: %q", call.Input)
	}
	p.runs.Add(1)
	return agent.Result{Terminate: true}, nil
}

// Exercise the real HTTP decoder, driver, client and watchdog together. The
// stream lasts longer than idle; only actual argument bytes keep it alive.
func TestResponsesToolGenerationWatchdog(t *testing.T) {
	for _, mode := range []string{"progress", "stall", "empty", "metadata", "failed", "first-budget", "idle-budget", "cancel", "break"} {
		t.Run(mode, func(t *testing.T) {
			var finished atomic.Bool
			var runs atomic.Int32
			closed := make(chan struct{})
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(closed)
				w.Header().Set("Content-Type", "text/event-stream")
				send := func(v any) {
					b, _ := json.Marshal(v)
					_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
					w.(http.Flusher).Flush()
				}
				wait := func(d time.Duration) bool {
					select {
					case <-r.Context().Done():
						return false
					case <-time.After(d):
						return true
					}
				}
				if mode == "first-budget" {
					if !wait(200 * time.Millisecond) {
						return
					}
				}
				send(map[string]any{"type": "response.output_item.added", "item": map[string]any{"type": "function_call", "id": "fc", "call_id": "call", "name": "Write"}})
				send(map[string]any{"type": "response.function_call_arguments.delta", "item_id": "fc", "delta": `{"content":"`})
				if mode == "stall" || mode == "cancel" || mode == "break" {
					<-r.Context().Done()
					return
				}
				if mode == "idle-budget" {
					if !wait(300 * time.Millisecond) {
						return
					}
				}
				for range 12 {
					if !wait(30 * time.Millisecond) {
						return
					}
					switch mode {
					case "empty":
						send(map[string]any{"type": "response.function_call_arguments.delta", "item_id": "fc", "delta": ""})
					case "metadata":
						send(map[string]any{"type": "response.output_item.added", "item": map[string]any{"type": "function_call", "id": "fc", "call_id": "call", "name": "Write"}})
					default:
						send(map[string]any{"type": "response.function_call_arguments.delta", "item_id": "fc", "delta": "x"})
					}
				}
				send(map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "function_call", "id": "fc", "call_id": "call", "name": "Write", "arguments": `{"content":"` + strings.Repeat("x", 12) + `"}`}})
				if mode == "failed" {
					send(map[string]any{"type": "response.failed", "response": map[string]any{"status": "failed", "error": map[string]any{"code": "server_error", "message": "failed"}}})
					return
				}
				finished.Store(true)
				send(map[string]any{"type": "response.completed", "response": map[string]any{"status": "completed"}})
			}))
			defer s.Close()
			m := ai.Model{ID: "offline", API: ai.APIOpenAIResponses, BaseURL: s.URL}
			d, err := responses.New(ai.Config{Model: m, HTTPClient: s.Client()})
			if err != nil {
				t.Fatal(err)
			}
			first, idle := time.Second, 150*time.Millisecond
			if mode == "first-budget" {
				first = 500 * time.Millisecond
			}
			if mode == "idle-budget" {
				first = 2 * time.Second
			}
			a, err := agent.New(ai.NewClientWithDriver(d, m), agent.WithStreamTimeout(first, idle), agent.WithRetry(0, 0), agent.WithTools(progressTool{t, &finished, &runs}))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			var finalErr error
			progress := 0
			for e, err := range a.Run(ctx, ai.UserMessage("write")) {
				if err != nil {
					finalErr = err
				}
				if end, ok := e.(agent.TurnEnd); ok {
					finalErr = end.Err
				}
				if update, ok := e.(agent.InferenceUpdate); ok && update.Delta.Type == ai.EventProgress {
					progress++
					if mode == "cancel" {
						cancel()
					}
					if mode == "break" {
						break
					}
				}
			}
			select {
			case <-closed:
			case <-time.After(time.Second):
				t.Fatal("HTTP stream leaked after cancellation/break")
			}
			success := mode == "progress" || mode == "first-budget"
			if success {
				if finalErr != nil || runs.Load() != 1 || progress != 13 {
					t.Fatalf("err=%v runs=%d progress=%d", finalErr, runs.Load(), progress)
				}
			} else {
				if runs.Load() != 0 {
					t.Fatalf("failed inference executed %d tools", runs.Load())
				}
				if mode != "break" && finalErr == nil {
					t.Fatal("expected timeout/cancellation/provider failure")
				}
				if mode == "empty" || mode == "metadata" || mode == "stall" || mode == "idle-budget" {
					if !strings.Contains(finalErr.Error(), "went silent") {
						t.Fatalf("not watchdog timeout: %v", finalErr)
					}
				}
			}
		})
	}
}
