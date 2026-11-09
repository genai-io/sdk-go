package agent_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
)

func TestToolEndStarted(t *testing.T) {
	for _, mode := range []string{"success", "error", "panic", "denied", "unknown", "invalid", "unadmitted", "abandoned", "cancelled"} {
		t.Run(mode, func(t *testing.T) {
			var ran atomic.Int32
			tool := agent.ToolFunc("work", "Work", func(context.Context, struct {
				Text string `json:"text"`
			}) (agent.Result, error) {
				ran.Add(1)
				if mode == "panic" {
					panic("failure")
				}
				if mode == "error" {
					return agent.Result{}, errors.New("failure")
				}
				return agent.TextResult("done"), nil
			})
			call := ai.ToolCall{ID: "1", Name: "work", Input: `{"text":"hello"}`}
			if mode == "unknown" {
				call.Name = "missing"
			}
			if mode == "invalid" {
				call.Input = `{"text":42}`
			}
			opts := []agent.Option{agent.WithTools(tool)}
			if mode == "denied" || mode == "abandoned" {
				opts = append(opts, agent.WithHooks(agent.Hook{PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
					if mode == "abandoned" {
						return agent.Decision{}, errors.New("hook failed")
					}
					return agent.Decision{Block: true}, nil
				}}))
			}
			if mode == "unadmitted" {
				opts = append(opts, agent.WithToolAdmission(func(context.Context, agent.ToolAdmission) error { return errors.New("commit failed") }))
			}
			if mode == "cancelled" {
				opts = append(opts, agent.WithToolAdmission(func(context.Context, agent.ToolAdmission) error { return context.Canceled }))
			}
			a := newAgent(t, aitest.New(aitest.Asks(call), aitest.Says("done")), opts...)
			events, _ := collect(t, a, ai.UserMessage("go"))
			want := mode == "success" || mode == "error" || mode == "panic"
			ends := 0
			for _, e := range events {
				if end, ok := e.(agent.ToolEnd); ok {
					ends++
					if end.Started != want {
						t.Errorf("Started=%v want %v", end.Started, want)
					}
				}
			}
			count := int32(0)
			if want {
				count = 1
			}
			if ends != 1 || ran.Load() != count {
				t.Fatalf("ends=%d runs=%d want runs=%d", ends, ran.Load(), count)
			}
		})
	}
}
