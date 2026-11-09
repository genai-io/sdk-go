package agent_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
)

func TestInferenceAdmissionRejectsBeforeProviderWithoutRecovery(t *testing.T) {
	failure := &ai.Error{Kind: ai.KindNetwork, Message: "commit failed"}
	d := aitest.Always(aitest.Says("must not run"))
	admitted := 0
	a := newAgent(t, d, agent.WithRetry(3, 0), agent.WithInferenceAdmission(func(_ context.Context, c agent.InferenceAdmission) error {
		admitted++
		if c.Request.System != "final" || c.Request.MaxTokens != 19 || c.Attempt != 1 {
			t.Errorf("not final: %+v", c)
		}
		return failure
	}), agent.WithHooks(agent.Hook{
		PreInfer: func(_ context.Context, inf *agent.Inference) error {
			inf.Options = append(inf.Options, ai.WithSystem("final"), ai.WithMaxTokens(19))
			return nil
		},
		OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Recovery, error) {
			t.Error("admission offered to recovery")
			return nil, nil
		},
	}))
	out, err := outcome(t, a, ai.UserMessage("go"))
	var admission *agent.AdmissionError
	if !errors.Is(err, failure) || !errors.As(err, &admission) || out.StopReason != agent.StopError || admitted != 1 || d.Calls() != 0 {
		t.Fatalf("out=%+v err=%v admitted=%d requests=%d", out, err, admitted, d.Calls())
	}
}

func TestSlowInferenceAdmissionDoesNotConsumeProviderTimeout(t *testing.T) {
	for _, reject := range []bool{true, false} {
		t.Run(fmt.Sprint(reject), func(t *testing.T) {
			failure := errors.New("durable commit rejected")
			d := aitest.Always(aitest.Says("answer"))
			admitted, recovered := 0, 0
			a := newAgent(t, d, agent.WithStreamTimeout(20*time.Millisecond, time.Second),
				agent.WithRetry(2, 0),
				agent.WithInferenceAdmission(func(ctx context.Context, _ agent.InferenceAdmission) error {
					admitted++
					time.Sleep(60 * time.Millisecond)
					if ctx.Err() != nil {
						t.Errorf("provider silence timer canceled admission: %v", ctx.Err())
					}
					if reject {
						return failure
					}
					return nil
				}),
				agent.WithHooks(agent.Hook{OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Recovery, error) {
					recovered++
					return nil, nil
				}}))
			out, err := outcome(t, a, ai.UserMessage("go"))
			if admitted != 1 || recovered != 0 {
				t.Fatalf("admitted=%d recovered=%d err=%v", admitted, recovered, err)
			}
			if reject {
				var admission *agent.AdmissionError
				if !errors.Is(err, failure) || !errors.As(err, &admission) || out.StopReason != agent.StopError || d.Calls() != 0 {
					t.Fatalf("out=%+v err=%v provider=%d", out, err, d.Calls())
				}
			} else if err != nil || out.StopReason != agent.StopEndTurn || d.Calls() != 1 {
				t.Fatalf("out=%+v err=%v provider=%d", out, err, d.Calls())
			}
		})
	}
}

func TestToolAdmissionAfterGatesAndRewrites(t *testing.T) {
	for _, block := range []bool{false, true} {
		t.Run(fmt.Sprint(block), func(t *testing.T) {
			failure := errors.New("commit failed")
			var ran atomic.Int32
			tool := agent.ToolFunc("echo", "Echo", func(_ context.Context, args struct {
				Text string `json:"text"`
			}) (agent.Result, error) {
				ran.Add(1)
				return agent.TextResult(args.Text), nil
			})
			d := aitest.New(aitest.Asks(ai.ToolCall{ID: "c", Name: "echo", Input: `{"text":"original"}`}), aitest.Says("done"))
			admitted := 0
			a := newAgent(t, d, agent.WithTools(tool), agent.WithHooks(agent.Hook{PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				return agent.Decision{Block: block, Arguments: `{"text":"rewritten"}`}, nil
			}}), agent.WithToolAdmission(func(_ context.Context, c agent.ToolAdmission) error {
				admitted++
				if c.Call.Input != `{"text":"rewritten"}` {
					t.Errorf("input=%s", c.Call.Input)
				}
				return failure
			}))
			out, err := outcome(t, a, ai.UserMessage("go"))
			if ran.Load() != 0 {
				t.Fatal("tool dispatched")
			}
			if block {
				if admitted != 0 || err != nil {
					t.Fatalf("blocked admitted=%d err=%v", admitted, err)
				}
			} else if admitted != 1 || !errors.Is(err, failure) || out.StopReason != agent.StopError || len(a.Messages()) != 2 {
				t.Fatalf("admitted=%d out=%+v messages=%v", admitted, out, a.Messages())
			}
		})
	}
}

func TestOutputAdmissionRejectsPostHookOutputBeforePublication(t *testing.T) {
	for _, toolOutput := range []bool{false, true} {
		t.Run(fmt.Sprint(toolOutput), func(t *testing.T) {
			failure := errors.New("output commit failed")
			tool := agent.ToolFunc("echo", "Echo", func(context.Context, struct{}) (agent.Result, error) { return agent.TextResult("raw"), nil })
			turns := []aitest.Turn{aitest.Says("raw")}
			if toolOutput {
				turns = []aitest.Turn{aitest.Asks(ai.ToolCall{ID: "c", Name: "echo", Input: `{}`}), aitest.Says("not reached")}
			}
			d := aitest.New(turns...)
			id := 0
			a := newAgent(t, d, agent.WithTools(tool), agent.WithMessageIDs(func() string { id++; return fmt.Sprint(id) }), agent.WithHooks(agent.Hook{
				PostInfer: func(_ context.Context, r *ai.Response) error {
					if !toolOutput {
						r.Content = ai.TextContent("final")
					}
					return nil
				},
				PostTool: func(context.Context, agent.PostToolContext) (*agent.Result, error) {
					r := agent.TextResult("final")
					return &r, nil
				},
			}), agent.WithOutputAdmission(func(_ context.Context, c agent.OutputAdmission) error {
				if toolOutput && c.Response != nil {
					return nil
				}
				if c.Message.ID == "" {
					t.Error("missing message identity")
				}
				text := c.Message.Text()
				if toolOutput {
					text = c.Message.ToolResults()[0].Text()
				}
				if text != "final" {
					t.Errorf("not post-hook: %q", text)
				}
				return failure
			}))
			events, err := collect(t, a, ai.UserMessage("go"))
			if !errors.Is(err, failure) {
				t.Fatalf("err=%v", err)
			}
			want := 1
			if toolOutput {
				want = 2
			}
			if len(a.Messages()) != want || d.Calls() != 1 {
				t.Fatalf("history=%v requests=%d", a.Messages(), d.Calls())
			}
			added := 0
			for _, e := range events {
				if _, ok := e.(agent.MessageAdded); ok {
					added++
				}
			}
			if added != want {
				t.Fatalf("published %d messages", added)
			}
		})
	}
}

func TestParallelAdmissionFailureDrainsAdmittedWork(t *testing.T) {
	failure := errors.New("second commit failed")
	started, release := make(chan struct{}), make(chan struct{})
	var ran atomic.Int32
	tool := agent.ToolFunc("work", "Work", func(context.Context, struct{}) (agent.Result, error) {
		ran.Add(1)
		close(started)
		<-release
		return agent.TextResult("done"), nil
	})
	d := aitest.New(aitest.Asks(ai.ToolCall{ID: "1", Name: "work", Input: `{}`}, ai.ToolCall{ID: "2", Name: "work", Input: `{}`}, ai.ToolCall{ID: "3", Name: "work", Input: `{}`}))
	var admitted []string
	a := newAgent(t, d, agent.WithTools(tool), agent.WithToolAdmission(func(_ context.Context, c agent.ToolAdmission) error {
		admitted = append(admitted, c.Call.ID)
		if c.Call.ID == "2" {
			<-started
			close(release)
			return failure
		}
		return nil
	}))
	events, err := collect(t, a, ai.UserMessage("go"))
	ends := 0
	for _, event := range events {
		if end, ok := event.(agent.ToolEnd); ok {
			ends++
			if end.Started != (end.ID == "1") {
				t.Errorf("call %s Started=%v", end.ID, end.Started)
			}
		}
	}
	if ends != 3 {
		t.Errorf("ToolEnd count=%d, want 3", ends)
	}
	if !errors.Is(err, failure) || ran.Load() != 1 || !reflect.DeepEqual(admitted, []string{"1", "2"}) || len(a.Messages()) != 2 {
		t.Fatalf("err=%v ran=%d admitted=%v history=%v", err, ran.Load(), admitted, a.Messages())
	}
}

func TestAdmissionSnapshotsCannotRewriteDispatchOrOutput(t *testing.T) {
	d := aitest.New(aitest.Says("answer"))
	options := 0
	client := ai.NewClientWithDriver(d, ai.Model{ID: "chosen", API: "stub"}, ai.WithMaxTokens(27))
	var acceptedID string
	a, err := agent.New(client, agent.WithMessageIDs(func() string { return "message-id" }), agent.WithHooks(agent.Hook{PreInfer: func(_ context.Context, inf *agent.Inference) error {
		inf.Options = append(inf.Options, func(r *ai.Request) { options++; r.System = "final" })
		return nil
	}}), agent.WithInferenceAdmission(func(_ context.Context, c agent.InferenceAdmission) error {
		if c.Model.ID != "chosen" || c.Request.MaxTokens != 27 {
			t.Errorf("missing defaults/routing: %+v", c)
		}
		c.Request.System = "mutated"
		c.Request.Messages[0].Content[0].Text = "mutated"
		return nil
	}), agent.WithOutputAdmission(func(_ context.Context, c agent.OutputAdmission) error {
		acceptedID = c.Message.ID
		c.Message.Content[0].Text = "mutated"
		c.Response.Content[0].Text = "mutated"
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = outcome(t, a, ai.UserMessage("input"))
	if err != nil || options != 1 || d.Last().System != "final" || d.Last().Messages[0].Text() != "input" || a.Messages()[1].Text() != "answer" || a.Messages()[1].ID != acceptedID {
		t.Fatalf("err=%v options=%d request=%+v history=%v", err, options, d.Last(), a.Messages())
	}
}

func TestCancellationDuringAdmissionPreventsDispatch(t *testing.T) {
	for _, toolAdmission := range []bool{false, true} {
		t.Run(fmt.Sprint(toolAdmission), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var ran atomic.Int32
			tool := agent.ToolFunc("work", "Work", func(context.Context, struct{}) (agent.Result, error) {
				ran.Add(1)
				return agent.TextResult("done"), nil
			})
			d := aitest.New(aitest.Asks(ai.ToolCall{ID: "1", Name: "work", Input: `{}`}))
			opts := []agent.Option{agent.WithTools(tool)}
			if toolAdmission {
				opts = append(opts, agent.WithToolAdmission(func(context.Context, agent.ToolAdmission) error { cancel(); return nil }))
			} else {
				opts = append(opts, agent.WithInferenceAdmission(func(context.Context, agent.InferenceAdmission) error { cancel(); return nil }))
			}
			a := newAgent(t, d, opts...)
			ends := 0
			for event := range a.Run(ctx, ai.UserMessage("go")) {
				if end, ok := event.(agent.ToolEnd); ok {
					ends++
					if end.Started {
						t.Error("cancelled-before-dispatch call reported Started")
					}
				}
			}
			if toolAdmission && ends != 1 {
				t.Errorf("ToolEnd count=%d, want 1", ends)
			}
			wantCalls := 0
			if toolAdmission {
				wantCalls = 1
			}
			if ran.Load() != 0 || d.Calls() != wantCalls {
				t.Fatalf("tool ran=%d provider=%d", ran.Load(), d.Calls())
			}
		})
	}
}

func TestSequentialAdmissionFailurePreventsQueuedTools(t *testing.T) {
	failure := errors.New("commit failed")
	var ran atomic.Int32
	tool := agent.Sequential(agent.ToolFunc("work", "Work", func(context.Context, struct{}) (agent.Result, error) {
		ran.Add(1)
		return agent.TextResult("done"), nil
	}))
	d := aitest.New(aitest.Asks(ai.ToolCall{ID: "1", Name: "work", Input: `{}`}, ai.ToolCall{ID: "2", Name: "work", Input: `{}`}, ai.ToolCall{ID: "3", Name: "work", Input: `{}`}))
	admitted := 0
	a := newAgent(t, d, agent.WithTools(tool), agent.WithToolAdmission(func(_ context.Context, c agent.ToolAdmission) error {
		admitted++
		if c.Call.ID == "2" {
			if ran.Load() != 1 {
				t.Error("sequential admission preceded prior completion")
			}
			return failure
		}
		return nil
	}))
	_, err := outcome(t, a, ai.UserMessage("go"))
	if !errors.Is(err, failure) || admitted != 2 || ran.Load() != 1 {
		t.Fatalf("err=%v admitted=%d ran=%d", err, admitted, ran.Load())
	}
}

func TestAdmissionNotCalledByHistoryReplay(t *testing.T) {
	calls := 0
	a := newAgent(t, aitest.New(), agent.WithInferenceAdmission(func(context.Context, agent.InferenceAdmission) error { calls++; return nil }), agent.WithToolAdmission(func(context.Context, agent.ToolAdmission) error { calls++; return nil }), agent.WithOutputAdmission(func(context.Context, agent.OutputAdmission) error { calls++; return nil }))
	a.SetMessages([]ai.Message{ai.UserMessage("saved"), ai.Message{Role: ai.RoleAssistant, Content: ai.TextContent("answer")}})
	a.AddMessages(ai.UserMessage("queued"))
	_ = a.Messages()
	if calls != 0 {
		t.Fatalf("replay called admission %d times", calls)
	}
}
