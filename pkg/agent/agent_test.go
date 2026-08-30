package agent_test

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"iter"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// A scripted driver stands in for an endpoint: each call consumes one script,
// so a test states the model's behaviour as data and asserts on the event
// sequence that behaviour produces.
type scripted struct {
	mu      sync.Mutex
	scripts [][]ai.Delta
	errs    []error
	calls   int
	// got is what reached the driver, after the client merged its defaults in
	// and repaired the history: the last word on what was really sent.
	got []*ai.Request
}

func (d *scripted) Name() string { return "scripted" }

func (d *scripted) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	d.got = append(d.got, req)
	d.mu.Unlock()

	return func(yield func(ai.Delta, error) bool) {
		// Whatever the script has for this call goes out first, then the
		// failure if one was set — which is how an endpoint really fails: some
		// of the answer arrives, and the tokens for it are already spent.
		if n < len(d.scripts) {
			for _, delta := range d.scripts[n] {
				if !yield(delta, nil) {
					return
				}
			}
		}
		if n < len(d.errs) && d.errs[n] != nil {
			yield(ai.Delta{}, d.errs[n])
			return
		}
		if n >= len(d.scripts) {
			yield(ai.Delta{}, errors.New("scripted: no script for call "+fmt.Sprint(n)))
		}
	}
}

func text(s string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.TextBlock(s)},
		{EndBlock: true},
		{StopReason: ai.StopEndTurn},
	}
}

func toolCall(id, name, input string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.ToolCallBlock(ai.ToolCall{ID: id, Name: name, Input: input})},
		{StopReason: ai.StopToolUse},
	}
}

func newAgent(t *testing.T, d ai.Driver, opts ...agent.Option) *agent.Agent {
	t.Helper()
	client := ai.NewClientWithDriver(d, ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client, append([]agent.Option{agent.WithSystem("You are a test.")}, opts...)...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

// names renders an event sequence the way the design documents it, so a golden
// assertion reads like the diagram it came from.
func names(events []agent.Event) []string {
	out := make([]string, 0, len(events))
	for _, e := range events {
		name := strings.TrimPrefix(fmt.Sprintf("%T", e), "agent.")
		switch v := e.(type) {
		case agent.MessageAdded:
			name += "(" + string(v.Message.Role) + ")"
		case agent.MessageStart:
			name += fmt.Sprintf("(attempt=%d)", v.Attempt)
		case agent.MessageEnd:
			if v.Err != nil {
				name += "(err)"
			}
		}
		out = append(out, name)
	}
	return out
}

// collect drives one exchange and returns everything the agent reported —
// which is what a golden-sequence assertion compares against. It is also the
// smallest complete use of the package: two channels and a range.
// outcome folds an exchange the way a caller with nothing to render would:
// range it, keep the TurnEnd. Four lines, at every such call site — which is
// what a fold in the SDK would have saved.
func outcome(t *testing.T, a *agent.Agent, msgs ...ai.Message) (agent.TurnEnd, error) {
	t.Helper()

	var out agent.TurnEnd
	for e, err := range a.Run(context.Background(), msgs...) {
		if err != nil {
			return out, err
		}
		if v, ok := e.(agent.TurnEnd); ok {
			out = v
		}
	}
	return out, out.Err
}

func collect(t *testing.T, a *agent.Agent, msgs ...ai.Message) ([]agent.Event, error) {
	t.Helper()

	var events []agent.Event
	for e, err := range a.Run(context.Background(), msgs...) {
		if err != nil {
			return events, err
		}
		events = append(events, e)
	}

	// A failed exchange is reported on the stream, not returned.
	for _, e := range events {
		if v, ok := e.(agent.TurnEnd); ok && v.Err != nil {
			return events, v.Err
		}
	}
	return events, nil
}

// steps counts what the loop counts: a model call that is not a retry.
// TurnEnd does not carry this — it is on the stream, and this is the fold.
func steps(events []agent.Event) int {
	n := 0
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok && v.Attempt == 1 {
			n++
		}
	}
	return n
}

func assertSequence(t *testing.T, got []agent.Event, want []string) {
	t.Helper()
	if g := names(got); !slices.Equal(g, want) {
		t.Errorf("event sequence\n got: %v\nwant: %v", g, want)
	}
}

// The plain exchange from the design's first sequence diagram.
func TestTurnEmitsTheDocumentedTextSequence(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("hello there")}})

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	assertSequence(t, events, []string{
		"TurnStart",
		"MessageAdded(user)",
		"MessageStart(attempt=1)",
		"MessageUpdate", // block start
		"MessageUpdate", // block delta
		"MessageUpdate", // block end
		"MessageEnd",
		"MessageAdded(assistant)",
		"TurnEnd",
	})

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopEndTurn {
		t.Errorf("stop reason = %q, want end_turn", last.StopReason)
	}
	msgs := a.Messages()
	if got := msgs[len(msgs)-1].Text(); got != "hello there" {
		t.Errorf("the answer that entered the conversation = %q", got)
	}
}

// The tool sequence, including the second inference that reads the result.
func TestTurnWithAToolRunsASecondInference(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo the argument.",
		func(_ context.Context, args struct {
			Text string `json:"text"`
		}) (agent.Result, error) {
			return agent.TextResult("echoed: " + args.Text), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "echo", `{"text":"hi"}`),
		text("done"),
	}}, agent.WithTools(echo))

	events, err := collect(t, a, ai.UserMessage("echo hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	assertSequence(t, events, []string{
		"TurnStart",
		"MessageAdded(user)",
		"MessageStart(attempt=1)",
		"MessageUpdate", // the tool call block arrives complete: start
		"MessageUpdate", // ...and end
		"MessageEnd",
		"MessageAdded(assistant)",
		"ToolStart",
		"ToolEnd",
		"MessageAdded(user)", // tool results are a user turn on every protocol
		"MessageStart(attempt=1)",
		"MessageUpdate",
		"MessageUpdate",
		"MessageUpdate",
		"MessageEnd",
		"MessageAdded(assistant)",
		"TurnEnd",
	})

	msgs := a.Messages()
	results := msgs[2].ToolResults()
	if len(results) != 1 || results[0].Content != "echoed: hi" {
		t.Errorf("tool result = %+v", results)
	}
}

// ToolEnd is emitted the moment the tool lands, before PostTool runs, so a
// reader never waits on hooks. The consequence is pinned here rather than left
// to be discovered: a hook that replaces a result changes what the model is
// told and not what the stream reported, and the two are allowed to differ.
func TestToolEndCarriesTheToolsOwnResult(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo it.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.TextResult("raw"), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "echo", `{}`),
		text("done"),
	}}, agent.WithTools(echo), agent.WithHooks(agent.Hook{
		PostTool: func(context.Context, agent.PostToolContext) (*agent.Result, error) {
			replacement := agent.TextResult("replaced")
			return &replacement, nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	var ended *agent.ToolEnd
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok {
			ended = &v
		}
	}
	if ended == nil {
		t.Fatal("no ToolEnd was emitted")
	}
	if got := ended.Result.Text(); got != "raw" {
		t.Errorf("ToolEnd result = %q, want the tool's own — the emit waited on the hook", got)
	}

	results := a.Messages()[2].ToolResults()
	if len(results) != 1 || results[0].Content != "replaced" {
		t.Errorf("the model was told %+v, want the hook's replacement", results)
	}
}

// The claim that lets a retry need no event of its own: it is two spans with
// nothing appended between them, and that absence is the signal.
func TestARetryAppendsNothingBetweenAttempts(t *testing.T) {
	a := newAgent(t, &scripted{
		errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		scripts: [][]ai.Delta{nil, text("second time lucky")},
	}, agent.WithRetry(3, 0))

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	assertSequence(t, events, []string{
		"TurnStart",
		"MessageAdded(user)",
		"MessageStart(attempt=1)",
		"MessageEnd(err)", // carries the error, and no message follows
		"MessageStart(attempt=2)",
		"MessageUpdate",
		"MessageUpdate",
		"MessageUpdate",
		"MessageEnd",
		"MessageAdded(assistant)",
		"TurnEnd",
	})

	// Between the failed attempt and the retry, nothing entered the
	// conversation. That is what tells a consumer to discard what it drew.
	firstEnd := slices.IndexFunc(events, func(e agent.Event) bool {
		v, ok := e.(agent.MessageEnd)
		return ok && v.Err != nil
	})
	secondStart := slices.IndexFunc(events, func(e agent.Event) bool {
		v, ok := e.(agent.MessageStart)
		return ok && v.Attempt == 2
	})
	for _, e := range events[firstEnd+1 : secondStart] {
		if v, ok := e.(agent.MessageEnd); ok && v.Err == nil {
			t.Fatal("a message was appended between a failed attempt and its retry")
		}
	}
}

func TestAFatalErrorEndsTheTurnAndIsYielded(t *testing.T) {
	a := newAgent(t, &scripted{errs: []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}}})

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err == nil {
		t.Fatal("expected the auth failure to reach the caller")
	}
	if !ai.IsAuth(err) {
		t.Errorf("error = %v, want an auth failure", err)
	}
	// One attempt only: an auth failure is not retryable.
	attempts := 0
	for _, e := range events {
		if _, ok := e.(agent.MessageStart); ok {
			attempts++
		}
	}
	if attempts != 1 {
		t.Errorf("attempts = %d, want 1", attempts)
	}
}

// Two orders, both load-bearing: completion order for the spans, source order
// for what lands in the conversation.
func TestParallelToolsEndInCompletionOrderButRecordInSourceOrder(t *testing.T) {
	release := map[string]chan struct{}{
		"a": make(chan struct{}),
		"b": make(chan struct{}),
	}
	wait := agent.ToolFunc("wait", "Wait for a signal.",
		func(ctx context.Context, args struct {
			Key string `json:"key"`
		}) (agent.Result, error) {
			<-release[args.Key]
			return agent.TextResult("done: " + args.Key), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "wait", Input: `{"key":"a"}`})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "2", Name: "wait", Input: `{"key":"b"}`})},
			{StopReason: ai.StopToolUse},
		},
		text("both done"),
	}}, agent.WithTools(wait))

	// b finishes first, though the model asked for a first.
	go func() {
		time.Sleep(10 * time.Millisecond)
		close(release["b"])
		time.Sleep(10 * time.Millisecond)
		close(release["a"])
	}()

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	var ended []string
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok {
			ended = append(ended, v.ID)
		}
	}
	if want := []string{"2", "1"}; !slices.Equal(ended, want) {
		t.Errorf("ToolEnd order = %v, want completion order %v", ended, want)
	}

	results := a.Messages()[2].ToolResults()
	var recorded []string
	for _, r := range results {
		recorded = append(recorded, r.ToolCallID)
	}
	if want := []string{"1", "2"}; !slices.Equal(recorded, want) {
		t.Errorf("recorded order = %v, want source order %v", recorded, want)
	}
}

func TestTheGateSeesTheMessageThatRequestedTheCall(t *testing.T) {
	var seen int
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "noop", `{}`),
		text("fine"),
	}}, ai.Model{ID: "stub", API: "stub"})

	noop := agent.ToolFunc("noop", "Do nothing.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		})

	a, err := agent.New(client,
		agent.WithTools(noop),
		agent.WithHooks(agent.Hook{
			PreTool: func(_ context.Context, c agent.PreToolContext) (agent.Decision, error) {
				seen = len(c.Messages)
				last := c.Messages[len(c.Messages)-1]
				if last.Role != ai.RoleAssistant || len(last.ToolCalls()) == 0 {
					t.Errorf("gate saw %q as the last message, want the assistant turn that asked", last.Role)
				}
				return agent.Decision{}, nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if seen != 2 {
		t.Errorf("gate saw %d messages, want the user turn and the assistant turn", seen)
	}
}

func TestABlockedCallBecomesAToolErrorTheModelCanRead(t *testing.T) {
	dangerous := agent.ToolFunc("rm", "Delete everything.",
		func(context.Context, struct{}) (agent.Result, error) {
			t.Fatal("a blocked tool must not run")
			return agent.Result{}, nil
		})

	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "rm", `{}`),
		text("understood"),
	}}, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client,
		agent.WithTools(dangerous),
		agent.WithHooks(agent.Hook{
			PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				return agent.Decision{Block: true, Reason: "rm is disabled in this workspace"}, nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("delete everything")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	results := a.Messages()[2].ToolResults()
	if len(results) != 1 || !results[0].IsError {
		t.Fatalf("blocked call did not become a tool error: %+v", results)
	}
	if !strings.Contains(results[0].Content, "disabled") {
		t.Errorf("the model was told %q, which does not say why", results[0].Content)
	}
}

func TestAnUnknownToolIsReportedRatherThanFatal(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "nosuchtool", `{}`),
		text("sorry"),
	}})

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("an unknown tool should not fail the turn: %v", err)
	}
	results := a.Messages()[2].ToolResults()
	if len(results) != 1 || !results[0].IsError {
		t.Fatalf("unknown tool produced %+v", results)
	}
}

func TestBadArgumentsAreCaughtBeforeTheToolRuns(t *testing.T) {
	strict := agent.ToolFunc("strict", "Needs a count.",
		func(_ context.Context, args struct {
			Count int `json:"count"`
		}) (agent.Result, error) {
			t.Fatal("the tool ran on arguments that do not match its schema")
			return agent.Result{}, nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "strict", `{"count":"not a number"}`),
		text("retrying"),
	}}, agent.WithTools(strict))

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if results := a.Messages()[2].ToolResults(); len(results) != 1 || !results[0].IsError {
		t.Fatalf("schema violation produced %+v", results)
	}
}

func TestMaxStepsStopsTheExchange(t *testing.T) {
	loop := agent.ToolFunc("again", "Ask again.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("again"), nil
		})

	scripts := make([][]ai.Delta, 10)
	for i := range scripts {
		scripts[i] = toolCall(fmt.Sprintf("c%d", i), "again", `{}`)
	}

	client := ai.NewClientWithDriver(&scripted{scripts: scripts}, ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client, agent.WithTools(loop), agent.WithMaxSteps(3))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopMaxSteps {
		t.Errorf("stop reason = %q, want max_steps", last.StopReason)
	}
	if n := steps(events); n != 3 {
		t.Errorf("steps = %d, want 3", n)
	}
}

func TestATerminatingToolEndsTheExchangeWithoutAnotherCall(t *testing.T) {
	done := agent.ToolFunc("finish", "Report completion.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.Result{Content: ai.TextContent("finished"), Terminate: true}, nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("call-1", "finish", `{}`),
		text("should never be asked for"),
	}}, agent.WithTools(done))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopTerminated {
		t.Errorf("stop reason = %q, want terminated", last.StopReason)
	}
	inferences := 0
	for _, e := range events {
		if _, ok := e.(agent.MessageStart); ok {
			inferences++
		}
	}
	if inferences != 1 {
		t.Errorf("inferences = %d, want 1 — the follow-up call should be skipped", inferences)
	}
}

// Several hooks run in order, and the first refusal is final: a later one must
// not be able to allow what an earlier one denied, or adding a hook could
// weaken the gate.
func TestTheFirstRefusalIsFinal(t *testing.T) {
	tool := agent.ToolFunc("rm", "Delete things.",
		func(context.Context, struct{}) (agent.Result, error) {
			t.Fatal("a refused tool ran")
			return agent.Result{}, nil
		})

	var asked []string
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		toolCall("c1", "rm", `{}`),
		text("understood"),
	}}, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client,
		agent.WithTools(tool),
		agent.WithHooks(
			agent.Hook{PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				asked = append(asked, "first")
				return agent.Decision{Block: true, Reason: "not in this workspace"}, nil
			}},
			agent.Hook{PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				asked = append(asked, "second")
				return agent.Decision{}, nil // would allow it
			}},
		))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("delete everything")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	if want := []string{"first"}; !slices.Equal(asked, want) {
		t.Errorf("hooks asked = %v, want %v — a refusal must end the questioning", asked, want)
	}
	results := a.Messages()[2].ToolResults()
	if len(results) != 1 || !results[0].IsError {
		t.Fatalf("the refusal did not become a tool error: %+v", results)
	}
}

// Rewrites chain: each hook sees what the one before it produced.
func TestHooksChainTheirRewrites(t *testing.T) {
	var got string
	tool := agent.ToolFunc("echo", "Echo it back.",
		func(_ context.Context, args struct {
			Text string `json:"text"`
		}) (agent.Result, error) {
			got = args.Text
			return agent.TextResult(args.Text), nil
		})

	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		toolCall("c1", "echo", `{"text":"one"}`),
		text("done"),
	}}, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client,
		agent.WithTools(tool),
		agent.WithHooks(
			agent.Hook{PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				return agent.Decision{Arguments: `{"text":"two"}`}, nil
			}},
			agent.Hook{PreTool: func(_ context.Context, c agent.PreToolContext) (agent.Decision, error) {
				if c.Call.Input != `{"text":"two"}` {
					t.Errorf("the second hook saw %q, want the first hook's rewrite", c.Call.Input)
				}
				return agent.Decision{Arguments: `{"text":"three"}`}, nil
			}},
		))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if got != "three" {
		t.Errorf("the tool ran on %q, want the last rewrite", got)
	}
}

// The same claim without the concurrency: every exit sets a stop reason.
func TestEveryOutcomeSaysWhyItStopped(t *testing.T) {
	for _, tc := range []struct {
		name   string
		build  func(*testing.T) *agent.Agent
		reason agent.StopReason
	}{
		{"the model answered", func(t *testing.T) *agent.Agent {
			return newAgent(t, &scripted{scripts: [][]ai.Delta{text("done")}})
		}, agent.StopEndTurn},

		{"the step budget ran out", func(t *testing.T) *agent.Agent {
			again := agent.ToolFunc("again", "Ask again.",
				func(context.Context, struct{}) (agent.Result, error) {
					return agent.TextResult("again"), nil
				})
			scripts := make([][]ai.Delta, 8)
			for i := range scripts {
				scripts[i] = toolCall(fmt.Sprintf("c%d", i), "again", `{}`)
			}
			client := ai.NewClientWithDriver(&scripted{scripts: scripts}, ai.Model{ID: "stub", API: "stub"})
			a, err := agent.New(client, agent.WithTools(again), agent.WithMaxSteps(2))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			return a
		}, agent.StopMaxSteps},

		{"a tool asked to stop", func(t *testing.T) *agent.Agent {
			done := agent.ToolFunc("finish", "Report completion.",
				func(context.Context, struct{}) (agent.Result, error) {
					return agent.Result{Content: ai.TextContent("finished"), Terminate: true}, nil
				})
			return newAgent(t, &scripted{scripts: [][]ai.Delta{toolCall("c1", "finish", `{}`)}},
				agent.WithTools(done))
		}, agent.StopTerminated},

		{"the model call failed", func(t *testing.T) *agent.Agent {
			return newAgent(t, &scripted{errs: []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}}})
		}, agent.StopError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events, _ := collect(t, tc.build(t), ai.UserMessage("go"))
			last := events[len(events)-1].(agent.TurnEnd)
			if last.StopReason != tc.reason {
				t.Errorf("stop reason = %q, want %q", last.StopReason, tc.reason)
			}
			if last.StopReason == "" {
				t.Error("the outcome does not say why it stopped")
			}
		})
	}
}

// A failed attempt still spent whatever it spent. Folding the retry into the
// step made that visible: usage now accumulates per attempt, not per step.
func TestAFailedAttemptStillCountsWhatItCost(t *testing.T) {
	a := newAgent(t, &scripted{
		errs: []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		scripts: [][]ai.Delta{
			{{Usage: &ai.Usage{Input: 40, Output: 0}}}, // the attempt that failed
			{
				{Block: ai.TextBlock("second time lucky")},
				{EndBlock: true},
				{Usage: &ai.Usage{Input: 40, Output: 6}, StopReason: ai.StopEndTurn},
			},
		},
	}, agent.WithRetry(3, 0))

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if n := steps(events); n != 1 {
		t.Errorf("steps = %d, want 1 — two attempts are still one step", n)
	}
	attempts := 0
	for _, e := range events {
		if _, ok := e.(agent.MessageStart); ok {
			attempts++
		}
	}
	if attempts != 2 {
		t.Errorf("attempts = %d, want 2", attempts)
	}
	if got := last.Usage.Output; got != 6 {
		t.Errorf("output tokens = %d, want 6", got)
	}
}

// PreInfer runs at the top of every step, not once per turn — a hook that
// prunes history has to see the history each time it grew.
func TestPreInferRunsOnEveryStep(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo it back.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		})

	var sizes []int
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		toolCall("c1", "echo", `{}`),
		text("done"),
	}}, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client,
		agent.WithTools(echo),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, inf *agent.Inference) error {
				sizes = append(sizes, len(inf.Messages))
				return nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	// Two steps: the first sees the user turn, the second also sees the
	// assistant turn and the tool results.
	if want := []int{1, 3}; !slices.Equal(sizes, want) {
		t.Errorf("PreInfer saw %v messages, want %v", sizes, want)
	}
}

// A hook edits the call, not the agent: what it changes goes out this once and
// the agent keeps what it had.
func TestPreInferChangesTheCallNotTheAgent(t *testing.T) {
	tools := []agent.Tool{
		agent.ToolFunc("keep", "Stays.", func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		}),
		agent.ToolFunc("hide", "Hidden for one call.", func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		}),
	}

	var sent *agent.Inference
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("fine")}}, ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client,
		agent.WithSystem("the agent's own prompt"),
		agent.WithTools(tools...),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, inf *agent.Inference) error {
				inf.System += "\n\nand a line true only right now"
				inf.Tools = inf.Tools[:1] // hide the second for this call
				inf.Messages = append(inf.Messages, ai.UserMessage("injected"))
				return nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok {
			sent = v.Inference
		}
	}

	if sent == nil {
		t.Fatal("no request was reported")
	}
	if !strings.Contains(sent.System, "only right now") {
		t.Errorf("the system edit did not go out: %q", sent.System)
	}
	if len(sent.Tools) != 1 {
		t.Errorf("sent %d tools, want the one the hook left", len(sent.Tools))
	}
	if len(sent.Messages) != 2 {
		t.Errorf("sent %d messages, want the injected one too", len(sent.Messages))
	}

	// None of it stuck to the agent.
	if a.System() != "the agent's own prompt" {
		t.Errorf("the agent's prompt changed: %q", a.System())
	}
	if len(a.Tools()) != 2 {
		t.Errorf("the agent holds %d tools, want both", len(a.Tools()))
	}
	if len(a.Messages()) != 2 {
		t.Errorf("the injected message stuck to the conversation: %d messages", len(a.Messages()))
	}
}

// A PreInfer error ends the turn before the model is called at all. It is
// not turned into a message the model gets to see: nothing had happened yet,
// and inventing a turn to carry the error would be reporting something that
// never took place.
func TestAPreInferErrorEndsTheTurnBeforeAnythingIsSent(t *testing.T) {
	driver := &scripted{scripts: [][]ai.Delta{text("never reached")}}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"})

	var second bool
	a, err := agent.New(client, agent.WithHooks(
		agent.Hook{PreInfer: func(context.Context, *agent.Inference) error {
			return errors.New("the context is too large to send")
		}},
		agent.Hook{PreInfer: func(context.Context, *agent.Inference) error {
			second = true
			return nil
		}},
	))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err == nil {
		t.Fatal("the hook's error never reached the caller")
	}

	if second {
		t.Error("a hook after the one that failed still ran")
	}
	driver.mu.Lock()
	calls := driver.calls
	driver.mu.Unlock()
	if calls != 0 {
		t.Errorf("the model was called %d times after a hook refused", calls)
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", last.StopReason)
	}
	if last.Err == nil || !strings.Contains(last.Err.Error(), "too large") {
		t.Errorf("the outcome does not say what went wrong: %v", last.Err)
	}
	if u := last.Usage; u.Total() != 0 {
		t.Errorf("usage = %+v, want nothing — the model was never called", u)
	}

	// The user's message is in the conversation; no assistant turn followed it.
	if got := a.Messages(); len(got) != 1 {
		t.Errorf("conversation holds %d messages, want just the input", len(got))
	}
}

// The hook is handed this agent's own request: its prompt, its conversation,
// its tools. Not the client's — the client merges its defaults and repairs the
// history after the hook has had its say, so what the hook edits is an input to
// that, not a copy of its output. MessageStart reports the same request,
// so a consumer sees exactly what the hook saw.
func TestPreInferSeesTheAgentsOwnRequest(t *testing.T) {
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("done")}},
		ai.Model{ID: "stub", API: "stub"},
		ai.WithMaxTokens(4096)) // a client default, applied after the hook

	var seen *agent.Inference
	a, err := agent.New(client,
		agent.WithSystem("mine"),
		agent.WithMessages([]ai.Message{ai.UserMessage("earlier")}),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, inf *agent.Inference) error {
				seen = inf
				return nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go on"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if seen == nil {
		t.Fatal("the hook never ran")
	}
	if seen.System != "mine" {
		t.Errorf("System = %q, want the agent's", seen.System)
	}
	if len(seen.Messages) != 2 {
		t.Errorf("the hook saw %d messages, want the agent's 2", len(seen.Messages))
	}

	var sent *agent.Inference
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok {
			sent = v.Inference
		}
	}
	if sent != seen {
		t.Error("MessageStart reported a different request than the hook was given")
	}
}

// PreInfer runs before every attempt, not before the first one. A retry is a
// new call to the endpoint and gets to be a new decision — and the request is
// assembled fresh each time, so a hook is never handed its own previous edits.
func TestPreInferRunsBeforeEveryAttempt(t *testing.T) {
	var prompts []string
	client := ai.NewClientWithDriver(&scripted{
		errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		scripts: [][]ai.Delta{nil, text("second time lucky")},
	}, ai.Model{ID: "stub", API: "stub"})

	attempts := 0
	a, err := agent.New(client,
		agent.WithSystem("base"),
		agent.WithRetry(3, 0),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, inf *agent.Inference) error {
				attempts++
				// An edit that would compound if the request were reused.
				inf.System += " · attempt"
				prompts = append(prompts, inf.System)
				return nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("hi")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	if attempts != 2 {
		t.Errorf("PreInfer ran %d times, want once per attempt", attempts)
	}
	for i, p := range prompts {
		if p != "base · attempt" {
			t.Errorf("attempt %d saw %q — the hook was handed its own edit", i+1, p)
		}
	}
}

// Several PreInfer hooks run in order, each seeing what the one before it did.
func TestPreInferHooksChain(t *testing.T) {
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("fine")}},
		ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client,
		agent.WithSystem("one"),
		agent.WithHooks(
			agent.Hook{PreInfer: func(_ context.Context, inf *agent.Inference) error {
				inf.System += " two"
				return nil
			}},
			agent.Hook{PreInfer: func(_ context.Context, inf *agent.Inference) error {
				if inf.System != "one two" {
					t.Errorf("the second hook saw %q, want the first hook's edit", inf.System)
				}
				inf.System += " three"
				return nil
			}},
		))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok && v.Inference.System != "one two three" {
			t.Errorf("sent %q, want every hook's edit", v.Inference.System)
		}
	}
}

// A hook edits what the agent contributes — prompt, conversation, toolset —
// directly, and reaches everything else by appending an option. Both halves
// are pinned, and the second is the point: a seam that silently ignores what
// is written to it is worse than one that never offered the field, and writing
// inf.MaxTokens used to be exactly that.
func TestAPreInferHookEditsTheCallItIsGiven(t *testing.T) {
	weather := agent.ToolFunc("weather", "Look up the weather.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.TextResult("mild"), nil
		})

	driver := &scripted{scripts: [][]ai.Delta{text("ok")}}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"},
		ai.WithMaxTokens(4096), ai.WithEffort(ai.EffortLow))

	temp := 0.25
	a, err := agent.New(client,
		agent.WithSystem("original"),
		agent.WithTools(weather),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, inf *agent.Inference) error {
				inf.System = "set by the hook"
				inf.Tools = nil
				inf.Messages = append(inf.Messages, ai.UserMessage("and this"))

				// Everything else is a layer, and layers are unambiguous:
				// present means asked for, absent means left alone.
				inf.Options = append(inf.Options,
					ai.WithMaxTokens(99), ai.WithTemperature(temp))
				return nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	driver.mu.Lock()
	defer driver.mu.Unlock()
	if len(driver.got) != 1 {
		t.Fatalf("the driver saw %d requests, want 1", len(driver.got))
	}
	sent := driver.got[0]

	// What the agent owns, as the hook left it.
	if sent.System != "set by the hook" {
		t.Errorf("System = %q, want the hook's", sent.System)
	}
	if len(sent.Tools) != 0 {
		t.Errorf("Tools = %d, want none — the hook cleared them", len(sent.Tools))
	}
	if len(sent.Messages) != 2 {
		t.Errorf("the endpoint saw %d messages, want 2 — the hook's addition was dropped",
			len(sent.Messages))
	}

	// What the hook asked for by option, which used to be dropped in silence.
	if sent.MaxTokens != 99 {
		t.Errorf("MaxTokens = %d, want the hook's 99", sent.MaxTokens)
	}
	if sent.Temperature == nil || *sent.Temperature != temp {
		t.Errorf("Temperature = %v, want the hook's %v", sent.Temperature, temp)
	}

	// And what nobody touched is still the client's.
	if sent.Effort != ai.EffortLow {
		t.Errorf("Effort = %q, want the client's — an untouched setting was overwritten", sent.Effort)
	}
}

// An agent with no tools offers none, whatever the client was built with. The
// toolset is the agent's, not a suggestion — and until Tools went out
// unconditionally there was no way for it to say so.
func TestAnAgentWithNoToolsOffersNone(t *testing.T) {
	stray := ai.ToolFunc("stray", "configured on the client",
		func(context.Context, struct{}) (string, error) { return "", nil })
	driver := &scripted{scripts: [][]ai.Delta{text("ok")}}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"},
		ai.WithTools(stray))

	a, err := agent.New(client)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	driver.mu.Lock()
	defer driver.mu.Unlock()
	if n := len(driver.got[0].Tools); n != 0 {
		t.Errorf("the endpoint was offered %d tools; the agent has none", n)
	}
}

// A span that never started is never ended. A hook that refuses stops the call
// before it is announced, so neither event fires — otherwise a consumer would
// see an inference end that it never saw begin, and no spinner survives that.
func TestARefusedCallEmitsNeitherHalfOfTheSpan(t *testing.T) {
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("never reached")}},
		ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client, agent.WithHooks(agent.Hook{
		PreInfer: func(context.Context, *agent.Inference) error {
			return errors.New("this conversation is too large to send")
		},
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err == nil {
		t.Fatal("the refusal never reached the caller")
	}

	for _, e := range events {
		switch e.(type) {
		case agent.MessageStart:
			t.Error("a refused call was announced as started")
		case agent.MessageEnd:
			t.Error("a call that never started was reported as ended")
		}
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", last.StopReason)
	}
	if last.Err == nil || !strings.Contains(last.Err.Error(), "too large") {
		t.Errorf("the outcome does not say what refused: %v", last.Err)
	}
}

// And when nothing refuses, the span is announced with the request that was
// actually sent — every hook's edit included.
func TestTheAnnouncedRequestIsTheOneThatWentOut(t *testing.T) {
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("fine")}},
		ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client,
		agent.WithSystem("base"),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, inf *agent.Inference) error {
				inf.System += " · edited"
				return nil
			},
		}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	var announced int
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok && v.Inference != nil {
			announced++
			if v.Inference.System != "base · edited" {
				t.Errorf("announced %q, want the edited prompt", v.Inference.System)
			}
		}
	}
	if announced != 1 {
		t.Errorf("announced %d inferences, want 1", announced)
	}
}

// PostInfer runs on what came back, and edits it before it enters the
// conversation — the seam a redaction or an annotation needs.
func TestPostInferEditsWhatEntersTheConversation(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("my number is 555-1234")}},
		agent.WithHooks(agent.Hook{
			PostInfer: func(_ context.Context, resp *ai.Response) error {
				for i, b := range resp.Content {
					if b.Type == ai.BlockText {
						resp.Content[i].Text = strings.ReplaceAll(b.Text, "555-1234", "[redacted]")
					}
				}
				return nil
			},
		}))

	events, err := collect(t, a, ai.UserMessage("what is it?"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	msgs := a.Messages()
	if got := msgs[len(msgs)-1].Text(); !strings.Contains(got, "[redacted]") {
		t.Errorf("the conversation kept %q — the edit did not land", got)
	}
	// And the event stream agrees: nobody saw the original.
	for _, e := range events {
		if v, ok := e.(agent.MessageAdded); ok && strings.Contains(v.Message.Text(), "555-1234") {
			t.Error("the unedited message reached a consumer")
		}
	}
}

// A PostInfer that objects ends the turn without another attempt: it made a
// decision, and repeating the call would be ignoring it.
func TestAPostInferRefusalIsNotRetried(t *testing.T) {
	driver := &scripted{scripts: [][]ai.Delta{text("one"), text("two"), text("three")}}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client, agent.WithHooks(agent.Hook{
		PostInfer: func(context.Context, *ai.Response) error {
			return errors.New("the answer failed review")
		},
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, err := collect(t, a, ai.UserMessage("go"))
	if err == nil {
		t.Fatal("the refusal never reached the caller")
	}

	driver.mu.Lock()
	calls := driver.calls
	driver.mu.Unlock()
	if calls != 1 {
		t.Errorf("the model was called %d times, want 1 — a refusal is not a retry", calls)
	}

	// The span is still closed: this call did happen.
	var ended int
	for _, e := range events {
		if v, ok := e.(agent.MessageEnd); ok {
			ended++
			if v.Err == nil {
				t.Error("the call reported success on a refused response")
			}
		}
	}
	if ended != 1 {
		t.Errorf("the call was closed %d times, want 1", ended)
	}
}

// Retries that all fail must surface the failure. Returning a zero message and
// a nil error would read as a successful empty answer, and the turn would end
// on it as if the model had simply said nothing.
func TestExhaustingRetriesReturnsTheFailure(t *testing.T) {
	overloaded := &ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}
	a := newAgent(t, &scripted{errs: []error{overloaded, overloaded, overloaded}})

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err == nil {
		t.Fatal("every attempt failed and the turn reported success")
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", last.StopReason)
	}
	// No assistant turn was produced, so none should have been appended.
	if got := a.Messages(); len(got) != 1 {
		t.Errorf("conversation holds %d messages, want just the input", len(got))
	}
}

// A call that failed still spent what it spent. Losing that hides real money.
func TestAFailedCallStillReportsItsCost(t *testing.T) {
	a := newAgent(t, &scripted{
		errs: []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}},
		scripts: [][]ai.Delta{
			{{Usage: &ai.Usage{Input: 120, Output: 4}}},
		},
	})

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err == nil {
		t.Fatal("expected the auth failure")
	}
	last := events[len(events)-1].(agent.TurnEnd)
	if got := last.Usage.TotalInput(); got != 120 {
		t.Errorf("input tokens = %d, want 120 — a failed call's cost was dropped", got)
	}
}

// A hook that refuses is refusing, whatever kind of error it picks. One that
// happens to look transient must still not be retried, and must not leave an
// MessageEnd with no MessageStart before it.
func TestARetryableLookingRefusalIsStillARefusal(t *testing.T) {
	driver := &scripted{scripts: [][]ai.Delta{text("one"), text("two"), text("three")}}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client, agent.WithHooks(agent.Hook{
		PreInfer: func(context.Context, *agent.Inference) error {
			return &ai.Error{Kind: ai.KindOverloaded, Message: "not while the queue is deep"}
		},
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, _ := collect(t, a, ai.UserMessage("go"))

	driver.mu.Lock()
	calls := driver.calls
	driver.mu.Unlock()
	if calls != 0 {
		t.Errorf("the model was called %d times after a refusal", calls)
	}

	var started, ended int
	for _, e := range events {
		switch e.(type) {
		case agent.MessageStart:
			started++
		case agent.MessageEnd:
			ended++
		}
	}
	if started != ended {
		t.Errorf("%d spans opened, %d closed — a span must not be half reported", started, ended)
	}
}

// stalling is an endpoint that says nothing until its context ends, which is
// the failure a stream watchdog exists for: it looks exactly like work.
type stalling struct {
	mu      sync.Mutex
	started chan struct{}
	calls   int
	// after is how many calls stall before one answers.
	after int
}

func (d *stalling) Name() string { return "stalling" }

func (d *stalling) Stream(ctx context.Context, req *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	d.mu.Unlock()

	return func(yield func(ai.Delta, error) bool) {
		if n >= d.after {
			for _, delta := range text("answered at last") {
				if !yield(delta, nil) {
					return
				}
			}
			return
		}
		if d.started != nil && n == 0 {
			close(d.started)
		}
		<-ctx.Done()
		yield(ai.Delta{}, ctx.Err())
	}
}

// A stream that goes quiet is a transient failure, and the one the client
// structurally cannot retry — ending the stall cancels the context ai.Retry
// would wait on — so it is what the agent's own budget is for.
func TestAStalledStreamIsRetried(t *testing.T) {
	a := newAgent(t, &stalling{after: 1},
		agent.WithStreamTimeout(20*time.Millisecond, 20*time.Millisecond),
		agent.WithRetry(2, 0))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("the stall was not recovered from: %v", err)
	}

	if n := steps(events); n != 1 {
		t.Errorf("steps = %d, want 1 — a retry is not a second step", n)
	}
	var attempts []int
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok {
			attempts = append(attempts, v.Attempt)
		}
	}
	if want := []int{1, 2}; !slices.Equal(attempts, want) {
		t.Errorf("attempts = %v, want %v", attempts, want)
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopEndTurn {
		t.Errorf("stop reason = %q, want the retry to have carried the turn", last.StopReason)
	}
}

// A watchdog nobody asked for does not fire: zero means no limit, and a slow
// endpoint is not an error.
func TestNoStreamTimeoutMeansNoWatchdog(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("fine")}},
		agent.WithStreamTimeout(0, 0))

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
}

// Interrupt between turns has nothing to end, and must not poison the next one.
func TestInterruptBetweenTurnsIsANoOp(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("fine")}})
	a.Interrupt()

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if last := events[len(events)-1].(agent.TurnEnd); last.StopReason != agent.StopEndTurn {
		t.Errorf("stop reason = %q, want end_turn", last.StopReason)
	}
}

// A model that ran out of output room did not finish answering, and saying
// end_turn would tell a caller the reply is whole when it is cut off.
func TestATruncatedAnswerSaysSo(t *testing.T) {
	cut := []ai.Delta{
		{Block: ai.TextBlock("the first half of a sentence that")},
		{EndBlock: true},
		{StopReason: ai.StopMaxTokens},
	}
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{cut}})

	events, err := collect(t, a, ai.UserMessage("write me an essay"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopMaxTokens {
		t.Errorf("stop reason = %q, want max_tokens", last.StopReason)
	}

	// What did arrive is still the conversation's: a truncated answer is an
	// answer to continue from, not one to throw away.
	msgs := a.Messages()
	if got := msgs[len(msgs)-1].Text(); !strings.Contains(got, "first half") {
		t.Errorf("the partial answer did not enter the conversation: %q", got)
	}
}

// The model's last message is on TurnEnd because the obvious way to find it is
// wrong: when a turn stops on a terminating tool, the conversation's last
// message is the tool results, not the model's.
func TestTurnEndCarriesTheModelsLastMessage(t *testing.T) {
	stop := agent.ToolFunc("finish", "Finish the task.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.Result{Content: ai.TextContent("done"), Terminate: true}, nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.TextBlock("wrapping up")},
			{EndBlock: true},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "finish", Input: `{}`})},
			{StopReason: ai.StopToolUse},
		},
	}}, agent.WithTools(stop))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	last := events[len(events)-1].(agent.TurnEnd)
	if last.StopReason != agent.StopTerminated {
		t.Fatalf("stop reason = %q, want terminated", last.StopReason)
	}
	if got := last.Message.Text(); got != "wrapping up" {
		t.Errorf("TurnEnd.Message = %q, want the model's", got)
	}
	if last.Message.Role != ai.RoleAssistant {
		t.Errorf("TurnEnd.Message role = %q, want assistant", last.Message.Role)
	}

	// And this is the trap the field exists to avoid: reaching for the
	// conversation's last message would have got the tool results.
	msgs := a.Messages()
	if got := msgs[len(msgs)-1].Role; got != ai.RoleUser {
		t.Errorf("the conversation ends on %q — the trap this field avoids has moved", got)
	}
}

// Collect folds an exchange for a caller that wants the answer rather than the
// progress — the shape a subagent behind a tool call needs.
func TestCollectFoldsAnExchangeIntoItsOutcome(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("the answer")}})

	out, err := outcome(t, a, ai.UserMessage("ask"))
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if out.StopReason != agent.StopEndTurn {
		t.Errorf("stop reason = %q, want end_turn", out.StopReason)
	}
	if got := out.Message.Text(); got != "the answer" {
		t.Errorf("Collect returned %q", got)
	}
}

// An exchange closes nothing, so a caller may take several in a row, and the
// turns are numbered in order.
func TestExchangesRunInSequence(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("first"), text("second")}})

	for i, want := range []string{"first", "second"} {
		out, err := outcome(t, a, ai.UserMessage("ask"))
		if err != nil {
			t.Fatalf("exchange %d: %v", i+1, err)
		}
		if got := out.Message.Text(); got != want {
			t.Errorf("exchange %d returned %q, want %q", i+1, got, want)
		}
		if out.Turn != i+1 {
			t.Errorf("exchange %d numbered %d", i+1, out.Turn)
		}
	}
}

// A failure ends its own exchange and nothing more: the next one runs.
func TestAFailedExchangeDoesNotPoisonTheNext(t *testing.T) {
	a := newAgent(t, &scripted{
		errs:    []error{&ai.Error{Kind: ai.KindAuth, Message: "no key"}},
		scripts: [][]ai.Delta{nil, text("second time")},
	})

	if _, err := outcome(t, a, ai.UserMessage("first")); err == nil {
		t.Fatal("the failure never reached the caller")
	}

	out, err := outcome(t, a, ai.UserMessage("second"))
	if err != nil {
		t.Fatalf("the second exchange failed too: %v", err)
	}
	if out.StopReason != agent.StopEndTurn {
		t.Errorf("stop reason = %q, want the agent to have carried on", out.StopReason)
	}
}

// One conversation advances one exchange at a time.
func TestAConcurrentExchangeIsRefused(t *testing.T) {
	release := make(chan struct{})
	blocking := agent.ToolFunc("wait", "Wait.",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			<-release
			return agent.TextResult("done"), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("c1", "wait", `{}`),
		text("finished"),
	}}, agent.WithTools(blocking))

	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		var once bool
		for e, err := range a.Run(context.Background(), ai.UserMessage("go")) {
			if _, ok := e.(agent.ToolStart); ok && !once {
				once = true
				close(started)
			}
			if err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	<-started

	_, err := outcome(t, a, ai.UserMessage("also"))
	if !errors.Is(err, agent.ErrBusy) {
		t.Errorf("a second exchange = %v, want ErrBusy", err)
	}

	close(release)
	if err := <-done; err != nil {
		t.Fatalf("the first exchange failed: %v", err)
	}
}

// Breaking out of the range ends the exchange: a consumer that stopped reading
// has stopped caring about this turn.
func TestBreakingOutOfTheRangeEndsTheExchange(t *testing.T) {
	driver := &scripted{scripts: [][]ai.Delta{
		toolCall("c1", "never", `{}`),
		text("unreachable"),
	}}
	a := newAgent(t, driver)

	for e, err := range a.Run(context.Background(), ai.UserMessage("go")) {
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if _, ok := e.(agent.MessageEnd); ok {
			break
		}
	}

	driver.mu.Lock()
	calls := driver.calls
	driver.mu.Unlock()
	if calls != 1 {
		t.Errorf("the model was called %d times after the consumer left, want 1", calls)
	}
}

// Repeating an exchange is a for loop the caller writes, which is what lets it
// decide the batching and what a failure means.
func TestSeveralExchangesAreACallersLoop(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("first"), text("second")}})

	var answers []string
	for _, batch := range [][]ai.Message{
		{ai.UserMessage("one")},
		{ai.UserMessage("two"), ai.UserMessage("and this too")},
	} {
		for e, err := range a.Run(context.Background(), batch...) {
			if err != nil {
				t.Fatalf("turn: %v", err)
			}
			if v, ok := e.(agent.TurnEnd); ok {
				answers = append(answers, v.Message.Text())
			}
		}
	}

	if want := []string{"first", "second"}; !slices.Equal(answers, want) {
		t.Errorf("answers = %v, want %v", answers, want)
	}
	// Five: one, then two-and-this-too as one exchange, each with its answer.
	if got := len(a.Messages()); got != 5 {
		t.Errorf("the conversation holds %d messages, want 5 — the second batch is one exchange", got)
	}
}

// A message added while an exchange is running joins it at the next step
// boundary, and is reported there — so the stream and the conversation agree
// about what the model was shown.
func TestAddedMessagesJoinTheExchangeAtAStepBoundary(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.TextResult("echoed"), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("c1", "echo", `{}`),
		text("done"),
	}}, agent.WithTools(echo))

	var announced []string
	for e, err := range a.Run(context.Background(), ai.UserMessage("first")) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		switch v := e.(type) {
		case agent.ToolEnd:
			// Added while the exchange is in flight, from the range body.
			a.AddMessages(ai.UserMessage("and also"), ai.UserMessage("and this"))
		case agent.MessageAdded:
			if v.Message.Role == ai.RoleUser {
				announced = append(announced, v.Message.Text())
			}
		}
	}

	// The tool-results message is a user turn too, hence the blank.
	want := []string{"first", "", "and also", "and this"}
	if !slices.Equal(announced, want) {
		t.Errorf("announced %q, want %q", announced, want)
	}

	// And the stream said everything the conversation holds.
	var folded int
	for _, m := range a.Messages() {
		_ = m
		folded++
	}
	if folded != 6 {
		t.Errorf("the conversation holds %d messages, want 6", folded)
	}
}

// A tool that takes a while shows its work: Report reaches the consumer as
// ToolUpdate, and the finished result still arrives on ToolEnd.
func TestAToolReportsWhileItWorks(t *testing.T) {
	slow := agent.ToolFunc("build", "Build the project.",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			agent.Report(ctx, agent.TextResult("compiling…"))
			agent.Report(ctx, agent.TextResult("linking…"))
			return agent.TextResult("built"), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("c1", "build", `{}`),
		text("done"),
	}}, agent.WithTools(slow))

	var partials []string
	var final string
	for e, err := range a.Run(context.Background(), ai.UserMessage("build it")) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		switch v := e.(type) {
		case agent.ToolUpdate:
			partials = append(partials, v.Partial.Text())
		case agent.ToolEnd:
			final = v.Result.Text()
		}
	}

	if want := []string{"compiling…", "linking…"}; !slices.Equal(partials, want) {
		t.Errorf("reported %q, want %q", partials, want)
	}
	if final != "built" {
		t.Errorf("ToolEnd result = %q, want the finished one", final)
	}
}

// A tool that reports where nobody installed a reporter — outside an exchange,
// or in a test — is not a tool that panics.
func TestReportOutsideAToolDoesNothing(t *testing.T) {
	agent.Report(context.Background(), agent.TextResult("into the void"))
}

// A tool declares its arguments once, as a Go type, and that type is both what
// the model is shown and what the arguments are decoded into. An argument the
// schema does not have is the model inventing something: it comes back as a
// tool error the model can read and correct, not as a field quietly dropped on
// the way to code that then acts on a request it never received.
func TestAToolRejectsAnArgumentItsSchemaDoesNotHave(t *testing.T) {
	type args struct {
		Path string `json:"path"`
	}
	var ran bool
	tool := agent.ToolFunc("read", "read a file", func(_ context.Context, a args) (agent.Result, error) {
		ran = true
		return agent.TextResult(a.Path), nil
	})

	if _, err := tool.Run(context.Background(), ai.ToolCall{
		Name: "read", Input: `{"path":"main.go","recursive":true}`,
	}); err == nil {
		t.Fatal("an unknown argument was accepted; the model would never learn it was ignored")
	}
	if ran {
		t.Error("the tool ran on arguments that did not match its schema")
	}

	// The same call without the invention is fine, and an absent optional
	// field keeps whatever the target already held.
	got, err := tool.Run(context.Background(), ai.ToolCall{Name: "read", Input: `{"path":"main.go"}`})
	if err != nil {
		t.Fatalf("valid arguments were rejected: %v", err)
	}
	if got.Text() != "main.go" {
		t.Errorf("got %q, want the decoded path", got.Text())
	}
}

// The two halves of a tool are the schema the model reads and the function
// that answers it, and ToolFunc derives the first from the second's argument
// type — so they cannot come to describe different things.
func TestAToolsSchemaComesFromItsArgumentType(t *testing.T) {
	type args struct {
		City string `json:"city" description:"which city"`
	}
	tool := agent.ToolFunc("weather", "look up the weather", func(context.Context, args) (agent.Result, error) {
		return agent.Result{}, nil
	})

	schema := tool.Schema()
	if schema.Name != "weather" || schema.Description != "look up the weather" {
		t.Errorf("Schema = %+v, want the name and description it was given", schema)
	}
	props, _ := schema.DefinitionMap()["properties"].(map[string]any)
	if _, ok := props["city"]; !ok {
		t.Errorf("properties = %v, want the field from the argument type", props)
	}
	if err := schema.Validate(`{"city":42}`); err == nil {
		t.Error("a wrongly typed argument passed validation")
	}
}

// A model that was stopped rather than one that chose to stop did not answer,
// whatever text arrived before it. Collapsing a refusal into end_turn tells a
// caller the reply is whole, which is the same lie max_tokens was fixed for.
func TestAStoppedModelSaysWhatStoppedIt(t *testing.T) {
	for _, tc := range []struct {
		wire ai.StopReason
		want agent.StopReason
	}{
		{ai.StopEndTurn, agent.StopEndTurn},
		{ai.StopMaxTokens, agent.StopMaxTokens},
		{ai.StopRefusal, agent.StopRefusal},
		{ai.StopSequence, agent.StopSequence},
	} {
		t.Run(string(tc.wire), func(t *testing.T) {
			d := &scripted{scripts: [][]ai.Delta{{
				{Block: ai.TextBlock("as far as it got")},
				{EndBlock: true},
				{StopReason: tc.wire},
			}}}
			a := newAgent(t, d)

			var end agent.TurnEnd
			for e, err := range a.Run(context.Background(), ai.UserMessage("go on")) {
				if err != nil {
					t.Fatal(err)
				}
				if v, ok := e.(agent.TurnEnd); ok {
					end = v
				}
			}
			if end.StopReason != tc.want {
				t.Errorf("StopReason = %q, want %q for a wire reason of %q",
					end.StopReason, tc.want, tc.wire)
			}
		})
	}
}

// Sequential is a promise that a tool never runs beside another, and it holds
// through a caller's own wrapper as long as the mark stays outermost — which
// is the rule Sequential documents, because no marker survives a decorator
// that embeds the Tool interface.
func TestASequentialToolRunsAloneThroughADecorator(t *testing.T) {
	var live, peak atomic.Int64
	slow := agent.ToolFunc("touch", "touch shared state",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			n := live.Add(1)
			for {
				old := peak.Load()
				if n <= old || peak.CompareAndSwap(old, n) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			live.Add(-1)
			return agent.TextResult("ok"), nil
		})

	d := &scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "touch", Input: "{}"})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "2", Name: "touch", Input: "{}"})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "3", Name: "touch", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("done"),
	}}
	a := newAgent(t, d, agent.WithTools(agent.Sequential(logged{slow})))

	for _, err := range a.Run(context.Background(), ai.UserMessage("touch it three times")) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := peak.Load(); got != 1 {
		t.Errorf("%d of them ran at once; Sequential promised 1", got)
	}
}

// logged is the kind of wrapper a caller writes: it passes everything through
// and adds nothing the agent knows about.
type logged struct{ agent.Tool }

// An endpoint that says how long to wait is the whole reason RetryAfter is on
// the error. Replaying immediately is what rate limiting exists to punish, and
// a loop that counts attempts without ever waiting does exactly that.
func TestARateLimitIsWaitedOutForAsLongAsItAsked(t *testing.T) {
	const askedFor = 60 * time.Millisecond
	a := newAgent(t, &scripted{
		errs: []error{&ai.Error{
			Kind: ai.KindRateLimit, Message: "slow down", RetryAfter: askedFor,
		}},
		scripts: [][]ai.Delta{nil, text("thank you for waiting")},
		// A backoff of zero, so anything waited for came from the endpoint.
	}, agent.WithRetry(2, 0))

	start := time.Now()
	if _, err := outcome(t, a, ai.UserMessage("hi")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if waited := time.Since(start); waited < askedFor {
		t.Errorf("retried after %v; the endpoint asked for %v", waited, askedFor)
	}
}

// Retry is off unless asked for, because the client already has ai.Retry and
// two budgets multiply: three attempts here on a client wrapping ai.Retry(3)
// is nine model calls for one step.
func TestAnAgentDoesNotRetryUnlessAsked(t *testing.T) {
	d := &scripted{
		errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		scripts: [][]ai.Delta{nil, text("would have been the retry")},
	}
	a := newAgent(t, d)

	if _, err := outcome(t, a, ai.UserMessage("hi")); err == nil {
		t.Fatal("the failure never reached the caller")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.calls != 1 {
		t.Errorf("the driver was called %d times, want 1 — retry is the client's", d.calls)
	}
}

// SetTools, SetSystem and AddHook all promise the same thing: the change lands
// on the next inference, not mid-stream. A turn holds several inferences, so
// "next" is a real moment inside one — and nothing pinned it.
func TestTheAgentIsReconfiguredBetweenInferences(t *testing.T) {
	d := &scripted{scripts: [][]ai.Delta{
		toolCall("1", "before", "{}"),
		text("done"),
	}}
	before := agent.ToolFunc("before", "the tool it starts with",
		func(context.Context, struct{}) (agent.Result, error) { return agent.TextResult("ok"), nil })
	after := agent.ToolFunc("after", "the tool it is given mid-turn",
		func(context.Context, struct{}) (agent.Result, error) { return agent.TextResult("ok"), nil })

	a := newAgent(t, d, agent.WithSystem("first prompt"), agent.WithTools(before))

	var hookSaw []string
	for e, err := range a.Run(context.Background(), ai.UserMessage("go")) {
		if err != nil {
			t.Fatal(err)
		}
		// Between the first inference and the second, change everything.
		if _, ok := e.(agent.ToolEnd); ok {
			a.SetTools(after)
			a.SetSystem("second prompt")
			a.AddHooks(agent.Hook{
				PreInfer: func(_ context.Context, inf *agent.Inference) error {
					hookSaw = append(hookSaw, inf.System)
					return nil
				},
			})
		}
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.got) != 2 {
		t.Fatalf("the driver saw %d calls, want 2", len(d.got))
	}
	if d.got[0].System != "first prompt" {
		t.Errorf("call 1 system = %q, want the original", d.got[0].System)
	}
	if d.got[1].System != "second prompt" {
		t.Errorf("call 2 system = %q, want the replacement", d.got[1].System)
	}
	if n := len(d.got[0].Tools); n != 1 || d.got[0].Tools[0].Schema.Name != "before" {
		t.Errorf("call 1 offered %v, want the original toolset", toolNames(d.got[0].Tools))
	}
	if n := len(d.got[1].Tools); n != 1 || d.got[1].Tools[0].Schema.Name != "after" {
		t.Errorf("call 2 offered %v, want the replacement", toolNames(d.got[1].Tools))
	}
	// The hook was added after the first call, so it saw only the second.
	if len(hookSaw) != 1 || hookSaw[0] != "second prompt" {
		t.Errorf("the added hook ran %d times %v, want once on the second call", len(hookSaw), hookSaw)
	}
}

func toolNames(tools []ai.Tool) []string {
	out := make([]string, len(tools))
	for i, t := range tools {
		out[i] = t.Schema.Name
	}
	return out
}

// FromAI is the bridge that lets a tool written against pkg/ai run in an
// agent without being rewritten. Its schema and its answer both have to
// survive the crossing, and a failing one has to fail the agent's way — as a
// result the model can read, not as a turn that dies.
func TestAToolWrittenForTheClientRunsInAnAgent(t *testing.T) {
	type args struct {
		City string `json:"city" description:"which city"`
	}
	plain := ai.ToolFunc("weather", "Look up the weather.",
		func(_ context.Context, a args) (string, error) {
			if a.City == "" {
				return "", errors.New("no city given")
			}
			return "mild in " + a.City, nil
		})

	lifted := agent.FromAI(plain)
	if got := lifted.Schema(); got.Name != "weather" || got.Description != "Look up the weather." {
		t.Errorf("Schema = %+v, want the ai.Tool's", got)
	}
	if _, ok := lifted.Schema().DefinitionMap()["properties"]; !ok {
		t.Error("the derived schema did not cross over")
	}

	got, err := lifted.Run(context.Background(),
		ai.ToolCall{ID: "1", Name: "weather", Input: `{"city":"Delhi"}`})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Text() != "mild in Delhi" {
		t.Errorf("Run = %q, want what the ai.Tool returned", got.Text())
	}

	// A tool offered with nothing to run it is a configuration mistake, and
	// says so rather than panicking.
	empty := agent.FromAI(ai.Tool{Schema: ai.Schema{Name: "hollow"}})
	if _, err := empty.Run(context.Background(), ai.ToolCall{Name: "hollow"}); err == nil {
		t.Error("a tool with no Run was called without complaint")
	}
}

// A tool is the caller's code, but it runs on a goroutine this package
// created — the one place a panic cannot be recovered by whoever wrote it.
// Unrecovered it takes the whole process down mid-conversation. A failing tool
// already has a way to fail, so a panic becomes that: the model is told, and
// the turn carries on.
func TestAPanickingToolDoesNotTakeTheProcessWithIt(t *testing.T) {
	boom := agent.ToolFunc("boom", "panics",
		func(context.Context, struct{}) (agent.Result, error) {
			var m map[string]int
			m["nil map write"] = 1 // panics
			return agent.TextResult("unreachable"), nil
		})
	fine := agent.ToolFunc("fine", "works",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("still here"), nil
		})

	d := &scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "boom", Input: "{}"})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "2", Name: "fine", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("carried on"),
	}}
	a := newAgent(t, d, agent.WithTools(boom, fine))

	var failed *agent.PanicError
	ended := map[string]bool{}
	for e, err := range a.Run(context.Background(), ai.UserMessage("run both")) {
		if err != nil {
			t.Fatalf("the turn died: %v", err)
		}
		if v, ok := e.(agent.ToolEnd); ok {
			ended[v.Name] = true
			if v.Name == "boom" {
				if !errors.As(v.Err, &failed) {
					t.Fatalf("boom ended with %v, want a *agent.PanicError", v.Err)
				}
			}
		}
	}

	if !ended["boom"] || !ended["fine"] {
		t.Errorf("tools that ended: %v, want both — one panic stalled the batch", ended)
	}
	if failed == nil {
		t.Fatal("the panic was never reported")
	}
	if len(failed.Stack) == 0 {
		t.Error("the stack was not kept; a recovered panic with no stack is undebuggable")
	}
	// The model is told one line, not a stack trace.
	if got := agent.ResultText(agent.Result{}, failed); strings.Contains(got, "goroutine") {
		t.Errorf("the model was told %q — that is a stack trace", got)
	}

	// And the conversation went on: the model answered after the tool results.
	last := a.Messages()[len(a.Messages())-1]
	if last.Text() != "carried on" {
		t.Errorf("the conversation ended at %q, want the model's answer", last.Text())
	}
}

// The mutex on an Agent is a promise: a caller may read and change the agent
// from another goroutine while a turn is running. Nothing tested that promise,
// so nothing would have caught a setter added without the lock. Run under
// -race, which is where this test does its work.
func TestAnAgentIsSafeToTouchWhileItRuns(t *testing.T) {
	slow := agent.ToolFunc("slow", "takes long enough to be raced",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			time.Sleep(30 * time.Millisecond)
			return agent.TextResult("done"), nil
		})
	d := &scripted{scripts: [][]ai.Delta{
		toolCall("1", "slow", "{}"),
		toolCall("2", "slow", "{}"),
		text("finished"),
	}}
	a := newAgent(t, d, agent.WithTools(slow))

	running := make(chan struct{})
	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Everything a caller is told it may do from elsewhere.
	for _, touch := range []func(){
		func() { a.SetSystem("changed") },
		func() { a.SetTools(slow) },
		func() { a.AddHooks(agent.Hook{}) },
		func() { a.AddMessages(ai.UserMessage("from outside")) },
		func() { _ = a.Messages() },
		func() { _ = a.Tools() },
		func() { _ = a.System() },
		func() { _ = a.String() },
	} {
		wg.Add(1)
		go func(f func()) {
			defer wg.Done()
			<-running
			for {
				select {
				case <-stop:
					return
				default:
					f()
				}
			}
		}(touch)
	}

	var turnEnded bool
	for e, err := range a.Run(context.Background(), ai.UserMessage("go")) {
		if err != nil {
			t.Fatalf("the turn failed: %v", err)
		}
		if _, ok := e.(agent.TurnStart); ok {
			close(running) // let the other goroutines loose mid-turn
		}
		if _, ok := e.(agent.TurnEnd); ok {
			turnEnded = true
		}
	}
	close(stop)
	wg.Wait()

	if !turnEnded {
		t.Error("the turn never finished")
	}
}

// A second Run while one is in flight is refused, not queued: two of them
// appending to one conversation would interleave into a history neither asked
// for.
func TestASecondRunWhileOneIsInFlightIsRefused(t *testing.T) {
	held := make(chan struct{})
	blocking := agent.ToolFunc("hold", "blocks until released",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			<-held
			return agent.TextResult("released"), nil
		})
	d := &scripted{scripts: [][]ai.Delta{toolCall("1", "hold", "{}"), text("done")}}
	a := newAgent(t, d, agent.WithTools(blocking))

	inTool := make(chan struct{})
	var second error
	go func() {
		<-inTool
		for _, err := range a.Run(context.Background(), ai.UserMessage("me too")) {
			second = err
		}
		close(held)
	}()

	for e, err := range a.Run(context.Background(), ai.UserMessage("first")) {
		if err != nil {
			t.Fatalf("the first turn failed: %v", err)
		}
		if _, ok := e.(agent.ToolStart); ok {
			close(inTool)
		}
	}

	if !errors.Is(second, agent.ErrBusy) {
		t.Errorf("the second Run returned %v, want ErrBusy", second)
	}
}

// Every ai.StopReason has to mean something here, and the compiler cannot say
// so: the loop translates them with a switch whose default is end_turn, so one
// added to pkg/ai and forgotten here becomes "the model answered normally".
// That has happened once already — refusal read as a complete answer.
//
// So the list is read from pkg/ai itself rather than written down twice. A new
// constant fails this test until somebody decides what a turn ending that way
// should be called.
func TestEveryStopReasonIsTranslatedDeliberately(t *testing.T) {
	want := map[ai.StopReason]agent.StopReason{
		ai.StopEndTurn:   agent.StopEndTurn,
		ai.StopMaxTokens: agent.StopMaxTokens,
		ai.StopRefusal:   agent.StopRefusal,
		ai.StopSequence:  agent.StopSequence,
		// A turn that ended in one of these never reaches the translation: the
		// call failed, and the loop reports the failure instead.
		ai.StopToolUse: agent.StopEndTurn,
		ai.StopError:   agent.StopEndTurn,
		ai.StopAborted: agent.StopEndTurn,
	}

	for _, reason := range stopReasonsDeclaredIn(t, "../ai/response.go") {
		expected, decided := want[reason]
		if !decided {
			t.Errorf("ai.StopReason %q is new: decide what a turn ending that way "+
				"is called, add it to endedBecause and to this table", reason)
			continue
		}
		t.Run(string(reason), func(t *testing.T) {
			d := &scripted{scripts: [][]ai.Delta{{
				{Block: ai.TextBlock("as far as it got")},
				{EndBlock: true},
				{StopReason: reason},
			}}}
			out, err := outcome(t, newAgent(t, d), ai.UserMessage("go"))
			if err != nil {
				t.Fatal(err)
			}
			if out.StopReason != expected {
				t.Errorf("a wire reason of %q became %q, want %q", reason, out.StopReason, expected)
			}
		})
	}
}

// stopReasonsDeclaredIn reads the StopReason constants out of a source file, so
// the set this test checks is the set that exists rather than one remembered.
func stopReasonsDeclaredIn(t *testing.T, path string) []ai.StopReason {
	t.Helper()

	file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}

	var out []ai.StopReason
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.CONST {
			continue
		}
		for _, spec := range gen.Specs {
			v, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			if id, ok := v.Type.(*ast.Ident); !ok || id.Name != "StopReason" {
				continue
			}
			for _, val := range v.Values {
				lit, ok := val.(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					continue
				}
				out = append(out, ai.StopReason(strings.Trim(lit.Value, `"`)))
			}
		}
	}
	if len(out) == 0 {
		t.Fatalf("no StopReason constants found in %s — the parser or the file moved", path)
	}
	return out
}
