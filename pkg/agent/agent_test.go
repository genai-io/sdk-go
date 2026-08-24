package agent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
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
func collect(t *testing.T, a *agent.Agent, msgs ...ai.Message) ([]agent.Event, error) {
	t.Helper()
	for _, m := range msgs {
		a.In() <- m
	}
	close(a.In())

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var events []agent.Event
	for e := range a.Out() {
		// Run's own boundaries are not part of an exchange's sequence.
		switch e.(type) {
		case agent.RunStart, agent.RunEnd:
			continue
		}
		events = append(events, e)
	}
	<-done

	// A failed exchange is reported on the stream, not returned by Run.
	for _, e := range events {
		if v, ok := e.(agent.TurnEnd); ok && v.Err != nil {
			return events, v.Err
		}
	}
	return events, nil
}

// steps counts what the loop counts: an inference that is not a retry.
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
		}, _ func(agent.Result)) (agent.Result, error) {
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
		func(_ context.Context, _ struct{}, _ func(agent.Result)) (agent.Result, error) {
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
	})

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
		}, _ func(agent.Result)) (agent.Result, error) {
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
		func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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
		func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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
		}, _ func(agent.Result)) (agent.Result, error) {
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
		func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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
		func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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

// Run closes out when it stops, so a caller ranging over it terminates without
// being told to — and RunStart / RunEnd bracket everything in between.
func TestRunBracketsAndClosesTheEventChannel(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("one"), text("two")}})

	go func() {
		a.In() <- ai.UserMessage("first")
		for len(a.Messages()) < 2 {
			time.Sleep(time.Millisecond)
		}
		a.In() <- ai.UserMessage("second")
		// The failed exchange appends only the user turn, so the second one
		// brings the total to three.
		for len(a.Messages()) < 3 {
			time.Sleep(time.Millisecond)
		}
		close(a.In())
	}()

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var seen []string
	for e := range a.Out() { // terminates because Run closes out
		seen = append(seen, strings.TrimPrefix(fmt.Sprintf("%T", e), "agent."))
	}
	if err := <-done; err != nil {
		t.Fatalf("Run: %v", err)
	}

	if seen[0] != "RunStart" {
		t.Errorf("first event = %q, want RunStart", seen[0])
	}
	if last := seen[len(seen)-1]; last != "RunEnd" {
		t.Errorf("last event = %q, want RunEnd", last)
	}
	turns := 0
	for _, n := range seen {
		if n == "TurnStart" {
			turns++
		}
	}
	if turns != 2 {
		t.Errorf("turns = %d, want 2", turns)
	}
}

// Messages that arrive while the agent works join one exchange, not several:
// someone who typed three lines meant them together.
func TestMessagesQueuedDuringAnExchangeArriveAsOneBatch(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("1"), text("2")}})

	a.In() <- ai.UserMessage("first")

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var prompts []string
	var turns int
	queued, closed := false, false
	for e := range a.Out() {
		switch v := e.(type) {
		case agent.MessageAdded:
			if v.Message.Role == ai.RoleUser {
				prompts = append(prompts, v.Message.Text())
			}
		case agent.TurnStart:
			turns++
		case agent.MessageStart:
			if !queued {
				queued = true
				a.In() <- ai.UserMessage("and also")
				a.In() <- ai.UserMessage("and this")
			}
		case agent.TurnEnd:
			if queued && !closed {
				closed = true
				close(a.In())
			}
		}
	}
	<-done

	if want := []string{"first", "and also", "and this"}; !slices.Equal(prompts, want) {
		t.Errorf("prompts = %v, want %v", prompts, want)
	}
	if turns != 2 {
		t.Errorf("turns = %d, want 2 — the two queued lines belong to one exchange", turns)
	}
}

// A failed exchange is reported and the run carries on. The failure reaches
// the caller as TurnEnd — with why it stopped and what it cost — not as a
// second, weaker event saying the same thing.
func TestAFailedExchangeIsReportedAndTheRunCarriesOn(t *testing.T) {
	a := newAgent(t, &scripted{
		errs:    []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}},
		scripts: [][]ai.Delta{nil, text("second is fine")},
	})

	go func() {
		a.In() <- ai.UserMessage("first")
		for len(a.Messages()) < 1 {
			time.Sleep(time.Millisecond)
		}
		a.In() <- ai.UserMessage("second")
		// The failed exchange appends only the user turn, so the second one
		// brings the total to three.
		for len(a.Messages()) < 3 {
			time.Sleep(time.Millisecond)
		}
		close(a.In())
	}()

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var outcomes []agent.TurnEnd
	for e := range a.Out() {
		if v, ok := e.(agent.TurnEnd); ok {
			outcomes = append(outcomes, v)
		}
	}
	if err := <-done; err != nil {
		t.Fatalf("a failed exchange ended the run: %v", err)
	}

	if len(outcomes) != 2 {
		t.Fatalf("saw %d turns, want 2 — the run stopped at the failure", len(outcomes))
	}
	if outcomes[0].Err == nil || outcomes[0].StopReason != agent.StopError {
		t.Errorf("first turn = %+v, want the failure on its outcome", outcomes[0])
	}
	if outcomes[1].Err != nil {
		t.Errorf("second turn failed too: %v", outcomes[1].Err)
	}
}

// Several hooks run in order, and the first refusal is final: a later one must
// not be able to allow what an earlier one denied, or adding a hook could
// weaken the gate.
func TestTheFirstRefusalIsFinal(t *testing.T) {
	tool := agent.ToolFunc("rm", "Delete things.",
		func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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
		}, _ func(agent.Result)) (agent.Result, error) {
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

// Every way a turn ends says why. A cancelled one used to report an empty stop
// reason, because the branch that set StopCanceled sat after the return that
// made it unreachable.
func TestACancelledTurnSaysItWasCancelled(t *testing.T) {
	gate := make(chan struct{})
	slow := agent.ToolFunc("slow", "Block until released.",
		func(ctx context.Context, _ struct{}, _ func(agent.Result)) (agent.Result, error) {
			select {
			case <-gate:
			case <-ctx.Done():
			}
			return agent.TextResult("done"), nil
		})

	a := newAgent(t, &scripted{scripts: [][]ai.Delta{
		toolCall("c1", "slow", `{}`),
		text("never asked for"),
	}}, agent.WithTools(slow))

	ctx, stop := context.WithCancel(context.Background())
	go func() {
		a.In() <- ai.UserMessage("go")
		for _, ok := <-a.Out(); ok; _, ok = <-a.Out() {
		}
	}()

	done := make(chan error, 1)
	go func() { done <- a.Run(ctx) }()

	time.Sleep(30 * time.Millisecond)
	stop()
	close(gate)

	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Errorf("Run = %v, want context.Canceled", err)
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
				func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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
				func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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

func TestTurnsAreNumberedInOrder(t *testing.T) {
	a := newAgent(t, &scripted{scripts: [][]ai.Delta{text("one"), text("two")}})

	go func() {
		a.In() <- ai.UserMessage("first")
		for len(a.Messages()) < 2 {
			time.Sleep(time.Millisecond)
		}
		a.In() <- ai.UserMessage("second")
		for len(a.Messages()) < 4 {
			time.Sleep(time.Millisecond)
		}
		close(a.In())
	}()

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var numbers []int
	for e := range a.Out() {
		if v, ok := e.(agent.TurnStart); ok {
			numbers = append(numbers, v.Turn)
		}
	}
	if err := <-done; err != nil {
		t.Fatalf("Run: %v", err)
	}

	if want := []int{1, 2}; !slices.Equal(numbers, want) {
		t.Errorf("turn numbers = %v, want %v", numbers, want)
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
	})

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
		func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
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
			PreInfer: func(_ context.Context, req *ai.Request) error {
				sizes = append(sizes, len(req.Messages))
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
		agent.ToolFunc("keep", "Stays.", func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		}),
		agent.ToolFunc("hide", "Hidden for one call.", func(context.Context, struct{}, func(agent.Result)) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		}),
	}

	var sent *ai.Request
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("fine")}}, ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client,
		agent.WithSystem("the agent's own prompt"),
		agent.WithTools(tools...),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, req *ai.Request) error {
				req.System += "\n\nand a line true only right now"
				req.Tools = req.Tools[:1] // hide the second for this call
				req.Messages = append(req.Messages, ai.UserMessage("injected"))
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
			sent = v.Request
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
		agent.Hook{PreInfer: func(context.Context, *ai.Request) error {
			return errors.New("the context is too large to send")
		}},
		agent.Hook{PreInfer: func(context.Context, *ai.Request) error {
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

	var seen *ai.Request
	a, err := agent.New(client,
		agent.WithSystem("mine"),
		agent.WithMessages([]ai.Message{ai.UserMessage("earlier")}),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, req *ai.Request) error {
				seen = req
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

	var sent *ai.Request
	for _, e := range events {
		if v, ok := e.(agent.MessageStart); ok {
			sent = v.Request
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
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, req *ai.Request) error {
				attempts++
				// An edit that would compound if the request were reused.
				req.System += " · attempt"
				prompts = append(prompts, req.System)
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
			agent.Hook{PreInfer: func(_ context.Context, req *ai.Request) error {
				req.System += " two"
				return nil
			}},
			agent.Hook{PreInfer: func(_ context.Context, req *ai.Request) error {
				if req.System != "one two" {
					t.Errorf("the second hook saw %q, want the first hook's edit", req.System)
				}
				req.System += " three"
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
		if v, ok := e.(agent.MessageStart); ok && v.Request.System != "one two three" {
			t.Errorf("sent %q, want every hook's edit", v.Request.System)
		}
	}
}

// A hook edits what the agent contributes — prompt, conversation, toolset —
// and all three reach the endpoint. It does not edit the client underneath:
// temperature, token ceilings and the rest were set when the client was built,
// and writing them here does nothing. Both halves are pinned, because a seam
// that silently ignores what is written to it is worse than one that never
// offered the field.
func TestAPreInferHookEditsTheAgentsHalfOfTheRequest(t *testing.T) {
	weather := agent.ToolFunc("weather", "Look up the weather.",
		func(_ context.Context, _ struct{}, _ func(agent.Result)) (agent.Result, error) {
			return agent.TextResult("fine"), nil
		})

	driver := &scripted{scripts: [][]ai.Delta{text("ok")}}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"},
		ai.WithMaxTokens(4096)) // the client's setting, not the agent's

	temp := 0.25
	a, err := agent.New(client,
		agent.WithSystem("original"),
		agent.WithTools(weather),
		agent.WithHooks(agent.Hook{
			PreInfer: func(_ context.Context, req *ai.Request) error {
				req.System = "set by the hook"
				req.Tools = nil
				req.Messages = append(req.Messages, ai.UserMessage("and this"))

				// The client's business, not the agent's: written here, ignored.
				req.MaxTokens = 99
				req.Temperature = &temp
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

	// What the client owns, as it was built.
	if sent.MaxTokens != 4096 {
		t.Errorf("MaxTokens = %d, want the client's 4096 — a hook reconfigured it", sent.MaxTokens)
	}
	if sent.Temperature != nil {
		t.Errorf("Temperature = %v, want unset — a hook reconfigured it", *sent.Temperature)
	}
}

// A span that never started is never ended. A hook that refuses stops the call
// before it is announced, so neither event fires — otherwise a consumer would
// see an inference end that it never saw begin, and no spinner survives that.
func TestARefusedCallEmitsNeitherHalfOfTheSpan(t *testing.T) {
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{text("never reached")}},
		ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client, agent.WithHooks(agent.Hook{
		PreInfer: func(context.Context, *ai.Request) error {
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
			PreInfer: func(_ context.Context, req *ai.Request) error {
				req.System += " · edited"
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
		if v, ok := e.(agent.MessageStart); ok && v.Request != nil {
			announced++
			if v.Request.System != "base · edited" {
				t.Errorf("announced %q, want the edited prompt", v.Request.System)
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
		PreInfer: func(context.Context, *ai.Request) error {
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

// A stream that goes quiet is a transient failure, so the attempt that hit it
// is retried rather than failing the turn.
func TestAStalledStreamIsRetried(t *testing.T) {
	a := newAgent(t, &stalling{after: 1},
		agent.WithStreamTimeout(20*time.Millisecond, 20*time.Millisecond))

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

// Interrupt ends the exchange in flight and leaves the run alive, which is
// what a user pressing escape asks for. Cancelling Run's context is the other
// thing, and ends everything.
func TestInterruptEndsTheTurnAndNotTheRun(t *testing.T) {
	driver := &stalling{after: 1, started: make(chan struct{})}
	a := newAgent(t, driver, agent.WithStreamTimeout(0, 0))

	go func() {
		a.In() <- ai.UserMessage("first")
		<-driver.started
		a.Interrupt()
	}()

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var turns []agent.TurnEnd
	go func() {
		// The run keeps going, so a second exchange has to be able to land.
		for range time.Tick(time.Millisecond) {
			if len(a.Messages()) > 0 {
				break
			}
		}
		a.In() <- ai.UserMessage("second")
	}()

	for e := range a.Out() {
		if v, ok := e.(agent.TurnEnd); ok {
			turns = append(turns, v)
			if len(turns) == 2 {
				close(a.In())
			}
		}
	}
	if err := <-done; err != nil {
		t.Fatalf("Interrupt ended the run: %v", err)
	}

	if len(turns) != 2 {
		t.Fatalf("saw %d turns, want 2 — the run stopped at the interrupt", len(turns))
	}
	if turns[0].StopReason != agent.StopCanceled {
		t.Errorf("first turn = %q, want canceled", turns[0].StopReason)
	}
	if turns[1].StopReason != agent.StopEndTurn {
		t.Errorf("second turn = %q, want the run to have carried on", turns[1].StopReason)
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

// Interrupting a turn abandons the work, not the reporting. The run's reader
// is still there, so every span the turn opened is closed and everything that
// entered the conversation was announced — otherwise a session restored from
// the stream would disagree with the agent that produced it.
func TestAnInterruptedTurnStillClosesWhatItOpened(t *testing.T) {
	driver := &stalling{after: 1, started: make(chan struct{})}
	a := newAgent(t, driver, agent.WithStreamTimeout(0, 0))

	go func() {
		a.In() <- ai.UserMessage("first")
		<-driver.started
		a.Interrupt()
	}()

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background()) }()

	var started, ended, added int
	for e := range a.Out() {
		switch e.(type) {
		case agent.MessageStart:
			started++
		case agent.MessageEnd:
			ended++
		case agent.MessageAdded:
			added++
		case agent.TurnEnd:
			close(a.In())
		}
	}
	if err := <-done; err != nil {
		t.Fatalf("Run: %v", err)
	}

	if started != ended {
		t.Errorf("%d spans opened, %d closed — an interrupt left one hanging", started, ended)
	}
	if got := len(a.Messages()); added != got {
		t.Errorf("%d messages announced, %d in the conversation — the stream and the agent disagree", added, got)
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
