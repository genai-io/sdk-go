package agent_test

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// The plain exchange from the design's first sequence diagram.
func TestTurnEmitsTheDocumentedTextSequence(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("hello there")}})

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

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
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

// The same claim without the concurrency: every exit sets a stop reason.
func TestEveryOutcomeSaysWhyItStopped(t *testing.T) {
	for _, tc := range []struct {
		name   string
		build  func(*testing.T) *agent.Agent
		reason agent.StopReason
		// also is what this ending claims beyond its name: that the budget was
		// spent exactly, that the loop stopped where it said it would.
		also func(*testing.T, []agent.Event)
	}{
		{"the model answered", func(t *testing.T) *agent.Agent {
			return newAgent(t, &scripted{Scripts: [][]ai.Delta{text("done")}})
		}, agent.StopEndTurn, nil},

		{"the step budget ran out", func(t *testing.T) *agent.Agent {
			again := agent.ToolFunc("again", "Ask again.",
				func(context.Context, struct{}) (agent.Result, error) {
					return agent.TextResult("again"), nil
				})
			scripts := make([][]ai.Delta, 8)
			for i := range scripts {
				scripts[i] = toolCall(fmt.Sprintf("c%d", i), "again", `{}`)
			}
			client := ai.NewClientWithDriver(&scripted{Scripts: scripts}, ai.Model{ID: "stub", API: "stub"})
			a, err := agent.New(client, agent.WithTools(again), agent.WithMaxSteps(2))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			return a
		}, agent.StopMaxSteps, func(t *testing.T, events []agent.Event) {
			if n := steps(events); n != 2 {
				t.Errorf("steps = %d, want the 2 it was given", n)
			}
		}},

		{"a tool asked to stop", func(t *testing.T) *agent.Agent {
			done := agent.ToolFunc("finish", "Report completion.",
				func(context.Context, struct{}) (agent.Result, error) {
					return agent.Result{Content: ai.TextContent("finished"), Terminate: true}, nil
				})
			return newAgent(t, &scripted{Scripts: [][]ai.Delta{toolCall("c1", "finish", `{}`)}},
				agent.WithTools(done))
		}, agent.StopTerminated, func(t *testing.T, events []agent.Event) {
			if n := steps(events); n != 1 {
				t.Errorf("inferences = %d, want 1 — the follow-up call is skipped", n)
			}
		}},

		{"the model call failed", func(t *testing.T) *agent.Agent {
			return newAgent(t, &scripted{Errs: []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}}})
		}, agent.StopError, func(t *testing.T, events []agent.Event) {
			last := events[len(events)-1].(agent.TurnEnd)
			if !ai.IsAuth(last.Err) {
				t.Errorf("the outcome carries %v, want the auth failure", last.Err)
			}
			if n := attempts(events); n != 1 {
				t.Errorf("attempts = %d, want 1 — an auth failure is not retryable", n)
			}
		}},
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
			if tc.also != nil {
				tc.also(t, events)
			}
		})
	}
}

// A failed attempt still spent whatever it spent. Folding the retry into the
// step made that visible: usage now accumulates per attempt, not per step.
func TestAFailedAttemptStillCountsWhatItCost(t *testing.T) {
	a := newAgent(t, &scripted{
		Errs: []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		Scripts: [][]ai.Delta{
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

// The model's last message is on TurnEnd because the obvious way to find it is
// wrong: when a turn stops on a terminating tool, the conversation's last
// message is the tool results, not the model's.
func TestTurnEndCarriesTheModelsLastMessage(t *testing.T) {
	stop := agent.ToolFunc("finish", "Finish the task.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.Result{Content: ai.TextContent("done"), Terminate: true}, nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
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

// A model that ran out of output room did not finish answering, and saying
// end_turn would tell a caller the reply is whole when it is cut off.
func TestATruncatedAnswerSaysSo(t *testing.T) {
	cut := []ai.Delta{
		{Block: ai.TextBlock("the first half of a sentence that")},
		{EndBlock: true},
		{StopReason: ai.StopMaxTokens},
	}
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{cut}})

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
			d := &scripted{Scripts: [][]ai.Delta{{
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

// And when nothing refuses, the span is announced with the request that was
// actually sent — every hook's edit included.
func TestTheAnnouncedRequestIsTheOneThatWentOut(t *testing.T) {
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{text("fine")}},
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

// The fragment a consumer almost always wants, without reaching through two
// levels of the event that carries it.
func TestAMessageUpdateSaysWhatFragmentItCarries(t *testing.T) {
	for _, tc := range []struct {
		name           string
		delta          ai.Event
		text, thinking string
	}{
		{"text", ai.Event{Type: ai.EventBlockDelta, Block: ai.Block{Type: ai.BlockText, Text: "hello"}}, "hello", ""},
		{"thinking", ai.Event{Type: ai.EventBlockDelta, Block: ai.Block{Type: ai.BlockThinking, Text: "hmm"}}, "", "hmm"},
		{"a block opening", ai.Event{Type: ai.EventBlockStart, Block: ai.Block{Type: ai.BlockText, Text: "no"}}, "", ""},
		{"a tool call taking shape", ai.Event{Type: ai.EventBlockDelta, Block: ai.Block{Type: ai.BlockToolCall}}, "", ""},
		{"the end", ai.Event{Type: ai.EventDone}, "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u := agent.MessageUpdate{Delta: tc.delta}
			if got := u.Text(); got != tc.text {
				t.Errorf("Text() = %q, want %q", got, tc.text)
			}
			if got := u.Thinking(); got != tc.thinking {
				t.Errorf("Thinking() = %q, want %q", got, tc.thinking)
			}
		})
	}
}

// A turn's own failure travels on TurnEnd and nowhere else. The iterator's
// error is for what happens outside a turn — ErrBusy — so a caller rendering
// both does not report one failure twice.
func TestAFailedTurnIsReportedOnceAndOnTheStream(t *testing.T) {
	a := newAgent(t, &scripted{Errs: []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}}})

	var events []agent.Event
	for e, err := range a.Run(context.Background(), ai.UserMessage("go")) {
		if err != nil {
			t.Fatalf("the iterator yielded %v; a turn's failure belongs on TurnEnd", err)
		}
		events = append(events, e)
	}

	last, ok := events[len(events)-1].(agent.TurnEnd)
	if !ok {
		t.Fatalf("the last event is %T, want TurnEnd", events[len(events)-1])
	}
	if last.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", last.StopReason)
	}
	if !ai.IsAuth(last.Err) {
		t.Errorf("TurnEnd.Err = %v, want the auth failure", last.Err)
	}
}

// Every event says which exchange it belongs to, so a consumer following
// several does not have to remember which one it is in: a fragment that cannot
// say what it belongs to is one an interface has to guess about.
func TestEveryEventCarriesItsTurn(t *testing.T) {
	slow := agent.ToolFunc("look", "Look something up.",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			agent.Report(ctx, agent.TextResult("halfway"))
			return agent.TextResult("found it"), nil
		})
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("c1", "look", `{}`),
		text("here it is"),
		text("and again"),
	}}, agent.WithTools(slow), agent.WithMessages([]ai.Message{ai.UserMessage("from a session")}))

	seen := map[string]bool{}
	for turn := 1; turn <= 2; turn++ {
		events, err := collect(t, a, ai.UserMessage("go"))
		if err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		for _, e := range events {
			name := fmt.Sprintf("%T", e)
			seen[name] = true
			field := reflect.ValueOf(e).FieldByName("Turn")
			if !field.IsValid() {
				t.Errorf("%s carries no turn at all", name)
				continue
			}
			if got := int(field.Int()); got != turn {
				t.Errorf("%s says turn %d, want %d", name, got, turn)
			}
		}
	}

	// The first exchange goes through every kind of event there is, so this is
	// the whole set and not the handful that happened to be easy.
	if len(seen) != 10 {
		t.Errorf("the exchanges reported %d kinds of event %v, want the 10 there are", len(seen), seen)
	}
}
