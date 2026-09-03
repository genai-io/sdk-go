package agent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestTheGateSeesTheMessageThatRequestedTheCall(t *testing.T) {
	var seen int
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{
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

	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{
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
	if !strings.Contains(results[0].Text(), "disabled") {
		t.Errorf("the model was told %q, which does not say why", results[0].Text())
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
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{
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

	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{
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

// PreInfer runs at the top of every step, not once per turn — a hook that
// prunes history has to see the history each time it grew.
func TestPreInferRunsOnEveryStep(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo it back.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("ok"), nil
		})

	var sizes []int
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{
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
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{text("fine")}}, ai.Model{ID: "stub", API: "stub"})
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

// A PreInfer error ends the turn before the model is called at all. It is not
// turned into a message the model gets to see: nothing had happened yet, and
// inventing a turn to carry the error would report something that never took
// place. Nor is there a span — a call that never started is never ended.
func TestAPreInferErrorEndsTheTurnBeforeAnythingIsSent(t *testing.T) {
	driver := &scripted{Scripts: [][]ai.Delta{text("never reached")}}
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
	calls := driver.Calls()
	if calls != 0 {
		t.Errorf("the model was called %d times after a hook refused", calls)
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
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{text("done")}},
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
		Errs:    []error{&ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}},
		Scripts: [][]ai.Delta{nil, text("second time lucky")},
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
	client := ai.NewClientWithDriver(&scripted{Scripts: [][]ai.Delta{text("fine")}},
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

	driver := &scripted{Scripts: [][]ai.Delta{text("ok")}, Keep: true}
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

	if got := driver.Sent(); len(got) != 1 {
		t.Fatalf("the driver saw %d requests, want 1", len(got))
	}
	sent := driver.Sent()[0]

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

// PostInfer runs on what came back, and edits it before it enters the
// conversation — the seam a redaction or an annotation needs.
func TestPostInferEditsWhatEntersTheConversation(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("my number is 555-1234")}},
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
	driver := &scripted{Scripts: [][]ai.Delta{text("one"), text("two"), text("three")}}
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

	calls := driver.Calls()
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

// A hook that refuses is refusing, whatever kind of error it picks. One that
// happens to look transient must still not be retried, and must not leave an
// MessageEnd with no MessageStart before it.
func TestARetryableLookingRefusalIsStillARefusal(t *testing.T) {
	driver := &scripted{Scripts: [][]ai.Delta{text("one"), text("two"), text("three")}}
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

	calls := driver.Calls()
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

// The two answers a gate can give are not the same answer. A Decision that
// blocks is policy: the model is told and gets to try something else. An error
// is infrastructure — the hook could not do its job — and the exchange ends.
func TestAGateRefusesWithADecisionAndFailsWithAnError(t *testing.T) {
	ran := 0
	tool := agent.ToolFunc("rm", "Delete things.",
		func(context.Context, struct{}) (agent.Result, error) {
			ran++
			return agent.TextResult("deleted"), nil
		})

	for _, tc := range []struct {
		name    string
		gate    func(context.Context, agent.PreToolContext) (agent.Decision, error)
		reason  agent.StopReason
		asked   int // model calls: a refusal earns another, a failure does not
		results bool
	}{
		{"a refusal is a tool error the model reads",
			func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				return agent.Decision{Block: true, Reason: "not on this machine"}, nil
			}, agent.StopEndTurn, 2, true},

		{"a failure ends the exchange",
			func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				return agent.Decision{}, errors.New("the permission service is down")
			}, agent.StopError, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ran = 0
			d := &scripted{Scripts: [][]ai.Delta{
				toolCall("c1", "rm", `{}`),
				text("all right then"),
			}}
			a := newAgent(t, d, agent.WithTools(tool),
				agent.WithHooks(agent.Hook{PreTool: tc.gate}))

			events, _ := collect(t, a, ai.UserMessage("go"))
			last := events[len(events)-1].(agent.TurnEnd)
			if last.StopReason != tc.reason {
				t.Errorf("stop reason = %q, want %q", last.StopReason, tc.reason)
			}
			if ran != 0 {
				t.Errorf("the tool ran %d times; neither answer lets it", ran)
			}
			if n := d.Calls(); n != tc.asked {
				t.Errorf("the model was called %d times, want %d", n, tc.asked)
			}

			// What the model was told, if anything: a refused call is answered,
			// a failed exchange leaves the call unanswered and ends.
			answered := false
			for _, m := range a.Messages() {
				if len(m.ToolResults()) > 0 {
					answered = true
				}
			}
			if answered != tc.results {
				t.Errorf("the conversation holds tool results = %v, want %v", answered, tc.results)
			}
			if tc.reason == agent.StopError && last.Err == nil {
				t.Error("the outcome does not carry the hook's error")
			}
		})
	}
}

// The span still closes on the way out: a reader that saw a call start has not
// gone anywhere, whichever hook failed.
func TestAFailedToolHookStillClosesTheSpan(t *testing.T) {
	tool := agent.ToolFunc("look", "Look something up.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("found"), nil
		})

	for _, tc := range []struct {
		name string
		hook agent.Hook
		ran  bool // whether the tool itself got to run before the hook failed
	}{
		{"before the tool", agent.Hook{
			PreTool: func(context.Context, agent.PreToolContext) (agent.Decision, error) {
				return agent.Decision{}, errors.New("gate unreachable")
			}}, false},
		{"after it", agent.Hook{
			PostTool: func(context.Context, agent.PostToolContext) (*agent.Result, error) {
				return nil, errors.New("the audit log is full")
			}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
				toolCall("c1", "look", `{}`),
				text("unreachable"),
			}}, agent.WithTools(tool), agent.WithHooks(tc.hook))

			events, _ := collect(t, a, ai.UserMessage("go"))

			var starts, ends int
			var end agent.ToolEnd
			for _, e := range events {
				switch v := e.(type) {
				case agent.ToolStart:
					starts++
				case agent.ToolEnd:
					ends, end = ends+1, v
				}
			}
			if starts != 1 || ends != 1 {
				t.Errorf("the call opened %d spans and closed %d, want one of each", starts, ends)
			}
			if tc.ran && end.Result.Text() != "found" {
				t.Errorf("ToolEnd carries %q, want what the tool returned before the hook failed",
					end.Result.Text())
			}
			last := events[len(events)-1].(agent.TurnEnd)
			if last.StopReason != agent.StopError || last.Err == nil {
				t.Errorf("the exchange ended %q / %v, want error and the hook's own", last.StopReason, last.Err)
			}
		})
	}
}

// A hook runs on the goroutine ranging over Run, so a panic in one is the
// caller's to recover — unlike a tool, which runs where nobody else can. What
// the agent owes is to come out of it idle rather than holding the exchange.
func TestAPanickingHookIsTheCallersToCatch(t *testing.T) {
	first := true
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("never reached"), text("fine")}},
		agent.WithHooks(agent.Hook{
			PreInfer: func(context.Context, *agent.Inference) error {
				if first {
					first = false
					panic("the hook blew up")
				}
				return nil
			},
		}))

	caught := func() (p any) {
		defer func() { p = recover() }()
		for range a.Run(context.Background(), ai.UserMessage("go")) {
		}
		return nil
	}()
	if caught == nil {
		t.Fatal("the panic never reached the caller; it was swallowed somewhere")
	}

	// And the agent is free: a second exchange is not ErrBusy.
	if out, err := outcome(t, a, ai.UserMessage("again")); err != nil || out.StopReason != agent.StopEndTurn {
		t.Errorf("the exchange after a panicking hook = %v / %q, want a clean end_turn", err, out.StopReason)
	}
}

// A hook that failed partway through a batch takes the batch with it. The calls
// that had already opened a span still close, and one that was vetted and will
// now never run says so rather than reporting a result it does not have.
func TestAFailedHookClosesTheCallsThatWillNeverRun(t *testing.T) {
	ran := 0
	tool := agent.ToolFunc("touch", "Touch something.",
		func(context.Context, struct{}) (agent.Result, error) {
			ran++
			return agent.TextResult("touched"), nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "touch", Input: `{}`})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c2", Name: "touch", Input: `{}`})},
			{StopReason: ai.StopToolUse},
		},
		text("unreachable"),
	}}, agent.WithTools(tool), agent.WithHooks(agent.Hook{
		PreTool: func(_ context.Context, c agent.PreToolContext) (agent.Decision, error) {
			if c.Call.ID == "c2" {
				return agent.Decision{}, errors.New("the gate lost its database")
			}
			return agent.Decision{}, nil
		},
	}))

	events, _ := collect(t, a, ai.UserMessage("go"))
	if ran != 0 {
		t.Errorf("%d tools ran; the batch was abandoned before any of them", ran)
	}

	ends := map[string]error{}
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok {
			ends[v.ID] = v.Err
		}
	}
	if len(ends) != 2 {
		t.Fatalf("%d calls were closed, want the 2 that were announced", len(ends))
	}
	if ends["c1"] == nil || strings.Contains(ends["c1"].Error(), "database") {
		t.Errorf("the vetted call closed with %v, want it saying it never ran", ends["c1"])
	}
	if ends["c2"] == nil || !strings.Contains(ends["c2"].Error(), "database") {
		t.Errorf("the call the hook failed on closed with %v, want the hook's own error", ends["c2"])
	}
	if last := events[len(events)-1].(agent.TurnEnd); last.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", last.StopReason)
	}
}

// PreStep replaces the conversation at the boundary, and the replacement is
// announced there: before the call that carries it, with nothing appended in
// between for a fold to pass through.
func TestPreStepReplacesTheConversationAtTheBoundary(t *testing.T) {
	driver := &scripted{Scripts: [][]ai.Delta{text("ok")}, Keep: true}
	a := newAgent(t, driver, agent.WithHooks(agent.Hook{
		PreStep: func(_ context.Context, c agent.PreStepContext) ([]ai.Message, error) {
			if c.Tokens == 0 {
				t.Error("the boundary priced the conversation at nothing")
			}
			return []ai.Message{ai.UserMessage("(the summary)")}, nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("and now this"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	at := slices.IndexFunc(events, func(e agent.Event) bool {
		_, ok := e.(agent.MessagesReplaced)
		return ok
	})
	call := slices.IndexFunc(events, func(e agent.Event) bool {
		_, ok := e.(agent.MessageStart)
		return ok
	})
	switch {
	case at < 0:
		t.Fatal("the hook replaced the conversation and nothing was announced")
	case at > call:
		t.Error("the replacement was announced after the call that carried it")
	}
	if msgs := events[at].(agent.MessagesReplaced).Messages; len(msgs) != 1 ||
		msgs[0].Text() != "(the summary)" {
		t.Errorf("the announcement carries %d messages, want the hook's one", len(msgs))
	}

	// And it is what went out, not what the hook was handed.
	if sent := driver.Sent()[0].Messages; len(sent) != 1 || sent[0].Text() != "(the summary)" {
		t.Errorf("the call sent %d messages, want the replacement", len(sent))
	}
}

// Nil is how a hook says it wants nothing changed, which is what it says on
// most steps: announcing a replacement that replaced nothing would record a
// snapshot per step forever.
func TestPreStepReturningNilChangesNothing(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
		agent.WithHooks(agent.Hook{
			PreStep: func(context.Context, agent.PreStepContext) ([]ai.Message, error) {
				return nil, nil
			},
		}))

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, e := range events {
		if _, ok := e.(agent.MessagesReplaced); ok {
			t.Error("a hook that changed nothing was reported as replacing the conversation")
		}
	}
	if got := len(a.Messages()); got != 2 {
		t.Errorf("the conversation holds %d messages, want the exchange's own two", got)
	}
}

// The figure is measured at the boundary, not remembered from the last call:
// a conversation still priced as the one it replaced asks to be replaced
// again, every step, forever.
func TestPreStepIsPricedAgainstTheConversationItHasNow(t *testing.T) {
	var seen []int
	done := false
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("c1", "noop", `{}`),
		text("done"),
	}},
		agent.WithMessages([]ai.Message{
			ai.UserMessage(strings.Repeat("a long history. ", 200)),
			ai.AssistantMessage(strings.Repeat("and a long answer. ", 200)),
		}),
		agent.WithHooks(agent.Hook{
			PreStep: func(_ context.Context, c agent.PreStepContext) ([]ai.Message, error) {
				seen = append(seen, c.Tokens)
				if done {
					return nil, nil
				}
				done = true
				return []ai.Message{ai.UserMessage("(short)")}, nil
			},
		}))

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if len(seen) != 2 {
		t.Fatalf("the hook was asked %d times, want once per step", len(seen))
	}
	if seen[1] >= seen[0] {
		t.Errorf("the second step was priced at %d against the first's %d, want the conversation it has now",
			seen[1], seen[0])
	}
}

// The hooks chain, so the second is asked about what the first left — and the
// pair is one replacement, because the conversation in between never reached
// a model.
func TestPreStepHooksChainIntoOneReplacement(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
		agent.WithHooks(
			agent.Hook{PreStep: func(context.Context, agent.PreStepContext) ([]ai.Message, error) {
				return []ai.Message{ai.UserMessage("first")}, nil
			}},
			agent.Hook{PreStep: func(_ context.Context, c agent.PreStepContext) ([]ai.Message, error) {
				if len(c.Messages) != 1 || c.Messages[0].Text() != "first" {
					t.Errorf("the second hook was handed %d messages, want what the first left", len(c.Messages))
				}
				return append(slices.Clone(c.Messages), ai.UserMessage("second")), nil
			}},
		))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	var announced []agent.MessagesReplaced
	for _, e := range events {
		if v, ok := e.(agent.MessagesReplaced); ok {
			announced = append(announced, v)
		}
	}
	if len(announced) != 1 {
		t.Fatalf("%d replacements were announced, want the one the pair made", len(announced))
	}
	if msgs := announced[0].Messages; len(msgs) != 2 || msgs[1].Text() != "second" {
		t.Errorf("the announcement carries %d messages, want both hooks' work", len(msgs))
	}
}

// A hook that could not do its job ends the exchange, the same as every other.
func TestPreStepFailingEndsTheExchange(t *testing.T) {
	boom := errors.New("the summariser is down")
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("never asked")}},
		agent.WithHooks(agent.Hook{
			PreStep: func(context.Context, agent.PreStepContext) ([]ai.Message, error) {
				return nil, boom
			},
		}))

	out, err := outcome(t, a, ai.UserMessage("go"))
	if err == nil || !errors.Is(err, boom) {
		t.Fatalf("the turn ended with %v, want the hook's own error", err)
	}
	if out.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", out.StopReason)
	}
}

// kinds renders the event types in order, which is what a claim about where a
// span opens and closes is really about.
func kinds(events []agent.Event) []string {
	out := make([]string, 0, len(events))
	for _, e := range events {
		out = append(out, strings.TrimPrefix(fmt.Sprintf("%T", e), "agent."))
	}
	return out
}

// Shortening the conversation is a model call of its own, and a stream that
// says nothing for that long looks stopped. The hook says when the wait
// begins, because only the code deciding to shorten knows it is about to; the
// loop closes it, after the replacement it caused.
func TestCompactingOpensASpanAroundTheWait(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
		agent.WithHooks(agent.Hook{
			PreStep: func(ctx context.Context, _ agent.PreStepContext) ([]ai.Message, error) {
				agent.Compacting(ctx)
				return []ai.Message{ai.UserMessage("(the summary)")}, nil
			},
		}))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	got := kinds(events)
	start := slices.Index(got, "CompactionStart")
	replaced := slices.Index(got, "MessagesReplaced")
	end := slices.Index(got, "CompactionEnd")
	switch {
	case start < 0 || end < 0:
		t.Fatalf("the span is %v, want it opened and closed", got)
	case start >= replaced || replaced >= end:
		t.Errorf("events are %v, want the replacement inside the span", got)
	}
	opened := events[start].(agent.CompactionStart)
	if len(opened.Messages) != 1 || opened.Tokens == 0 {
		t.Errorf("the span opened on %d messages priced at %d, want what the hook was handed",
			len(opened.Messages), opened.Tokens)
	}
	// And it closes on what opened it, like every other span here.
	closed := events[end].(agent.CompactionEnd)
	if closed.Err != nil || closed.Tokens != opened.Tokens ||
		len(closed.Messages) != len(opened.Messages) {
		t.Errorf("the span closed on %d messages priced at %d (%v), want what opened it",
			len(closed.Messages), closed.Tokens, closed.Err)
	}
}

// A hook that says nothing costs nothing: the span is not opened around every
// step boundary on the chance that this one was slow.
func TestAQuietPreStepOpensNoSpan(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
		agent.WithHooks(agent.Hook{
			PreStep: func(context.Context, agent.PreStepContext) ([]ai.Message, error) {
				return []ai.Message{ai.UserMessage("trimmed")}, nil
			},
		}))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, k := range kinds(events) {
		if strings.HasPrefix(k, "Compaction") {
			t.Errorf("events are %v, want no span around a hook that announced none", kinds(events))
			break
		}
	}
}

// The span closes whichever way the hook went: a consumer that opened a
// spinner has to be told to stop, and the two ways that used to leave it
// running are a hook that failed and one that thought better of it.
func TestAnAnnouncedCompactionAlwaysCloses(t *testing.T) {
	boom := errors.New("the summariser is down")
	for _, tc := range []struct {
		name string
		hook func(context.Context, agent.PreStepContext) ([]ai.Message, error)
		want error
	}{
		{"failed", func(ctx context.Context, _ agent.PreStepContext) ([]ai.Message, error) {
			agent.Compacting(ctx)
			return nil, boom
		}, boom},
		{"thought better of it", func(ctx context.Context, _ agent.PreStepContext) ([]ai.Message, error) {
			agent.Compacting(ctx)
			return nil, nil
		}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
				agent.WithHooks(agent.Hook{PreStep: tc.hook}))

			events, _ := collect(t, a, ai.UserMessage("go"))
			at := slices.IndexFunc(events, func(e agent.Event) bool {
				_, ok := e.(agent.CompactionEnd)
				return ok
			})
			if at < 0 {
				t.Fatalf("events are %v, want the span closed", kinds(events))
			}
			if got := events[at].(agent.CompactionEnd).Err; !errors.Is(got, tc.want) {
				t.Errorf("the span closed with %v, want %v", got, tc.want)
			}
		})
	}
}

// The failure a caller can fix and the loop cannot: a prompt the provider
// called too long is shortened and the same step is taken again. Replaying it
// unchanged, which is all WithRetry could do, fails the same way every time.
func TestOnInferErrorRecoversAnOversizedPrompt(t *testing.T) {
	driver := &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindContextExceeded, Message: "prompt is too long"}},
		Scripts: [][]ai.Delta{nil, text("that fits")},
		Keep:    true,
	}
	asked := 0
	a := newAgent(t, driver, agent.WithHooks(agent.Hook{
		OnInferError: func(ctx context.Context, c agent.InferErrorContext) (*agent.Retry, error) {
			asked++
			if !ai.IsContextExceeded(c.Err) {
				return nil, nil
			}
			agent.Compacting(ctx)
			return &agent.Retry{Messages: []ai.Message{ai.UserMessage("(the summary)")}}, nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("a very long conversation"))
	if err != nil {
		t.Fatalf("the turn was not recovered: %v", err)
	}
	if asked != 1 {
		t.Errorf("the hook was asked %d times, want once", asked)
	}

	// A recovery is the shape a retry already had — no event of its own — with
	// the replacement announced between the two attempts.
	assertSequence(t, events, []string{
		"TurnStart",
		"MessageAdded(user)",
		"MessageStart(attempt=1)",
		"MessageEnd(err)",
		"CompactionStart",
		"MessagesReplaced",
		"CompactionEnd",
		"MessageStart(attempt=2)",
		"MessageUpdate",
		"MessageUpdate",
		"MessageUpdate",
		"MessageEnd",
		"MessageAdded(assistant)",
		"TurnEnd",
	})

	// And the second attempt carried the replacement, not what was refused.
	if sent := driver.Sent()[1].Messages; len(sent) != 1 || sent[0].Text() != "(the summary)" {
		t.Errorf("the retried call sent %d messages, want the shortened one", len(sent))
	}
}

// Without the hook the loop's own answer stands: an oversized prompt is not
// retryable, so nothing recovers it and the exchange ends on it.
func TestAnOversizedPromptEndsTheExchangeWithNoHook(t *testing.T) {
	a := newAgent(t, &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindContextExceeded, Message: "prompt is too long"}},
		Scripts: [][]ai.Delta{nil},
	})

	out, err := outcome(t, a, ai.UserMessage("hi"))
	if err == nil {
		t.Fatal("the failure never reached the caller")
	}
	if out.StopReason != agent.StopError {
		t.Errorf("stop reason = %q, want error", out.StopReason)
	}
}

// The two budgets are separate and do not multiply. WithRetry replays a call
// the loop knows how to replay; a recovery hands over a call that has been
// changed, so it spends nothing and the replay budget starts over with it.
func TestARecoveryDoesNotSpendTheRetryBudgetAndRestartsIt(t *testing.T) {
	overloaded := &ai.Error{Kind: ai.KindOverloaded, Message: "overloaded"}
	driver := &scripted{
		Errs:    []error{overloaded, overloaded, overloaded, overloaded},
		Scripts: [][]ai.Delta{nil, nil, nil, nil},
	}

	var attempts []int
	a := newAgent(t, driver, agent.WithRetry(2, 0), agent.WithHooks(agent.Hook{
		OnInferError: func(_ context.Context, c agent.InferErrorContext) (*agent.Retry, error) {
			attempts = append(attempts, c.Attempt)
			if len(attempts) > 1 {
				return nil, nil // out of ideas the second time
			}
			return &agent.Retry{}, nil
		},
	}))

	if _, err := outcome(t, a, ai.UserMessage("hi")); err == nil {
		t.Fatal("every attempt failed and the turn reported success")
	}

	// Two attempts on the budget, a recovery, then two more on a budget that
	// started over — and the hook asked only where the loop ran out of answers.
	if driver.Calls() != 4 {
		t.Errorf("the model was called %d times, want 4", driver.Calls())
	}
	if want := []int{2, 4}; !slices.Equal(attempts, want) {
		t.Errorf("the hook was asked at attempts %v, want %v", attempts, want)
	}
}

// Nil is agreement to give up, and what the turn fails with is then the
// model's own failure: the hook declined to answer it, it did not replace it.
func TestDecliningToRecoverKeepsTheOriginalFailure(t *testing.T) {
	tooLong := &ai.Error{Kind: ai.KindContextExceeded, Message: "prompt is too long"}
	a := newAgent(t, &scripted{Errs: []error{tooLong}, Scripts: [][]ai.Delta{nil}},
		agent.WithHooks(agent.Hook{
			OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
				return nil, nil
			},
		}))

	out, _ := outcome(t, a, ai.UserMessage("hi"))
	if !errors.Is(out.Err, tooLong) {
		t.Errorf("the turn failed with %v, want the model's own failure", out.Err)
	}
}

// A hook that could not do its job is the application failing, not the model,
// so the exchange ends with what the hook said rather than what it was asked
// about — the rule every other hook here already keeps.
func TestAFailingRecoveryHookEndsTheExchange(t *testing.T) {
	boom := errors.New("the summariser is down")
	a := newAgent(t, &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindContextExceeded, Message: "too long"}},
		Scripts: [][]ai.Delta{nil},
	}, agent.WithHooks(agent.Hook{
		OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
			return nil, boom
		},
	}))

	out, _ := outcome(t, a, ai.UserMessage("hi"))
	if out.StopReason != agent.StopError || !errors.Is(out.Err, boom) {
		t.Errorf("the turn ended %q with %v, want error carrying the hook's own",
			out.StopReason, out.Err)
	}
}

// An objection to an answer that arrived does not earn another go at getting
// one. PostInfer's error is not a call that failed, and offering it here would
// let a hook loop on an answer the model already gave.
func TestAPostInferRefusalIsNotOfferedToTheRecoveryHook(t *testing.T) {
	refused := errors.New("that answer will not do")
	asked := false
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("here you go")}},
		agent.WithHooks(agent.Hook{
			PostInfer: func(context.Context, *ai.Response) error { return refused },
			OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
				asked = true
				return &agent.Retry{}, nil
			},
		}))

	out, _ := outcome(t, a, ai.UserMessage("hi"))
	if asked {
		t.Error("the hook was asked about an answer that arrived")
	}
	if !errors.Is(out.Err, refused) {
		t.Errorf("the turn failed with %v, want the refusal", out.Err)
	}
}

// The other half of the seam. This hook says whether there is another attempt;
// PreInfer says where it goes, which it can only decide knowing what ended the
// one before — so routing stays in one place and does not move here.
func TestPreInferIsToldWhatEndedTheAttemptBefore(t *testing.T) {
	tooLong := &ai.Error{Kind: ai.KindContextExceeded, Message: "prompt is too long"}
	var seen []error
	a := newAgent(t, &scripted{
		Errs:    []error{tooLong},
		Scripts: [][]ai.Delta{nil, text("that fits")},
	}, agent.WithHooks(agent.Hook{
		PreInfer: func(_ context.Context, inf *agent.Inference) error {
			seen = append(seen, inf.LastErr)
			return nil
		},
		OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
			return &agent.Retry{Messages: []ai.Message{ai.UserMessage("shorter")}}, nil
		},
	}))

	if _, err := outcome(t, a, ai.UserMessage("hi")); err != nil {
		t.Fatalf("the turn was not recovered: %v", err)
	}
	if len(seen) != 2 {
		t.Fatalf("PreInfer ran %d times, want once per attempt", len(seen))
	}
	if seen[0] != nil {
		t.Errorf("the first attempt was told it followed %v, want nothing", seen[0])
	}
	if !errors.Is(seen[1], tooLong) {
		t.Errorf("the second attempt was told it followed %v, want the failure", seen[1])
	}
}

// A caller that only wanted the call sent elsewhere leaves Messages nil, and
// nothing is announced: a consumer told the conversation was replaced would
// throw away a history nobody replaced.
func TestARouteOnlyRecoveryReplacesNothing(t *testing.T) {
	a := newAgent(t, &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindAuth, Message: "bad key"}},
		Scripts: [][]ai.Delta{nil, text("the other endpoint")},
	}, agent.WithHooks(agent.Hook{
		OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
			return &agent.Retry{}, nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("hi"))
	if err != nil {
		t.Fatalf("the turn was not recovered: %v", err)
	}
	if slices.Contains(kinds(events), "MessagesReplaced") {
		t.Errorf("events are %v, want nothing announced for a conversation left alone",
			kinds(events))
	}
}

// Two hooks that both want the next attempt: the first answer is taken, as the
// first refusal is at the gate. Letting the last one win would make the answer
// depend on registration order in a way nobody reading either could see.
func TestTheFirstRecoveryAnswerIsTaken(t *testing.T) {
	second := false
	a := newAgent(t, &scripted{
		Errs:    []error{&ai.Error{Kind: ai.KindContextExceeded, Message: "too long"}},
		Scripts: [][]ai.Delta{nil, text("ok")},
	},
		agent.WithHooks(agent.Hook{
			OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
				return &agent.Retry{Messages: []ai.Message{ai.UserMessage("first")}}, nil
			},
		}, agent.Hook{
			OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
				second = true
				return &agent.Retry{Messages: []ai.Message{ai.UserMessage("second")}}, nil
			},
		}))

	if _, err := outcome(t, a, ai.UserMessage("hi")); err != nil {
		t.Fatalf("the turn was not recovered: %v", err)
	}
	if second {
		t.Error("the second hook was asked after the first had answered")
	}
	if msgs := a.Messages(); len(msgs) == 0 || msgs[0].Text() != "first" {
		t.Errorf("the conversation is %v, want it starting with the first hook's replacement",
			msgs)
	}
}

// waiting is an endpoint that never answers, so the only thing that can end a
// call to it is the caller.
type waiting struct{}

func (waiting) Name() string { return "waiting" }

func (waiting) Stream(ctx context.Context, _ *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		<-ctx.Done()
		yield(ai.Delta{}, ctx.Err())
	}
}

// A turn that was stopped is not a turn that failed to answer. Asking the hook
// to recover it would restart the very exchange the caller just ended.
func TestACancelledTurnIsNotOfferedToTheRecoveryHook(t *testing.T) {
	var asked atomic.Bool
	a := newAgent(t, waiting{}, agent.WithHooks(agent.Hook{
		OnInferError: func(context.Context, agent.InferErrorContext) (*agent.Retry, error) {
			asked.Store(true)
			return &agent.Retry{}, nil
		},
	}))

	started := make(chan struct{})
	go func() {
		<-started
		a.Interrupt()
	}()

	var out agent.TurnEnd
	for e, err := range a.Run(context.Background(), ai.UserMessage("hi")) {
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if _, ok := e.(agent.MessageStart); ok {
			close(started)
		}
		if v, ok := e.(agent.TurnEnd); ok {
			out = v
		}
	}

	if asked.Load() {
		t.Error("the hook was asked to recover a turn the caller had ended")
	}
	if out.StopReason != agent.StopCanceled {
		t.Errorf("stop reason = %q, want canceled", out.StopReason)
	}
}
