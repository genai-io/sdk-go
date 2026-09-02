package agent_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/internal/scripted"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// The vocabulary these tests are written in: a scripted model, and the two
// shapes of answer it gives.
var (
	text     = scripted.Text
	toolCall = scripted.ToolCall
)

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

// attempts counts every model call, retries included, where steps counts only
// the ones that were not.
func attempts(events []agent.Event) int {
	n := 0
	for _, e := range events {
		if _, ok := e.(agent.MessageStart); ok {
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

// An exchange closes nothing, so a caller may take several in a row, and the
// turns are numbered in order.
func TestExchangesRunInSequence(t *testing.T) {
	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{text("first"), text("second")}})

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
	a := newAgent(t, &scripted.Driver{
		Errs:    []error{&ai.Error{Kind: ai.KindAuth, Message: "no key"}},
		Scripts: [][]ai.Delta{nil, text("second time")},
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

	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{
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

// Repeating an exchange is a for loop the caller writes, which is what lets it
// decide the batching and what a failure means.
func TestSeveralExchangesAreACallersLoop(t *testing.T) {
	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{text("first"), text("second")}})

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

// Collect folds an exchange for a caller that wants the answer rather than the
// progress — the shape a subagent behind a tool call needs.
func TestCollectFoldsAnExchangeIntoItsOutcome(t *testing.T) {
	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{text("the answer")}})

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

// A message added while an exchange is running joins it at the next step
// boundary, and is reported there — so the stream and the conversation agree
// about what the model was shown.
func TestAddedMessagesJoinTheExchangeAtAStepBoundary(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.TextResult("echoed"), nil
		})

	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{
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

// SetTools, SetSystem and AddHook all promise the same thing: the change lands
// on the next inference, not mid-stream. A turn holds several inferences, so
// "next" is a real moment inside one — and nothing pinned it.
func TestTheAgentIsReconfiguredBetweenInferences(t *testing.T) {
	d := &scripted.Driver{Scripts: [][]ai.Delta{
		toolCall("1", "before", "{}"),
		text("done"),
	}, Keep: true}
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

	sent := d.Sent()
	if len(sent) != 2 {
		t.Fatalf("the driver saw %d calls, want 2", len(sent))
	}
	if sent[0].System != "first prompt" {
		t.Errorf("call 1 system = %q, want the original", sent[0].System)
	}
	if sent[1].System != "second prompt" {
		t.Errorf("call 2 system = %q, want the replacement", sent[1].System)
	}
	if n := len(sent[0].Tools); n != 1 || sent[0].Tools[0].Schema.Name != "before" {
		t.Errorf("call 1 offered %v, want the original toolset", toolNames(sent[0].Tools))
	}
	if n := len(sent[1].Tools); n != 1 || sent[1].Tools[0].Schema.Name != "after" {
		t.Errorf("call 2 offered %v, want the replacement", toolNames(sent[1].Tools))
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
	d := &scripted.Driver{Scripts: [][]ai.Delta{
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

// An agent with no tools offers none, whatever the client was built with. The
// toolset is the agent's, not a suggestion — and until Tools went out
// unconditionally there was no way for it to say so.
func TestAnAgentWithNoToolsOffersNone(t *testing.T) {
	stray := ai.ToolFunc("stray", "configured on the client",
		func(context.Context, struct{}) (string, error) { return "", nil })
	driver := &scripted.Driver{Scripts: [][]ai.Delta{text("ok")}, Keep: true}
	client := ai.NewClientWithDriver(driver, ai.Model{ID: "stub", API: "stub"},
		ai.WithTools(stray))

	a, err := agent.New(client)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	if n := len(driver.Sent()[0].Tools); n != 0 {
		t.Errorf("the endpoint was offered %d tools; the agent has none", n)
	}
}

// Replacing the conversation while an exchange runs lands on it at once — that
// is what the lock is for — but the announcement waits for the next exchange.
// A MessagesReplaced in the middle of one would say the history the turn is
// working from was thrown away, which is not what happened.
func TestAReplacementDuringAnExchangeIsAnnouncedByTheNextOne(t *testing.T) {
	var a *agent.Agent
	swap := agent.ToolFunc("compact", "Replace the conversation.",
		func(context.Context, struct{}) (agent.Result, error) {
			a.SetMessages([]ai.Message{ai.UserMessage("(the summary)")})
			return agent.TextResult("compacted"), nil
		})
	a = newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{
		toolCall("c1", "compact", `{}`),
		text("done"),
		text("and again"),
	}}, agent.WithTools(swap))

	during, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, e := range during {
		if _, ok := e.(agent.MessagesReplaced); ok {
			t.Error("the replacement was announced inside the exchange that made it")
		}
	}

	// The announcement carries the conversation as it stands, which is the
	// replacement plus whatever the rest of that exchange appended to it.
	before := a.Messages()
	next, err := collect(t, a, ai.UserMessage("again"))
	if err != nil {
		t.Fatalf("the next turn failed: %v", err)
	}
	replaced, ok := next[1].(agent.MessagesReplaced)
	if !ok {
		t.Fatalf("the next exchange reported %T after TurnStart, want MessagesReplaced", next[1])
	}
	if len(replaced.Messages) != len(before) || replaced.Messages[0].Text() != "(the summary)" {
		t.Errorf("the announcement carries %d messages starting %q, want the %d the agent held",
			len(replaced.Messages), replaced.Messages[0].Text(), len(before))
	}
	if replaced.Turn != 2 {
		t.Errorf("the announcement belongs to turn %d, want 2", replaced.Turn)
	}
}

// Replacing nothing with nothing happened to nobody. Announcing it records a
// snapshot of an empty conversation, and a session that folded one came back
// saying it was corrupt — which is what seeding an agent from a fresh session
// does, in both of this repository's examples.
func TestReplacingAnEmptyConversationWithAnEmptyOneIsNotNews(t *testing.T) {
	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{text("hello"), text("still here")}})
	a.SetMessages(nil) // what session.Open hands back for a session that is new

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, e := range events {
		if _, ok := e.(agent.MessagesReplaced); ok {
			t.Error("a conversation that was empty and stayed empty was announced as replaced")
		}
	}

	// Clearing one that held something is the opposite: everything announced
	// before it is gone, and a consumer that missed that hands it back.
	a.SetMessages(nil)
	events, err = collect(t, a, ai.UserMessage("go on"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	replaced, ok := events[1].(agent.MessagesReplaced)
	if !ok {
		t.Fatalf("clearing a conversation reported %T, want MessagesReplaced", events[1])
	}
	if len(replaced.Messages) != 0 {
		t.Errorf("the announcement carries %d messages, want none", len(replaced.Messages))
	}
}

// A message queued after an exchange's last step boundary belongs to the
// conversation ahead of the next exchange's own input: it was said first, and a
// fold is only the conversation if that order is the truth.
func TestMessagesQueuedBetweenExchangesEnterAheadOfTheNextInput(t *testing.T) {
	a := newAgent(t, &scripted.Driver{Scripts: [][]ai.Delta{text("first"), text("second")}})

	if _, err := outcome(t, a, ai.UserMessage("one")); err != nil {
		t.Fatalf("first turn: %v", err)
	}
	a.AddMessages(ai.UserMessage("said while nothing was running"))

	events, err := collect(t, a, ai.UserMessage("two"))
	if err != nil {
		t.Fatalf("second turn: %v", err)
	}

	var added []string
	for _, e := range events {
		if v, ok := e.(agent.MessageAdded); ok {
			added = append(added, v.Message.Text())
		}
	}
	want := []string{"said while nothing was running", "two", "second"}
	if !slices.Equal(added, want) {
		t.Errorf("the exchange announced %v, want %v", added, want)
	}

	var held []string
	for _, m := range a.Messages() {
		held = append(held, m.Text())
	}
	if !slices.Equal(held, append([]string{"one", "first"}, want...)) {
		t.Errorf("the conversation reads %v; the queued message is out of order", held)
	}
}
