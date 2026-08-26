package session_test

import (
	"context"
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/agent/session/jsonl"
	"github.com/genai-io/sdk-go/pkg/ai"
)

type scripted struct {
	mu      sync.Mutex
	scripts [][]ai.Delta
	calls   int
}

func (d *scripted) Name() string { return "scripted" }

func (d *scripted) Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	d.mu.Unlock()
	return func(yield func(ai.Delta, error) bool) {
		if n >= len(d.scripts) {
			yield(ai.Delta{}, &ai.Error{Kind: ai.KindUnknown, Message: "no script"})
			return
		}
		for _, delta := range d.scripts[n] {
			if !yield(delta, nil) {
				return
			}
		}
	}
}

func text(s string) []ai.Delta {
	return []ai.Delta{{Block: ai.TextBlock(s)}, {EndBlock: true}, {StopReason: ai.StopEndTurn}}
}

func newAgent(t *testing.T, history []ai.Message, scripts ...[]ai.Delta) *agent.Agent {
	t.Helper()
	client := ai.NewClientWithDriver(&scripted{scripts: scripts}, ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client, agent.WithMessages(history))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

// converse drives an agent through several exchanges, recording as it goes. It
// is the whole shape of using this package with a session: your loop, your
// recorder call, and the agent knowing nothing about either.
func converse(t *testing.T, a *agent.Agent, rec *session.Recorder, msgs ...ai.Message) {
	t.Helper()

	for _, m := range msgs {
		for e, err := range a.Run(context.Background(), m) {
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			rec.Handle(e)
		}
	}
}

func store(t *testing.T) *jsonl.Store {
	t.Helper()
	s, err := jsonl.Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// The round trip the whole package exists for: work, stop, come back.
func TestASessionResumesWhereItLeftOff(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	first := newAgent(t, nil, text("the capital of France is Paris"))
	rec, history, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if history != nil {
		t.Fatalf("a new session came back with %d messages", len(history))
	}
	converse(t, first, rec, ai.UserMessage("what is the capital of France?"))
	if err := rec.Err(); err != nil {
		t.Fatalf("recording failed: %v", err)
	}

	resumed, history, err := session.Open(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	second := newAgent(t, history, text("it has about two million people"))

	msgs := second.Messages()
	if len(msgs) != 2 {
		t.Fatalf("resumed with %d messages, want the 2 that were recorded", len(msgs))
	}
	if !strings.Contains(msgs[1].Text(), "Paris") {
		t.Errorf("resumed history lost the answer: %q", msgs[1].Text())
	}

	// And the resumed agent carries on into the same session.
	converse(t, second, resumed, ai.UserMessage("how big is it?"))
	if got := len(second.Messages()); got != 4 {
		t.Errorf("after continuing, %d messages, want 4", got)
	}

	all, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(all) != 4 {
		t.Errorf("the stored session holds %d messages, want 4", len(all))
	}
}

// What is written is what a resume needs plus what an audit needs — and not
// the fragments, which exist only to keep an interface current.
func TestFragmentsAreNotPersistedButTheSpansThatCloseThemAre(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, text("hello"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("hi"))

	counts := map[session.EntryType]int{}
	for e, err := range st.Entries(ctx, rec.ID()) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		counts[e.Type]++
	}

	if counts[session.EntryMessage] != 2 {
		t.Errorf("message entries = %d, want 2", counts[session.EntryMessage])
	}
	if counts[session.EntryInference] != 1 {
		t.Errorf("inference entries = %d, want 1", counts[session.EntryInference])
	}
	if counts[session.EntryTurn] != 1 {
		t.Errorf("turn entries = %d, want 1", counts[session.EntryTurn])
	}
	if total := len(counts); total != 3 {
		t.Errorf("entry types = %d, want 3 — a fragment was persisted", total)
	}
}

func TestTheInferenceEntryCarriesWhatTheCallCost(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, []ai.Delta{
		{Block: ai.TextBlock("answer")},
		{EndBlock: true},
		{Usage: &ai.Usage{Input: 120, Output: 8}, StopReason: ai.StopEndTurn},
	})
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("hi"))

	var found *session.Inference
	for e, err := range st.Entries(ctx, rec.ID()) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		if e.Type == session.EntryInference {
			found = e.Inference
		}
	}
	if found == nil {
		t.Fatal("no inference was recorded")
	}
	if found.Usage.Input != 120 || found.Usage.Output != 8 {
		t.Errorf("usage = %+v, want the call's own", found.Usage)
	}
	if found.StopReason != ai.StopEndTurn {
		t.Errorf("stop reason = %q", found.StopReason)
	}
	if found.Attempt != 1 {
		t.Errorf("attempt = %d, want 1", found.Attempt)
	}
}

func TestForkingASessionLeavesTheOriginalAlone(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, text("one"), text("two"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("first"), ai.UserMessage("second"))

	all, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(all) != 4 {
		t.Fatalf("original holds %d messages, want 4", len(all))
	}

	// Branch back to just after the first exchange.
	var cut int64
	seen := 0
	for e, err := range st.Entries(ctx, rec.ID()) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		if e.Type == session.EntryMessage {
			seen++
			if seen == 2 {
				cut = e.Seq
			}
		}
	}

	forked, err := st.Fork(ctx, rec.ID(), cut)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}

	branch, err := session.Messages(ctx, st, forked.ID)
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(branch) != 2 {
		t.Errorf("branch holds %d messages, want the 2 up to the cut", len(branch))
	}
	if again, _ := session.Messages(ctx, st, rec.ID()); len(again) != 4 {
		t.Errorf("the original changed: %d messages", len(again))
	}
}

// Recording must not be able to stop the work. A store that fails surfaces on
// the session handle instead.
func TestAFailingStoreDoesNotStopTheAgent(t *testing.T) {
	ctx := context.Background()
	st := store(t)

	a := newAgent(t, nil, text("still working"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Delete the session out from under the recorder.
	if err := st.Delete(ctx, rec.ID()); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	converse(t, a, rec, ai.UserMessage("hi"))

	if got := a.Messages(); len(got) != 2 || got[1].Text() != "still working" {
		t.Errorf("the agent stopped because recording failed: %+v", got)
	}
	if rec.Err() == nil {
		t.Error("the failed write was never surfaced")
	}
}
