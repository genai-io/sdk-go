package session_test

import (
	"context"
	"errors"
	"iter"
	"slices"
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

	ctx := context.Background()
	for _, m := range msgs {
		for e, err := range a.Run(ctx, m) {
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			rec.Handle(ctx, e)
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
	if counts[session.EntryExchange] != 1 {
		t.Errorf("exchange entries = %d, want 1", counts[session.EntryExchange])
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

// The shape examples/agent-session teaches, and the one that matters: the
// process ends. A second store opened on the same directory — a different
// *jsonl.Store, holding none of the first one's state — restores what the
// first recorded, and carries on numbering where it left off.
func TestASecondProcessPicksUpTheConversation(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	first, err := jsonl.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	a := newAgent(t, nil, text("morning"), text("afternoon"))
	rec, history, err := session.Open(ctx, first, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(history) != 0 {
		t.Fatalf("a new session restored %d messages", len(history))
	}
	converse(t, a, rec, ai.UserMessage("hello"))
	id := rec.ID()
	if err := first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// A new process: nothing survives but the directory.
	second, err := jsonl.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer second.Close()

	// Finding it the way a program would, rather than by remembering the id.
	sessions, err := second.List(ctx)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 1 || sessions[0].ID != id {
		t.Fatalf("List = %+v, want the one session just written", sessions)
	}

	rec2, restored, err := session.Open(ctx, second, id)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(restored) != len(a.Messages()) {
		t.Fatalf("restored %d messages, the first process ended holding %d",
			len(restored), len(a.Messages()))
	}
	if got := restored[0].Text(); got != "hello" {
		t.Errorf("the conversation starts at %q, want what was asked first", got)
	}

	// And it carries on: a second exchange lands after the first, not on top.
	b := newAgent(t, restored, text("afternoon"))
	converse(t, b, rec2, ai.UserMessage("still there?"))

	final, err := session.Messages(ctx, second, id)
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(final) != len(b.Messages()) {
		t.Errorf("the store holds %d messages, the agent holds %d", len(final), len(b.Messages()))
	}
	if got := final[0].Text(); got != "hello" {
		t.Errorf("the first message is %q; the second process overwrote history", got)
	}
}

// flaky fails one append and works for everything else.
type flaky struct {
	session.Store
	failOn  int
	appends int
}

func (f *flaky) Append(ctx context.Context, id string, e ...session.Entry) error {
	f.appends++
	if f.appends == f.failOn {
		return errors.New("the disk, briefly")
	}
	return f.Store.Append(ctx, id, e...)
}

// A conversation is read back by folding it, so a hole in the middle is not a
// shorter conversation — it is a broken one. Drop the message carrying a tool
// call and the result answering it is orphaned, which is a shape no provider
// accepts and ai.Repair silently deletes. Recording therefore stops at the
// first failure: a prefix still folds.
func TestRecordingStopsAtTheFirstFailedWrite(t *testing.T) {
	ctx := context.Background()
	st := &flaky{Store: store(t), failOn: 3}

	a := newAgent(t, nil,
		[]ai.Delta{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "noop", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("done"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatal(err)
	}
	converse(t, a, rec, ai.UserMessage("go"))

	if rec.Err() == nil {
		t.Fatal("a failed write was not reported")
	}
	restored, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("the session no longer folds: %v", err)
	}
	// Whatever survived must be a conversation, not a fragment of one: every
	// tool result answers a call that is still there.
	if repaired := ai.Repair(restored); len(repaired) != len(restored) {
		t.Errorf("Repair dropped %d of %d restored messages — the fold has a hole",
			len(restored)-len(repaired), len(restored))
	}
	// And nothing was written after the failure.
	if st.appends != 3 {
		t.Errorf("the store saw %d appends, want 3 — recording carried on past the failure", st.appends)
	}
}

// The agent numbers turns from one every run. A session that stored that
// number as its own would hold two exchanges both called turn 1 after a
// resume, and no consumer could group by it.
func TestTurnsAreNumberedFromTheSessionsBeginning(t *testing.T) {
	ctx := context.Background()
	st := store(t)

	a := newAgent(t, nil, text("one"), text("two"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatal(err)
	}
	converse(t, a, rec, ai.UserMessage("first"), ai.UserMessage("second"))

	// A second run, resuming: the agent starts counting at one again.
	b := newAgent(t, nil, text("three"))
	rec2, history, err := session.Open(ctx, st, rec.ID())
	if err != nil {
		t.Fatal(err)
	}
	b.SetMessages(history)
	converse(t, b, rec2, ai.UserMessage("third"))

	var turns []int
	for e, err := range st.Entries(ctx, rec.ID()) {
		if err != nil {
			t.Fatal(err)
		}
		if e.Type == session.EntryExchange {
			turns = append(turns, e.Exchange.Turn)
		}
	}
	if want := []int{1, 2, 3}; !slices.Equal(turns, want) {
		t.Errorf("exchange numbers = %v, want %v", turns, want)
	}
}

// Resuming hands the agent the history it just read, and the agent announces
// it — correctly, it did replace its conversation. Storing that would write a
// copy of the whole conversation on every resume, forever.
func TestResumingDoesNotRecordACopyOfWhatItRead(t *testing.T) {
	ctx := context.Background()
	st := store(t)

	a := newAgent(t, nil, text("one"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatal(err)
	}
	converse(t, a, rec, ai.UserMessage("first"))

	for range 5 {
		b := newAgent(t, nil, text("again"))
		rec2, history, err := session.Open(ctx, st, rec.ID())
		if err != nil {
			t.Fatal(err)
		}
		b.SetMessages(history)
		converse(t, b, rec2, ai.UserMessage("more"))
	}

	snapshots := 0
	for e, err := range st.Entries(ctx, rec.ID()) {
		if err != nil {
			t.Fatal(err)
		}
		if e.Type == session.EntrySnapshot {
			snapshots++
		}
	}
	if snapshots != 0 {
		t.Errorf("%d snapshots written by resuming alone; the conversation was never replaced", snapshots)
	}
}
