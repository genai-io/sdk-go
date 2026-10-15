package session_test

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
	"github.com/genai-io/sdk-go/pkg/session"
	"github.com/genai-io/sdk-go/pkg/session/jsonl"
)

func newAgent(t *testing.T, history []ai.Message, turns ...aitest.Turn) *agent.Agent {
	t.Helper()
	client := aitest.New(turns...).Client()
	a, err := agent.New(client)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	a.SetMessages(history)
	return a
}

// converse drives an agent through several exchanges, recording as it goes. It
// is the whole shape of using this package with a session: your loop, your
// writer call, and the agent knowing nothing about either.
func converse(t *testing.T, a *agent.Agent, rec *session.Session, msgs ...ai.Message) {
	t.Helper()

	ctx := context.Background()
	for _, m := range msgs {
		for e, err := range a.Run(ctx, m) {
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			rec.RecordEvent(ctx, e)
		}
	}
}

// store is what these tests record into: the in-memory implementation, so what
// they exercise is the Store contract rather than a filesystem. The one test
// about a second process opens a jsonl store itself, because that is a claim
// about that store and not about sessions.
func store(t *testing.T) *memStore {
	t.Helper()
	return &memStore{}
}

// stores are both implementations, for what must hold of either.
var stores = []struct {
	name string
	open func(*testing.T) session.Store
}{
	{"memory", func(*testing.T) session.Store { return &memStore{} }},
	{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
}

// jsonlStore is for the tests that are about the jsonl store itself.
func jsonlStore(t *testing.T) *jsonl.Store {
	t.Helper()
	s, err := jsonl.Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return s
}

// The round trip the whole package exists for: work, stop, come back.
func TestASessionResumesWhereItLeftOff(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	first := newAgent(t, nil, aitest.Says("the capital of France is Paris"))
	rec, err := session.Create(ctx, st)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = rec.Close() })

	converse(t, first, rec, ai.UserMessage("what is the capital of France?"))
	if err := rec.Err(); err != nil {
		t.Fatalf("recording failed: %v", err)
	}

	if err := rec.Close(); err != nil {
		t.Fatal(err)
	}
	resumed, history, err := session.Open(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	t.Cleanup(func() { _ = resumed.Close() })
	second := newAgent(t, history, aitest.Says("it has about two million people"))

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

	all, err := st.Load(ctx, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(all) != 4 {
		t.Errorf("the stored session holds %d messages, want 4", len(all))
	}
}

// A session is the conversation and nothing else: what a turn cost and how it
// ended are the application's to keep, and fragments only kept a screen current.
func TestOnlyTheConversationIsRecorded(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, aitest.Says("hello"))
	rec, err := session.Create(ctx, st)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = rec.Close() })
	converse(t, a, rec, ai.UserMessage("hi"))

	if got, want := st.opsOf(rec.ID()), []string{"append", "append"}; !slices.Equal(got, want) {
		t.Errorf("writes = %v, want %v", got, want)
	}
}

// Recording must not be able to stop the work. A store that fails surfaces on
// the session handle instead.
func TestAFailingStoreDoesNotStopTheAgent(t *testing.T) {
	a := newAgent(t, nil, aitest.Says("still working"))
	rec, err := session.Create(context.Background(), &flaky{Store: &memStore{}, failOn: 1})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = rec.Close() })

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
// first recorded, and carries on after it.
func TestASecondProcessPicksUpTheConversation(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	first, err := jsonl.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	a := newAgent(t, nil, aitest.Says("morning"), aitest.Says("afternoon"))
	rec, err := session.Create(ctx, first)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = rec.Close() })
	converse(t, a, rec, ai.UserMessage("hello"))
	id := rec.ID()

	// A new process: nothing survives but the directory.
	second, err := jsonl.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	// Finding it the way a program would, rather than by remembering the id.
	sessions, err := second.List(ctx)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 1 || sessions[0].ID != id {
		t.Fatalf("List = %+v, want the one session just written", sessions)
	}

	if err := rec.Close(); err != nil {
		t.Fatal(err)
	}
	rec2, restored, err := session.Open(ctx, second, id)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = rec2.Close() })
	if len(restored) != len(a.Messages()) {
		t.Fatalf("restored %d messages, the first process ended holding %d",
			len(restored), len(a.Messages()))
	}
	if got := restored[0].Text(); got != "hello" {
		t.Errorf("the conversation starts at %q, want what was asked first", got)
	}

	// And it carries on: a second exchange lands after the first, not on top.
	b := newAgent(t, restored, aitest.Says("afternoon"))
	converse(t, b, rec2, ai.UserMessage("still there?"))

	final, err := second.Load(ctx, id)
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

// flaky fails one write and works for everything else.
type flaky struct {
	session.Store
	failOn int
	writes int
}

func (f *flaky) fail() error {
	f.writes++
	if f.writes == f.failOn {
		return errors.New("the disk, briefly")
	}
	return nil
}

func (f *flaky) Append(ctx context.Context, id string, msg ai.Message) error {
	if err := f.fail(); err != nil {
		return err
	}
	return f.Store.Append(ctx, id, msg)
}

func (f *flaky) Replace(ctx context.Context, id string, msgs []ai.Message) error {
	if err := f.fail(); err != nil {
		return err
	}
	return f.Store.Replace(ctx, id, msgs)
}

// A conversation is read back by folding it, so a hole in the middle is not a
// shorter conversation — it is a broken one. Drop the message carrying a tool
// call and the result answering it is orphaned, which is a shape no provider
// accepts and ai.Repair silently deletes. Recording therefore stops at the
// first failure: a prefix still folds.
func TestRecordingStopsAtTheFirstFailedWrite(t *testing.T) {
	ctx := context.Background()
	st := &flaky{Store: store(t), failOn: 2}

	a := newAgent(t, nil,
		aitest.Asks(ai.ToolCall{ID: "c1", Name: "noop", Input: "{}"}),
		aitest.Says("done"))
	rec, err := session.Create(ctx, st)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = rec.Close() })
	converse(t, a, rec, ai.UserMessage("go"))

	if rec.Err() == nil {
		t.Fatal("a failed write was not reported")
	}
	restored, err := st.Load(ctx, rec.ID())
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
	if st.writes != 2 {
		t.Errorf("the store saw %d writes, want 2 — recording carried on past the failure", st.writes)
	}
}

// Resuming hands the agent the history it just read, and the agent announces
// it — correctly, it did replace its conversation. Storing that would write a
// copy of the whole conversation on every resume, forever.
func TestResumingDoesNotRecordACopyOfWhatItRead(t *testing.T) {
	ctx := context.Background()
	st := store(t)

	a := newAgent(t, nil, aitest.Says("one"))
	rec, err := session.Create(ctx, st)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = rec.Close() })
	converse(t, a, rec, ai.UserMessage("first"))

	for range 5 {
		b := newAgent(t, nil, aitest.Says("again"))
		if err := rec.Close(); err != nil {
			t.Fatal(err)
		}
		rec2, history, err := session.Open(ctx, st, rec.ID())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = rec2.Close() })
		b.SetMessages(history)
		converse(t, b, rec2, ai.UserMessage("more"))
		if err := rec2.Close(); err != nil {
			t.Fatal(err)
		}
	}

	if snapshots := slices.Index(st.opsOf(rec.ID()), "replace"); snapshots >= 0 {
		t.Errorf("write %d replaced the conversation; resuming alone never does", snapshots)
	}
}

// A name survives being stored and read back, which is the point of having one
// at all: an application that records rows against these has to find the same
// rows after a restart. Both entry kinds carry it — a message that was
// appended, and a snapshot the whole conversation was replaced with.
func TestNamesSurviveTheRoundTrip(t *testing.T) {
	st := store(t)

	client := ai.NewClientWithDriver(
		aitest.New(aitest.Says("first"), aitest.Says("second")),
		ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client, agent.WithMessageIDs(counterIDs()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	rec, err := session.Create(context.Background(), st)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = rec.Close() })
	converse(t, a, rec, ai.UserMessage("hello"))

	// A compaction, so the fold has to come back through a snapshot too.
	a.SetMessages([]ai.Message{ai.UserMessage("(the summary)")})
	converse(t, a, rec, ai.UserMessage("and again"))

	restored, err := st.Load(context.Background(), rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(restored) == 0 {
		t.Fatal("nothing was restored")
	}
	for i, m := range restored {
		if m.ID == "" {
			t.Errorf("restored message %d came back unnamed", i)
		}
	}

	// And they are the agent's own names, not new ones invented on the way in.
	held := a.Messages()
	if len(held) != len(restored) {
		t.Fatalf("the agent holds %d messages and the store returned %d",
			len(held), len(restored))
	}
	for i := range held {
		if held[i].ID != restored[i].ID {
			t.Errorf("message %d is %q in the agent and %q in the store",
				i, held[i].ID, restored[i].ID)
		}
	}
}

func counterIDs() func() string {
	n := 0
	return func() string {
		n++
		return "m" + strconv.Itoa(n)
	}
}

func TestOpenedSessionKeepsExclusiveOwnershipUntilClose(t *testing.T) {
	for _, impl := range stores {
		t.Run(impl.name, func(t *testing.T) {
			ctx := context.Background()
			st := impl.open(t)
			opened, err := session.Create(ctx, st)
			if err != nil {
				t.Fatal(err)
			}
			opened.RecordEvent(ctx, agent.MessageAdded{Message: ai.UserMessage("saved")})
			if competing, _, err := session.Open(ctx, st, opened.ID()); err == nil {
				_ = competing.Close()
				t.Fatal("active session opened twice")
			}
			if err := opened.Close(); err != nil {
				t.Fatal(err)
			}
			if !errors.Is(opened.Err(), session.ErrClosed) {
				t.Fatalf("closed session Err: %v", opened.Err())
			}
			reopened, history, err := session.Open(ctx, st, opened.ID())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = reopened.Close() }()
			if len(history) != 1 || history[0].Text() != "saved" {
				t.Fatal(history)
			}
			opened.RecordEvent(ctx, agent.MessageAdded{Message: ai.UserMessage("late")})
			if err := opened.Close(); err != nil {
				t.Fatal(err)
			}
			if competing, _, err := session.Open(ctx, st, reopened.ID()); err == nil {
				_ = competing.Close()
				t.Fatal("old Close released the newer claim")
			}
			if got, err := st.Load(ctx, reopened.ID()); err != nil || len(got) != 1 {
				t.Fatalf("late event was written: %v, %v", got, err)
			}
		})
	}
}

func TestOpenFailureReleasesItsClaim(t *testing.T) {
	ctx := context.Background()
	st := store(t)
	id, err := st.Create(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := session.Open(ctx, unloadable{st}, id); err == nil {
		t.Fatal("a conversation that would not load was opened")
	}
	release, err := st.Claim(ctx, id)
	if err != nil {
		t.Fatalf("failed Open leaked claim: %v", err)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
}

// unloadable is a store whose conversation will not load.
type unloadable struct{ session.Store }

func (unloadable) Load(context.Context, string) ([]ai.Message, error) {
	return nil, errors.New("unreadable")
}
