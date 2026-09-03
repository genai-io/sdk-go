package session_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Compaction replaces the conversation, and a session that only folded what
// was appended would restore what the agent threw away. Nothing here says so:
// replacing the conversation is an event like any other, and the recorder
// consuming the stream already has it. That is the point of the test — a
// caller who forgets a step cannot exist if there is no step to forget.
func TestCompactionSurvivesARestore(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	a := newAgent(t, nil, text("one"), text("two"), text("three"))
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("first"), ai.UserMessage("second"))

	summary := []ai.Message{ai.UserMessage("(summary of the above)")}
	a.SetMessages(summary)

	converse(t, a, rec, ai.UserMessage("third"))

	restored, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(restored) != len(a.Messages()) {
		t.Fatalf("restored %d messages, agent holds %d — the compaction did not survive",
			len(restored), len(a.Messages()))
	}
	if got := restored[0].Text(); got != "(summary of the above)" {
		t.Errorf("the restore starts at %q, want the summary", got)
	}
}

// A compaction made during the last exchange of a session is recorded like any
// other. The announcement used to wait for the next exchange, and when there
// was none — the person quit, the run ended — the snapshot was never written:
// the session kept the messages appended after the compaction and none of the
// news that everything before them had been thrown away.
func TestACompactionInTheLastExchangeIsRecorded(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	var a *agent.Agent
	compact := agent.ToolFunc("compact", "Replace the conversation.",
		func(context.Context, struct{}) (agent.Result, error) {
			a.SetMessages([]ai.Message{ai.UserMessage("(the summary)")})
			return agent.TextResult("compacted"), nil
		})
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "compact", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("done"),
	}}, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client, agent.WithTools(compact))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("go"))

	restored, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(restored) == 0 {
		t.Fatal("nothing was restored: the compaction went unrecorded")
	}
	if got := restored[0].Text(); got != "(the summary)" {
		t.Fatalf("the restore starts at %q, want the compacted conversation", got)
	}
	if len(restored) != len(a.Messages()) {
		t.Errorf("restored %d messages, the agent holds %d", len(restored), len(a.Messages()))
	}
}

// Compaction through a PreStep hook is recorded at the step that made it: the
// snapshot goes in before the messages of the step that ran against it, so a
// fold never passes through a conversation the agent had already discarded.
func TestAPreStepCompactionIsRecordedWhereItHappened(t *testing.T) {
	st := store(t)
	ctx := context.Background()

	done := false
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "noop", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("done"),
	}}, ai.Model{ID: "stub", API: "stub"})

	a, err := agent.New(client, agent.WithHooks(agent.Hook{
		PreStep: func(_ context.Context, c agent.PreStepContext) ([]ai.Message, error) {
			if done {
				return nil, nil
			}
			done = true
			return []ai.Message{ai.UserMessage("(the summary)")}, nil
		},
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rec, _, err := session.Open(ctx, st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("go"))

	var snapshot, after int
	snapshot = -1
	for e, err := range st.Entries(ctx, rec.ID()) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		switch {
		case e.Type == session.EntrySnapshot:
			snapshot = int(e.Seq)
		case e.Type == session.EntryMessage && snapshot >= 0:
			after++
		}
	}
	if snapshot < 0 {
		t.Fatal("the compaction was never recorded")
	}
	if after == 0 {
		t.Error("the snapshot is the last thing in the session; the step that ran against it recorded nothing")
	}

	restored, err := session.Messages(ctx, st, rec.ID())
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}
	if len(restored) == 0 || restored[0].Text() != "(the summary)" {
		t.Fatalf("the restore holds %d messages, want the compacted conversation", len(restored))
	}
	if len(restored) != len(a.Messages()) {
		t.Errorf("restored %d messages, the agent holds %d", len(restored), len(a.Messages()))
	}
}

// The shape both examples in this repository have: open a session, hand the
// agent whatever came back, answer. What comes back from a new session is
// nothing, and a second run still has to reopen what the first wrote.
func TestASessionSeededWithItsOwnEmptyHistoryReopens(t *testing.T) {
	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", store},
		{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
	} {
		t.Run(impl.name, func(t *testing.T) {
			st := impl.open(t)
			ctx := context.Background()

			rec, history, err := session.Open(ctx, st, "")
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			first := newAgent(t, nil, text("hello"))
			first.SetMessages(history) // what the examples do, verbatim
			converse(t, first, rec, ai.UserMessage("hi"))
			if err := rec.Err(); err != nil {
				t.Fatalf("recording failed: %v", err)
			}

			rec2, restored, err := session.Open(ctx, st, rec.ID())
			if err != nil {
				t.Fatalf("the session the first run wrote will not reopen: %v", err)
			}
			if len(restored) != 2 {
				t.Fatalf("restored %d messages, want the 2 that were recorded", len(restored))
			}

			second := newAgent(t, nil, text("still here"))
			second.SetMessages(restored)
			converse(t, second, rec2, ai.UserMessage("again"))

			all, err := session.Messages(ctx, st, rec.ID())
			if err != nil {
				t.Fatalf("Messages: %v", err)
			}
			if len(all) != 4 {
				t.Errorf("the session holds %d messages, want 4", len(all))
			}
		})
	}
}

// Clearing a conversation is a state a session has to hold: an empty snapshot
// says everything before it is gone, which is not a record that carries
// nothing. Both stores, because it is the wire format that loses the difference.
func TestAClearedConversationRestoresAsCleared(t *testing.T) {
	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", store},
		{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
	} {
		t.Run(impl.name, func(t *testing.T) {
			st := impl.open(t)
			ctx := context.Background()

			a := newAgent(t, nil, text("one"), text("two"))
			rec, _, err := session.Open(ctx, st, "")
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			converse(t, a, rec, ai.UserMessage("first"))

			a.SetMessages(nil)
			converse(t, a, rec, ai.UserMessage("starting over"))

			restored, err := session.Messages(ctx, st, rec.ID())
			if err != nil {
				t.Fatalf("Messages: %v", err)
			}
			if len(restored) != len(a.Messages()) {
				t.Fatalf("restored %d messages, the agent holds %d — what was cleared came back",
					len(restored), len(a.Messages()))
			}
			if got := restored[0].Text(); got != "starting over" {
				t.Errorf("the restore starts at %q, want the first thing said after the clearing", got)
			}
		})
	}
}

// detailsAgent runs one tool that answers with details, which is the whole
// setup the three tests below need.
func detailsAgent(t *testing.T, details any) *agent.Agent {
	t.Helper()
	tool := agent.ToolFunc("run", "Do a thing.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.Result{Content: ai.TextContent("ok"), Details: details}, nil
		})
	client := ai.NewClientWithDriver(&scripted{scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "run", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("done"),
	}}, ai.Model{ID: "stub", API: "stub"})
	a, err := agent.New(client, agent.WithTools(tool))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

// toolRun reads back the one tool run a session recorded.
func toolRun(t *testing.T, st session.Store, id string) *session.ToolRun {
	t.Helper()
	var run *session.ToolRun
	for e, err := range st.Entries(context.Background(), id) {
		if err != nil {
			t.Fatalf("Entries: %v", err)
		}
		if e.Type == session.EntryToolRun {
			run = e.ToolRun
		}
	}
	if run == nil {
		t.Fatal("the tool run was not recorded")
	}
	return run
}

// A tool's structured answer is kept when a session asks for it, and comes
// back byte for byte: what an interface redrawing a transcript needs, and what
// it joins to the restored conversation by the call's own ID. Both stores,
// because it is the wire format that would lose it.
func TestAToolsDetailsAreKeptWhenASessionAsksForThem(t *testing.T) {
	type edit struct {
		Path  string `json:"path"`
		Added int    `json:"added"`
	}

	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", store},
		{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
	} {
		t.Run(impl.name, func(t *testing.T) {
			st := impl.open(t)
			a := detailsAgent(t, edit{Path: "main.go", Added: 12})

			var sawTool string
			rec, _, err := session.Open(context.Background(), st, "", session.WithToolDetails(
				func(e agent.ToolEnd) any {
					sawTool = e.Name // the whole event, not the value alone
					return e.Result.Details
				}))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			converse(t, a, rec, ai.UserMessage("go"))

			if sawTool != "run" {
				t.Errorf("the encoder was told the tool was %q", sawTool)
			}
			run := toolRun(t, st, rec.ID())
			var got edit
			if err := json.Unmarshal(run.Details, &got); err != nil {
				t.Fatalf("the details did not survive: %v (%s)", err, run.Details)
			}
			if got.Path != "main.go" || got.Added != 12 {
				t.Errorf("details = %+v, want what the tool produced", got)
			}
			// And it is joined to the conversation by the call it answers.
			if run.ID != "c1" {
				t.Errorf("the record is keyed %q, want the tool call's own ID", run.ID)
			}
		})
	}
}

// Without the option nothing is kept, which is the default because a
// structured answer is an interface's and has no size a session can assume.
func TestAToolsDetailsAreNotKeptByDefault(t *testing.T) {
	st := store(t)
	a := detailsAgent(t, map[string]any{"rows": 10_000})

	rec, _, err := session.Open(context.Background(), st, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("go"))

	if run := toolRun(t, st, rec.ID()); len(run.Details) > 0 {
		t.Errorf("a session that asked for nothing kept %s", run.Details)
	}
}

// A value that will not marshal loses its own payload and nothing else. It is
// what an interface draws; what a restore needs is the conversation, and
// stopping the session over the first would lose the second.
func TestABadDetailsPayloadKeepsTheSessionRecording(t *testing.T) {
	st := store(t)
	a := detailsAgent(t, nil)

	rec, _, err := session.Open(context.Background(), st, "", session.WithToolDetails(
		func(agent.ToolEnd) any { return json.RawMessage("{not json") }))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("go"))

	if err := rec.Err(); err != nil {
		t.Fatalf("recording stopped over a payload nobody replays: %v", err)
	}
	if run := toolRun(t, st, rec.ID()); len(run.Details) > 0 {
		t.Errorf("the unwritable payload was stored as %s", run.Details)
	}
	restored, err := session.Messages(context.Background(), st, rec.ID())
	if err != nil || len(restored) == 0 {
		t.Fatalf("the conversation no longer folds: %d messages, %v", len(restored), err)
	}
}

// Returning nil keeps nothing, and does not keep the JSON null it would
// otherwise marshal to: a record of nothing is not what the caller asked for.
func TestReturningNoDetailsKeepsNone(t *testing.T) {
	st := store(t)
	a := detailsAgent(t, map[string]any{"rows": 10_000})

	rec, _, err := session.Open(context.Background(), st, "", session.WithToolDetails(
		func(agent.ToolEnd) any { return nil }))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	converse(t, a, rec, ai.UserMessage("go"))

	if run := toolRun(t, st, rec.ID()); len(run.Details) > 0 {
		t.Errorf("an encoder that kept nothing stored %s", run.Details)
	}
}
