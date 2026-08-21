package san_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/llmtest"
	"github.com/genai-io/sdk-go/pkg/san"
)

// ── helpers ──

type fakeTool struct {
	name string
	run  func(ctx context.Context, input map[string]any) (string, error)
}

func (t *fakeTool) Name() string        { return t.name }
func (t *fakeTool) Description() string { return t.name + " tool" }
func (t *fakeTool) Schema() llm.Tool {
	return llm.Tool{Name: t.name, Description: t.name, Parameters: map[string]any{"type": "object"}}
}
func (t *fakeTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	return t.run(ctx, input)
}

func toolSet(tools ...san.Tool) *san.ToolSet {
	ts := san.NewToolSet()
	for _, t := range tools {
		ts.Add(t)
	}
	return ts
}

func newAgent(t *testing.T, drv *llmtest.Driver, opts ...san.AgentOption) *san.Agent {
	t.Helper()
	base := []san.AgentOption{san.WithModel(llmtest.Client(drv)), san.WithSystem("be helpful")}
	a, err := san.New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

// ── construction ──

func TestNewRequiresModel(t *testing.T) {
	if _, err := san.New(san.WithSystem("hi")); err == nil {
		t.Fatal("expected an error without a model")
	}
}

func TestNewRequiresSystem(t *testing.T) {
	if _, err := san.New(san.WithModel(llmtest.Client(llmtest.Text("x")))); err == nil {
		t.Fatal("expected an error without a system prompt")
	}
}

func TestNewAppliesOptions(t *testing.T) {
	a := newAgent(t, llmtest.Text("hi"), san.WithID("worker"), san.WithMaxSteps(3))
	if a.ID() != "worker" {
		t.Errorf("ID = %q", a.ID())
	}
	if len(a.Messages()) != 0 {
		t.Errorf("a new agent should start with no history: %+v", a.Messages())
	}
}

func TestSetMessagesCopies(t *testing.T) {
	a := newAgent(t, llmtest.Text("hi"))
	msgs := []llm.Message{llm.User("one")}
	a.SetMessages(msgs)

	msgs[0] = llm.User("mutated")
	if got := a.Messages(); got[0].Text() != "one" {
		t.Errorf("history aliases the caller's slice: %q", got[0].Text())
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	a := newAgent(t, llmtest.Text("hi"))
	if err := a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// ── the think-act loop ──

func TestThinkActSingleTurn(t *testing.T) {
	a := newAgent(t, llmtest.Text("the answer"))
	a.SetMessages([]llm.Message{llm.User("a question")})

	result, err := a.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	if result.Content != "the answer" {
		t.Errorf("Content = %q", result.Content)
	}
	if result.Steps != 1 || result.ToolUses != 0 {
		t.Errorf("result = %+v", result)
	}
	if result.StopReason != llm.StopEndTurn {
		t.Errorf("StopReason = %q", result.StopReason)
	}
}

func TestThinkActRunsToolsAndContinues(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{
			{ToolCall: &llm.ToolCall{ID: "1", Name: "echo", Input: `{"text":"hello"}`}},
			{StopReason: llm.StopToolUse, Usage: &llm.Usage{Input: 10, Output: 2}},
		}},
		{Deltas: []llm.Delta{
			{Text: "echo said hello"},
			{StopReason: llm.StopEndTurn, Usage: &llm.Usage{Input: 20, Output: 3}},
		}},
	}}

	var gotInput map[string]any
	tool := &fakeTool{name: "echo", run: func(_ context.Context, in map[string]any) (string, error) {
		gotInput = in
		return "hello", nil
	}}

	a := newAgent(t, drv, san.WithTools(toolSet(tool)))
	a.SetMessages([]llm.Message{llm.User("say hello")})

	result, err := a.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	if result.Steps != 2 || result.ToolUses != 1 {
		t.Errorf("result = %+v", result)
	}
	// Tool arguments must be decoded as real JSON, not stuffed into a
	// placeholder key.
	if gotInput["text"] != "hello" {
		t.Errorf("tool input = %+v", gotInput)
	}
	// Usage accumulates across every step of the turn.
	if want := (llm.Usage{Input: 30, Output: 5}); result.Usage != want {
		t.Errorf("Usage = %+v, want %+v", result.Usage, want)
	}

	// The history must pair the assistant's call with a single results turn,
	// or the next request is rejected.
	history := a.Messages()
	if len(history) != 3 {
		t.Fatalf("history = %d messages: %+v", len(history), history)
	}
	if len(history[1].ToolCalls) != 1 {
		t.Errorf("assistant turn lost its tool calls: %+v", history[1])
	}
	if !history[2].IsToolResult() || len(history[2].ToolResults) != 1 {
		t.Errorf("results turn = %+v", history[2])
	}
}

func TestSeveralToolResultsShareOneTurn(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{
			{ToolCall: &llm.ToolCall{ID: "1", Name: "a", Input: "{}"}},
			{ToolCall: &llm.ToolCall{ID: "2", Name: "a", Input: "{}"}},
			{StopReason: llm.StopToolUse},
		}},
		{Deltas: []llm.Delta{{Text: "done"}, {StopReason: llm.StopEndTurn}}},
	}}
	tool := &fakeTool{name: "a", run: func(context.Context, map[string]any) (string, error) {
		return "ok", nil
	}}

	a := newAgent(t, drv, san.WithTools(toolSet(tool)))
	a.SetMessages([]llm.Message{llm.User("do both")})
	if _, err := a.ThinkAct(context.Background()); err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}

	history := a.Messages()
	last := history[len(history)-1]
	if !last.IsToolResult() || len(last.ToolResults) != 2 {
		t.Errorf("both results should ride in one turn: %+v", history)
	}
}

func TestUnknownToolBecomesAToolError(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{
			{ToolCall: &llm.ToolCall{ID: "1", Name: "nope", Input: "{}"}},
			{StopReason: llm.StopToolUse},
		}},
		{Deltas: []llm.Delta{{Text: "sorry"}, {StopReason: llm.StopEndTurn}}},
	}}
	a := newAgent(t, drv)
	a.SetMessages([]llm.Message{llm.User("go")})

	// The model asking for a tool that does not exist is its mistake to
	// correct, not a reason to fail the turn.
	result, err := a.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	if result.Content != "sorry" {
		t.Errorf("Content = %q", result.Content)
	}
	res := a.Messages()[2].ToolResults[0]
	if !res.IsError || !strings.Contains(res.Content, "unknown tool") {
		t.Errorf("result = %+v", res)
	}
}

func TestToolFailureIsReportedToTheModel(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{
			{ToolCall: &llm.ToolCall{ID: "1", Name: "boom", Input: "{}"}},
			{StopReason: llm.StopToolUse},
		}},
		{Deltas: []llm.Delta{{Text: "recovered"}, {StopReason: llm.StopEndTurn}}},
	}}
	tool := &fakeTool{name: "boom", run: func(context.Context, map[string]any) (string, error) {
		return "", errors.New("disk on fire")
	}}
	a := newAgent(t, drv, san.WithTools(toolSet(tool)))
	a.SetMessages([]llm.Message{llm.User("go")})

	if _, err := a.ThinkAct(context.Background()); err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	res := a.Messages()[2].ToolResults[0]
	if !res.IsError || res.Content != "disk on fire" {
		t.Errorf("result = %+v", res)
	}
}

func TestMalformedToolInputIsReportedToTheModel(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{
			{ToolCall: &llm.ToolCall{ID: "1", Name: "echo", Input: `{"text":`}},
			{StopReason: llm.StopToolUse},
		}},
		{Deltas: []llm.Delta{{Text: "retrying"}, {StopReason: llm.StopEndTurn}}},
	}}
	var ran bool
	tool := &fakeTool{name: "echo", run: func(context.Context, map[string]any) (string, error) {
		ran = true
		return "", nil
	}}
	a := newAgent(t, drv, san.WithTools(toolSet(tool)))
	a.SetMessages([]llm.Message{llm.User("go")})

	if _, err := a.ThinkAct(context.Background()); err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	if ran {
		t.Error("the tool ran with arguments that did not parse")
	}
	res := a.Messages()[2].ToolResults[0]
	if !res.IsError || !strings.Contains(res.Content, "not valid JSON") {
		t.Errorf("result = %+v", res)
	}
	// The message names the tool, so a turn with six calls stays readable.
	if !strings.Contains(res.Content, "echo") {
		t.Errorf("result = %+v, want the tool named", res)
	}
}

func TestMaxStepsBoundsTheTurn(t *testing.T) {
	// A model that only ever asks for another tool call.
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{Deltas: []llm.Delta{
		{ToolCall: &llm.ToolCall{ID: "1", Name: "loop", Input: "{}"}},
		{StopReason: llm.StopToolUse},
	}}}}
	tool := &fakeTool{name: "loop", run: func(context.Context, map[string]any) (string, error) {
		return "again", nil
	}}
	a := newAgent(t, drv, san.WithTools(toolSet(tool)), san.WithMaxSteps(3))
	a.SetMessages([]llm.Message{llm.User("go")})

	result, err := a.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	if result.Steps != 3 || result.StopReason != san.StopMaxSteps {
		t.Errorf("result = %+v", result)
	}
}

func TestInferErrorFailsTheTurn(t *testing.T) {
	want := errors.New("upstream is down")
	a := newAgent(t, llmtest.Fail(want))
	a.SetMessages([]llm.Message{llm.User("go")})

	result, err := a.ThinkAct(context.Background())
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	if result.StopReason != llm.StopError {
		t.Errorf("StopReason = %q", result.StopReason)
	}
}

func TestCancelledContextStopsTheTurn(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	a := newAgent(t, llmtest.Text("never sent"))
	a.SetMessages([]llm.Message{llm.User("go")})

	result, err := a.ThinkAct(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if result.StopReason != san.StopCancelled {
		t.Errorf("StopReason = %q", result.StopReason)
	}
}

func TestRequestCarriesSystemPromptAndTools(t *testing.T) {
	drv := llmtest.Text("ok")
	tool := &fakeTool{name: "echo", run: func(context.Context, map[string]any) (string, error) {
		return "", nil
	}}
	a := newAgent(t, drv, san.WithTools(toolSet(tool)))
	a.SetMessages([]llm.Message{llm.User("go")})

	if _, err := a.ThinkAct(context.Background()); err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}
	req := drv.Last().Prompt
	if req.System != "be helpful" {
		t.Errorf("System = %q", req.System)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "echo" {
		t.Errorf("Tools = %+v", req.Tools)
	}
}

// ── Run loop and events ──

func TestRunEmitsLifecycleEvents(t *testing.T) {
	a := newAgent(t, llmtest.Text("hi there"))

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		a.Inbox() <- llm.User("hello")
	}()
	go func() { _ = a.Run(ctx) }()

	seen := map[san.EventType]bool{}
	for {
		select {
		case evt, ok := <-a.Outbox():
			if !ok {
				t.Fatalf("outbox closed before the turn completed; saw %v", seen)
			}
			seen[evt.Type] = true
			if evt.Type == san.OnTurn {
				result, ok := evt.Result()
				if !ok || result.Content != "hi there" {
					t.Errorf("turn event = %+v", evt)
				}
				for _, want := range []san.EventType{san.OnStart, san.PreInfer, san.PostInfer} {
					if !seen[want] {
						t.Errorf("missing %q; saw %v", want, seen)
					}
				}
				return
			}
		case <-ctx.Done():
			t.Fatalf("timed out; saw %v", seen)
		}
	}
}

func TestEventAccessors(t *testing.T) {
	chunk := san.Event{Type: san.OnChunk, Data: llm.Event{Type: llm.EventTextDelta, Text: "x"}}
	if c, ok := chunk.Chunk(); !ok || c.Text != "x" {
		t.Errorf("Chunk() = %+v, %v", c, ok)
	}
	if _, ok := chunk.Result(); ok {
		t.Error("Result() should not match a chunk event")
	}

	call := san.Event{Type: san.PreTool, Data: llm.ToolCall{Name: "ls"}}
	if tc, ok := call.ToolCall(); !ok || tc.Name != "ls" {
		t.Errorf("ToolCall() = %+v, %v", tc, ok)
	}

	failed := san.Event{Type: san.OnStop, Data: errors.New("nope")}
	if err, ok := failed.Error(); !ok || err == nil {
		t.Errorf("Error() = %v, %v", err, ok)
	}
}

// ── ToolSet ──

func TestToolSet(t *testing.T) {
	ts := san.NewToolSet()
	if len(ts.Schemas()) != 0 {
		t.Error("a new set should be empty")
	}

	first := &fakeTool{name: "a"}
	ts.Add(first)
	if ts.Get("a") != san.Tool(first) {
		t.Error("Add/Get round-trip failed")
	}

	replacement := &fakeTool{name: "a"}
	ts.Add(replacement)
	if ts.Get("a") != san.Tool(replacement) {
		t.Error("adding the same name should replace")
	}
	if len(ts.Schemas()) != 1 {
		t.Errorf("schemas = %+v", ts.Schemas())
	}

	ts.Remove("a")
	if ts.Get("a") != nil {
		t.Error("Remove left the tool behind")
	}
	ts.Remove("missing") // must not panic
}

// A turn that streamed text and burned tokens before failing must still report
// both — the spend happened whether or not the turn finished.
func TestFailedTurnStillAccountsForSpend(t *testing.T) {
	want := errors.New("connection reset")
	drv := &llmtest.Driver{Turns: []llmtest.Turn{{
		Deltas: []llm.Delta{
			{Text: "I was part way thr"},
			{Usage: &llm.Usage{Input: 3_000, Output: 40}},
		},
		Err: want,
	}}}

	a := newAgent(t, drv)
	a.SetMessages([]llm.Message{llm.User("go")})

	result, err := a.ThinkAct(context.Background())
	if !errors.Is(err, want) {
		t.Fatalf("err = %v", err)
	}
	if result.Usage.Input != 3_000 || result.Usage.Output != 40 {
		t.Errorf("Usage = %+v, want the tokens already billed", result.Usage)
	}
	if result.Steps != 1 {
		t.Errorf("Steps = %d, want the failed step counted", result.Steps)
	}
	if result.Content != "I was part way thr" {
		t.Errorf("Content = %q, want the text that arrived", result.Content)
	}
	if result.StopReason != llm.StopError {
		t.Errorf("StopReason = %q", result.StopReason)
	}
}

// A tool that declares a schema should not be run with arguments that violate
// it: a deletion with an empty path is worse than a tool error the model can
// correct.
func TestToolArgumentsAreValidatedBeforeExecution(t *testing.T) {
	type deleteArgs struct {
		Path string `json:"path"`
	}
	var ran bool
	tool := &schemaTool{
		Tool: llm.ToolFor[deleteArgs]("delete", "delete a file"),
		run: func(context.Context, map[string]any) (string, error) {
			ran = true
			return "deleted", nil
		},
	}

	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{
			// The required path is missing.
			{ToolCall: &llm.ToolCall{ID: "1", Name: "delete", Input: `{}`}},
			{StopReason: llm.StopToolUse},
		}},
		{Deltas: []llm.Delta{{Text: "sorry, retrying"}, {StopReason: llm.StopEndTurn}}},
	}}

	a := newAgent(t, drv, san.WithTools(toolSet(tool)))
	a.SetMessages([]llm.Message{llm.User("delete something")})
	if _, err := a.ThinkAct(context.Background()); err != nil {
		t.Fatalf("ThinkAct: %v", err)
	}

	if ran {
		t.Error("the tool ran with arguments its own schema rejects")
	}
	res := a.Messages()[2].ToolResults[0]
	if !res.IsError || !strings.Contains(res.Content, "path is required") {
		t.Errorf("result = %+v, want the schema violation reported back", res)
	}
}

// schemaTool carries a real llm.Tool so its declared schema is what gets
// checked.
type schemaTool struct {
	llm.Tool
	run func(ctx context.Context, input map[string]any) (string, error)
}

func (t *schemaTool) Name() string        { return t.Tool.Name }
func (t *schemaTool) Description() string { return t.Tool.Description }
func (t *schemaTool) Schema() llm.Tool    { return t.Tool }
func (t *schemaTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	return t.run(ctx, input)
}
