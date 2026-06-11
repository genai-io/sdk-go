package san

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// ── Mocks ──

// mockLLM implements llm.LLM for testing.
type mockLLM struct {
	mu         sync.Mutex
	inferFunc  func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error)
	inputLimit int
}

func (m *mockLLM) Infer(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
	if m.inferFunc != nil {
		return m.inferFunc(ctx, req)
	}
	return nil, nil
}

func (m *mockLLM) InputLimit() int { return m.inputLimit }

func newMockLLM() *mockLLM { return &mockLLM{inputLimit: 200000} }

type mockTool struct {
	name        string
	description string
	schema      llm.ToolSchema
	executeFunc func(ctx context.Context, input map[string]any) (string, error)
}

func (t *mockTool) Name() string                       { return t.name }
func (t *mockTool) Description() string                { return t.description }
func (t *mockTool) Schema() llm.ToolSchema             { return t.schema }
func (t *mockTool) Execute(ctx context.Context, input map[string]any) (string, error) {
	return t.executeFunc(ctx, input)
}

func newMockTool(name, description string) *mockTool {
	return &mockTool{
		name:        name,
		description: description,
		schema: llm.ToolSchema{
			Name:        name,
			Description: description,
			Parameters:  map[string]any{"type": "object"},
		},
		executeFunc: func(ctx context.Context, input map[string]any) (string, error) {
			return name + ": done", nil
		},
	}
}

// ── Streaming Helpers for Mock LLM ──

func makeTextChunk(text string) llm.Chunk {
	return llm.Chunk{Text: text}
}

func makeDoneChunk(content string) llm.Chunk {
	return llm.Chunk{
		Done: true,
		Response: &llm.InferResponse{
			Content:    content,
			StopReason: llm.StopEndTurn,
			TokensIn:   10,
			TokensOut:  5,
		},
	}
}

func makeToolDoneChunk(content string, calls []llm.ToolCall) llm.Chunk {
	return llm.Chunk{
		Done: true,
		Response: &llm.InferResponse{
			Content:    content,
			ToolCalls:  calls,
			StopReason: llm.StopToolUse,
			TokensIn:   20,
			TokensOut:  10,
		},
	}
}

func makeErrorChunk(err error) llm.Chunk {
	return llm.Chunk{Err: err}
}

func makeCancelledDoneChunk() llm.Chunk {
	return llm.Chunk{
		Done: true,
		Response: &llm.InferResponse{
			StopReason: llm.StopCancelled,
		},
	}
}

// multiTurnMock returns an inferFunc that cycles through turn responses.
// Each element of turns is passed to streamResponse for the corresponding call.
func multiTurnMock(turns ...[]llm.Chunk) func(context.Context, llm.InferRequest) (<-chan llm.Chunk, error) {
	return multiTurnMockT(nil, turns...)
}

// testingT is a dummy interface to avoid importing testing in the mock.
type testingT interface{ Errorf(string, ...any) }

// multiTurnMockT logs if there are more calls than configured turns.
func multiTurnMockT(t testingT, turns ...[]llm.Chunk) func(context.Context, llm.InferRequest) (<-chan llm.Chunk, error) {
	var mu sync.Mutex
	call := 0
	return func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
		mu.Lock()
		idx := call
		call++
		mu.Unlock()
		if idx >= len(turns) {
			if t != nil {
				t.Errorf("unexpected Infer call %d (only %d turns configured)", idx+1, len(turns))
			}
			// Return a simple done response so the test doesn't hang
			ch := make(chan llm.Chunk, 1)
			go func() {
				defer close(ch)
				ch <- makeDoneChunk("")
			}()
			return ch, nil
		}
		return streamResponse(turns[idx]...)(ctx, req)
	}
}

func streamResponse(chunks ...llm.Chunk) func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
	return func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
		ch := make(chan llm.Chunk, len(chunks))
		go func() {
			defer close(ch)
			for _, c := range chunks {
				select {
				case ch <- c:
				case <-ctx.Done():
					return
				}
			}
		}()
		return ch, nil
	}
}

// ── Agent Construction Tests ──

func TestNew_MissingLLM(t *testing.T) {
	_, err := New(WithSystem("system prompt"))
	if err == nil {
		t.Fatal("expected error for missing llm, got nil")
	}
	if !errors.Is(err, fieldError{field: "llm"}) {
		t.Errorf("expected fieldError{llm}, got %T: %v", err, err)
	}
}

func TestNew_MissingSystem(t *testing.T) {
	_, err := New(WithLLM(newMockLLM()))
	if err == nil {
		t.Fatal("expected error for missing system, got nil")
	}
}

func TestNew_Defaults(t *testing.T) {
	l := newMockLLM()
	agent, err := New(WithLLM(l), WithSystem("you are helpful"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if agent.ID() != "agent" {
		t.Errorf("expected default id 'agent', got %q", agent.ID())
	}
	if agent.maxSteps != 0 {
		t.Errorf("expected default maxSteps 0, got %d", agent.maxSteps)
	}
}

func TestNew_WithAllOptions(t *testing.T) {
	l := newMockLLM()
	ts := NewToolSet()
	ts.Add(newMockTool("echo", "echoes input"))

	agent, err := New(
		WithLLM(l),
		WithSystem("system"),
		WithID("test-agent"),
		WithTools(ts),
		WithMaxSteps(10),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if agent.ID() != "test-agent" {
		t.Errorf("expected id 'test-agent', got %q", agent.ID())
	}
	if agent.maxSteps != 10 {
		t.Errorf("expected maxSteps 10, got %d", agent.maxSteps)
	}
	if agent.tools != ts {
		t.Error("expected tool set to be set")
	}
}

// ── Agent Accessor Tests ──

func TestAgent_ID(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"), WithID("custom-id"))
	if agent.ID() != "custom-id" {
		t.Errorf("expected 'custom-id', got %q", agent.ID())
	}
}

func TestAgent_Inbox(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"))
	if agent.Inbox() == nil {
		t.Fatal("Inbox() returned nil")
	}
}

func TestAgent_Outbox(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"))
	if agent.Outbox() == nil {
		t.Fatal("Outbox() returned nil")
	}
}

func TestAgent_Messages_InitiallyEmpty(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"))
	msgs := agent.Messages()
	if len(msgs) != 0 {
		t.Errorf("expected 0 messages, got %d", len(msgs))
	}
}

func TestAgent_SetMessages(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"))
	input := []llm.Message{
		{Content: "hello"},
		{Content: "world"},
	}
	agent.SetMessages(input)
	msgs := agent.Messages()
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Content != "hello" || msgs[1].Content != "world" {
		t.Error("SetMessages did not preserve content")
	}
	// Mutating original should not affect agent
	input[0] = llm.Message{Content: "modified"}
	msgs2 := agent.Messages()
	if msgs2[0].Content == "modified" {
		t.Error("Messages() should return a copy, not reference")
	}
}

// ── Agent Close Tests ──

func TestAgent_Close(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"))
	err := agent.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	// Verify inbox is closed — sending to a closed channel panics.
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when sending to closed inbox")
		}
	}()
	agent.Inbox() <- llm.UserMessage("ping", nil)
}

func TestAgent_Close_Idempotent(t *testing.T) {
	l := newMockLLM()
	agent, _ := New(WithLLM(l), WithSystem("s"))
	if err := agent.Close(); err != nil {
		t.Fatalf("first Close() error = %v", err)
	}
	if err := agent.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
}

// ── ThinkAct Tests ──

func TestThinkAct_SingleTurn(t *testing.T) {
	l := newMockLLM()
	l.inferFunc = streamResponse(
		makeTextChunk("Hello"),
		makeTextChunk(" world"),
		makeDoneChunk("Hello world"),
	)

	agent, _ := New(WithLLM(l), WithSystem("you are helpful"))
	agent.SetMessages([]llm.Message{llm.UserMessage("hi", nil)})

	ctx := context.Background()
	// Drain outbox in background
	go func() {
		for range agent.Outbox() {
		}
	}()

	result, err := agent.ThinkAct(ctx)
	if err != nil {
		t.Fatalf("ThinkAct() error = %v", err)
	}
	if result.Content != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", result.Content)
	}
	if result.StopReason != llm.StopEndTurn {
		t.Errorf("expected StopEndTurn, got %q", result.StopReason)
	}
	if result.Steps != 1 {
		t.Errorf("expected 1 step, got %d", result.Steps)
	}
	if result.TokensIn != 10 || result.TokensOut != 5 {
		t.Errorf("expected tokens (10,5), got (%d,%d)", result.TokensIn, result.TokensOut)
	}
}

func TestThinkAct_WithToolCalls(t *testing.T) {
	l := newMockLLM()
	callCount := 0
	l.inferFunc = func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
		callCount++
		if callCount == 1 {
			// First call: model requests tool use
			return streamResponse(
				makeToolDoneChunk("let me check", []llm.ToolCall{
					{ID: "1", Name: "echo", Input: `{"text":"hello"}`},
				}),
			)(ctx, req)
		}
		// Second call: model responds with final answer
		return streamResponse(
			makeDoneChunk("the tool says: echo: done"),
		)(ctx, req)
	}

	ts := NewToolSet()
	ts.Add(newMockTool("echo", "echoes input"))
	agent, _ := New(WithLLM(l), WithSystem("s"), WithTools(ts))
	agent.SetMessages([]llm.Message{llm.UserMessage("use echo", nil)})

	// Drain outbox
	go func() {
		for range agent.Outbox() {
		}
	}()

	result, err := agent.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct() error = %v", err)
	}
	if result.ToolUses != 1 {
		t.Errorf("expected 1 tool use, got %d", result.ToolUses)
	}
	if result.Steps != 2 {
		t.Errorf("expected 2 steps, got %d", result.Steps)
	}
	if result.StopReason != llm.StopEndTurn {
		t.Errorf("expected StopEndTurn, got %q", result.StopReason)
	}
	if callCount != 2 {
		t.Errorf("expected 2 LLM calls, got %d", callCount)
	}
}

func TestThinkAct_UnknownTool(t *testing.T) {
	l := newMockLLM()
	// Turn 1: agent gets tool_use for unknown tool, executes it (error), loops.
	// Turn 2: agent calls Infer again, gets final end_turn response.
	l.inferFunc = multiTurnMock(
		[]llm.Chunk{makeToolDoneChunk("use tool", []llm.ToolCall{
			{ID: "1", Name: "nonexistent", Input: "{}"},
		})},
		[]llm.Chunk{makeDoneChunk("done")},
	)

	agent, _ := New(WithLLM(l), WithSystem("s"))
	agent.SetMessages([]llm.Message{llm.UserMessage("do it", nil)})

	go func() {
		for range agent.Outbox() {
		}
	}()

	result, err := agent.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct() error = %v", err)
	}
	if result.ToolUses != 1 {
		t.Errorf("expected 1 tool use, got %d", result.ToolUses)
	}
	// Verify tool error message was appended
	msgs := agent.Messages()
	foundToolResult := false
	for _, m := range msgs {
		if m.Role == llm.RoleTool && m.ToolResult != nil && m.ToolResult.IsError {
			foundToolResult = true
		}
	}
	if !foundToolResult {
		t.Error("expected tool error to be recorded")
	}
}

func TestThinkAct_ToolExecutionError(t *testing.T) {
	toolErr := errors.New("execution failed")
	ts := NewToolSet()
	ts.Add(&mockTool{
		name:        "failing_tool",
		description: "always fails",
		schema:      llm.ToolSchema{Name: "failing_tool", Description: "always fails"},
		executeFunc: func(ctx context.Context, input map[string]any) (string, error) {
			return "", toolErr
		},
	})

	l := newMockLLM()
	// Turn 1: model requests failing tool, agent executes (error), loops.
	// Turn 2: model returns final response.
	l.inferFunc = multiTurnMock(
		[]llm.Chunk{makeToolDoneChunk("use tool", []llm.ToolCall{
			{ID: "1", Name: "failing_tool", Input: "{}"},
		})},
		[]llm.Chunk{makeDoneChunk("handled error")},
	)

	agent, _ := New(WithLLM(l), WithSystem("s"), WithTools(ts))
	agent.SetMessages([]llm.Message{llm.UserMessage("do it", nil)})

	go func() {
		for range agent.Outbox() {
		}
	}()

	result, err := agent.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct() error = %v", err)
	}
	if result.ToolUses != 1 {
		t.Errorf("expected 1 tool use, got %d", result.ToolUses)
	}
}

func TestThinkAct_MaxSteps(t *testing.T) {
	l := newMockLLM()
	// Each call requests a tool — causes a loop
	l.inferFunc = streamResponse(
		makeToolDoneChunk("step 1", []llm.ToolCall{
			{ID: "1", Name: "echo", Input: "{}"},
		}),
	)

	ts := NewToolSet()
	ts.Add(newMockTool("echo", "echoes"))
	agent, _ := New(WithLLM(l), WithSystem("s"), WithTools(ts), WithMaxSteps(2))
	agent.SetMessages([]llm.Message{llm.UserMessage("go", nil)})

	go func() {
		for range agent.Outbox() {
		}
	}()

	result, err := agent.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct() error = %v", err)
	}
	if result.StopReason != llm.StopMaxSteps {
		t.Errorf("expected StopMaxSteps, got %q", result.StopReason)
	}
	if result.Steps != 2 {
		t.Errorf("expected 2 steps, got %d", result.Steps)
	}
}

func TestThinkAct_PreInferError(t *testing.T) {
	l := newMockLLM()
	inferErr := errors.New("infer error")
	l.inferFunc = func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
		return nil, inferErr
	}

	agent, _ := New(WithLLM(l), WithSystem("s"))
	agent.SetMessages([]llm.Message{llm.UserMessage("hi", nil)})

	go func() {
		for range agent.Outbox() {
		}
	}()

	_, err := agent.ThinkAct(context.Background())
	if err == nil {
		t.Fatal("expected error from ThinkAct(), got nil")
	}
}

func TestThinkAct_StreamError(t *testing.T) {
	l := newMockLLM()
	l.inferFunc = streamResponse(
		makeErrorChunk(errors.New("stream error")),
	)

	agent, _ := New(WithLLM(l), WithSystem("s"))
	agent.SetMessages([]llm.Message{llm.UserMessage("hi", nil)})

	go func() {
		for range agent.Outbox() {
		}
	}()

	result, err := agent.ThinkAct(context.Background())
	if err != nil {
		t.Fatalf("ThinkAct() error = %v", err)
	}
	// Stream errors set StopReason to cancelled but don't return an error from ThinkAct
	if result.StopReason != llm.StopCancelled {
		t.Errorf("expected StopCancelled, got %q", result.StopReason)
	}
}

func TestThinkAct_ContextCancelled(t *testing.T) {
	l := newMockLLM()
	l.inferFunc = func(ctx context.Context, req llm.InferRequest) (<-chan llm.Chunk, error) {
		ch := make(chan llm.Chunk)
		go func() {
			defer close(ch)
			<-ctx.Done()
		}()
		return ch, nil
	}

	agent, _ := New(WithLLM(l), WithSystem("s"))
	agent.SetMessages([]llm.Message{llm.UserMessage("hi", nil)})

	go func() {
		for range agent.Outbox() {
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately — ThinkAct's initial ctx.Done() check catches it

	_, err := agent.ThinkAct(ctx)
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}

// ── Event Tests ──

func TestEvent_Accessors(t *testing.T) {
	chunk := llm.Chunk{Text: "text"}
	evt := Event{Type: OnChunk, Source: "agent", Data: chunk}
	gotChunk, ok := evt.Chunk()
	if !ok || gotChunk.Text != "text" {
		t.Error("Chunk() accessor failed")
	}

	result := Result{Content: "result"}
	evt = Event{Type: OnTurn, Source: "agent", Data: result}
	gotResult, ok := evt.Result()
	if !ok || gotResult.Content != "result" {
		t.Error("Result() accessor failed")
	}

	tc := llm.ToolCall{ID: "1", Name: "tool"}
	evt = Event{Type: PreTool, Source: "tool", Data: tc}
	gotTC, ok := evt.ToolCall()
	if !ok || gotTC.ID != "1" {
		t.Error("ToolCall() accessor failed")
	}

	tr := llm.ToolResult{ToolCallID: "1", Content: "ok"}
	evt = Event{Type: PostTool, Source: "tool", Data: tr}
	gotTR, ok := evt.ToolResult()
	if !ok || gotTR.Content != "ok" {
		t.Error("ToolResult() accessor failed")
	}

	msg := llm.Message{Content: "hello"}
	evt = Event{Type: OnMessage, Source: "agent", Data: msg}
	gotMsg, ok := evt.Message()
	if !ok || gotMsg.Content != "hello" {
		t.Error("Message() accessor failed")
	}

	evt = Event{Type: OnStop, Source: "agent", Data: errors.New("stop")}
	_, ok = evt.Error()
	if !ok {
		t.Error("Error() accessor failed")
	}
}

func TestEvent_Accessors_WrongType(t *testing.T) {
	evt := Event{Type: OnChunk, Source: "agent", Data: "wrong type"}
	_, ok := evt.Chunk()
	if ok {
		t.Error("Chunk() should return false for wrong type")
	}
	_, ok = evt.Result()
	if ok {
		t.Error("Result() should return false for wrong type")
	}
}

func TestEventTypeConstants(t *testing.T) {
	if OnStart != "AgentStart" {
		t.Errorf("OnStart = %q", OnStart)
	}
	if OnStop != "AgentStop" {
		t.Errorf("OnStop = %q", OnStop)
	}
	if OnChunk != "Chunk" {
		t.Errorf("OnChunk = %q", OnChunk)
	}
	if OnTurn != "Turn" {
		t.Errorf("OnTurn = %q", OnTurn)
	}
	if PreInfer != "PreInfer" {
		t.Errorf("PreInfer = %q", PreInfer)
	}
	if PostInfer != "PostInfer" {
		t.Errorf("PostInfer = %q", PostInfer)
	}
	if PreTool != "PreTool" {
		t.Errorf("PreTool = %q", PreTool)
	}
	if PostTool != "PostTool" {
		t.Errorf("PostTool = %q", PostTool)
	}
}

// ── ToolSet Tests ──

func TestToolSet_Add_Get(t *testing.T) {
	ts := NewToolSet()
	if ts.Get("echo") != nil {
		t.Error("expected nil for non-existent tool")
	}
	to := newMockTool("echo", "echo input")
	ts.Add(to)
	if ts.Get("echo") != to {
		t.Error("Get() returned wrong tool")
	}
}

func TestToolSet_Add_Overwrite(t *testing.T) {
	ts := NewToolSet()
	ts.Add(newMockTool("echo", "v1"))
	ts.Add(newMockTool("echo", "v2"))
	if ts.Get("echo").Description() != "v2" {
		t.Error("Add() should overwrite")
	}
}

func TestToolSet_Remove(t *testing.T) {
	ts := NewToolSet()
	ts.Add(newMockTool("echo", "echo"))
	ts.Remove("echo")
	if ts.Get("echo") != nil {
		t.Error("expected nil after Remove()")
	}
	ts.Remove("nonexistent") // no-op
}

func TestToolSet_Schemas(t *testing.T) {
	ts := NewToolSet()
	ts.Add(newMockTool("a", "tool a"))
	ts.Add(newMockTool("b", "tool b"))

	schemas := ts.Schemas()
	if len(schemas) != 2 {
		t.Errorf("expected 2 schemas, got %d", len(schemas))
	}
	names := make(map[string]bool)
	for _, s := range schemas {
		names[s.Name] = true
	}
	if !names["a"] || !names["b"] {
		t.Error("schemas missing expected names")
	}
}

func TestToolSet_Schemas_Empty(t *testing.T) {
	ts := NewToolSet()
	schemas := ts.Schemas()
	if len(schemas) != 0 {
		t.Errorf("expected 0 schemas, got %d", len(schemas))
	}
}

// ── fieldError Tests ──

func TestFieldError(t *testing.T) {
	err := errMissingField("llm")
	if err.Error() != "required field missing: llm" {
		t.Errorf("expected 'required field missing: llm', got %q", err.Error())
	}
}

func TestFieldError_Is(t *testing.T) {
	err := errMissingField("llm")
	var fe fieldError
	if !errors.As(err, &fe) {
		t.Error("expected fieldError type")
	}
	if fe.field != "llm" {
		t.Errorf("expected field 'llm', got %q", fe.field)
	}
}

// ── Tool Interface Tests ──

func TestMockTool_Interface(t *testing.T) {
	var _ Tool = (*mockTool)(nil) // compile-time check
	tool := newMockTool("test", "test description")
	if tool.Name() != "test" {
		t.Errorf("Name() = %q", tool.Name())
	}
	if tool.Description() != "test description" {
		t.Errorf("Description() = %q", tool.Description())
	}
	content, err := tool.Execute(context.Background(), map[string]any{"key": "value"})
	if err != nil {
		t.Errorf("Execute() error = %v", err)
	}
	if content != "test: done" {
		t.Errorf("Execute() = %q", content)
	}
	schema := tool.Schema()
	if schema.Name != "test" {
		t.Errorf("Schema().Name = %q", schema.Name)
	}
}
