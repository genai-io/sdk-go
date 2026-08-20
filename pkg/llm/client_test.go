package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
)

// mockProvider implements Provider for testing.
type mockProvider struct {
	mu         sync.Mutex
	name       string
	models     []ModelInfo
	streamFunc func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk
}

func (m *mockProvider) Name() string { return m.name }

func (m *mockProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.models, nil
}

func (m *mockProvider) Stream(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
	if m.streamFunc != nil {
		return m.streamFunc(ctx, opts)
	}
	return nil
}

func newMockProvider(name string) *mockProvider {
	return &mockProvider{name: name}
}

func makeTextChunk(text string) StreamChunk {
	return StreamChunk{Type: ChunkTypeText, Text: text}
}

func makeDoneChunk(content string) StreamChunk {
	return StreamChunk{
		Type: ChunkTypeDone,
		Response: &CompletionResponse{
			Content:    content,
			StopReason: "end_turn",
			Usage: Usage{
				InputTokens:  100,
				OutputTokens: 50,
			},
		},
	}
}

func makeDoneChunkWithTools(content string, calls []ToolCall) StreamChunk {
	return StreamChunk{
		Type: ChunkTypeDone,
		Response: &CompletionResponse{
			Content:    content,
			ToolCalls:  calls,
			StopReason: "tool_use",
			Usage: Usage{
				InputTokens:  200,
				OutputTokens: 100,
			},
		},
	}
}

func makeErrorChunk(err error) StreamChunk {
	return StreamChunk{Type: ChunkTypeError, Error: err}
}

func makeThinkingChunk(text string) StreamChunk {
	return StreamChunk{Type: ChunkTypeThinking, Text: text}
}

// ── Client Tests ──

func TestClient_Infer(t *testing.T) {
	p := newMockProvider("test-provider")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 4)
		go func() {
			defer close(ch)
			ch <- makeTextChunk("Hello ")
			ch <- makeTextChunk("World")
			ch <- makeDoneChunk("Hello World")
		}()
		return ch
	}

	c := NewClient(p, "test-model", 0)
	ctx := context.Background()
	ch, err := c.Infer(ctx, InferRequest{
		System:   "You are helpful.",
		Messages: []Message{UserMessage("hi", nil)},
	})
	if err != nil {
		t.Fatalf("Infer() error = %v", err)
	}

	var text string
	var done bool
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("unexpected error chunk: %v", chunk.Err)
		}
		if chunk.Done {
			done = true
			if chunk.Response == nil {
				t.Fatal("done chunk has nil Response")
			}
			if chunk.Response.Content != "Hello World" {
				t.Errorf("expected 'Hello World', got %q", chunk.Response.Content)
			}
			if chunk.Response.TokensIn != 100 || chunk.Response.TokensOut != 50 {
				t.Errorf("expected tokens (100, 50), got (%d, %d)",
					chunk.Response.TokensIn, chunk.Response.TokensOut)
			}
		}
		text += chunk.Text
	}
	if !done {
		t.Fatal("stream closed without done chunk")
	}
	if text != "Hello World" {
		t.Errorf("expected text 'Hello World', got %q", text)
	}
}

func TestClient_Infer_Error(t *testing.T) {
	testErr := errors.New("provider error")
	p := newMockProvider("test-provider")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 2)
		go func() {
			defer close(ch)
			ch <- makeTextChunk("partial ")
			ch <- makeErrorChunk(testErr)
		}()
		return ch
	}

	c := NewClient(p, "test-model", 0)
	ctx := context.Background()
	ch, _ := c.Infer(ctx, InferRequest{
		Messages: []Message{UserMessage("hi", nil)},
	})

	var gotErr error
	for chunk := range ch {
		if chunk.Err != nil {
			gotErr = chunk.Err
		}
	}
	if gotErr == nil {
		t.Fatal("expected error in stream, got nil")
	}
	if !errors.Is(gotErr, testErr) {
		t.Errorf("expected %v, got %v", testErr, gotErr)
	}
}

func TestClient_Complete(t *testing.T) {
	p := newMockProvider("test-provider")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 2)
		go func() {
			defer close(ch)
			ch <- makeTextChunk("response")
			ch <- makeDoneChunk("response")
		}()
		return ch
	}

	c := NewClient(p, "test-model", 0)
	ctx := context.Background()
	resp, err := c.Complete(ctx, InferRequest{
		System:   "system",
		Messages: []Message{UserMessage("hello", nil)},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.Content != "response" {
		t.Errorf("expected 'response', got %q", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("expected 'end_turn', got %q", resp.StopReason)
	}
}

func TestClient_Complete_Error(t *testing.T) {
	testErr := errors.New("boom")
	p := newMockProvider("test-provider")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			ch <- makeErrorChunk(testErr)
		}()
		return ch
	}

	c := NewClient(p, "test-model", 0)
	_, err := c.Complete(context.Background(), InferRequest{
		Messages: []Message{UserMessage("hi", nil)},
	})
	if err == nil {
		t.Fatal("expected error from Complete(), got nil")
	}
}

func TestClient_Complete_NoDone(t *testing.T) {
	p := newMockProvider("test-provider")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 2)
		go func() {
			defer close(ch)
			ch <- makeTextChunk("text")
			ch <- makeTextChunk("more")
		}()
		return ch
	}

	c := NewClient(p, "test-model", 0)
	_, err := c.Complete(context.Background(), InferRequest{
		Messages: []Message{UserMessage("hi", nil)},
	})
	if err == nil {
		t.Fatal("expected error from Complete() without done chunk")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected DeadlineExceeded, got %v", err)
	}
}

func TestClient_Name(t *testing.T) {
	t.Run("with provider", func(t *testing.T) {
		p := newMockProvider("anthropic")
		c := NewClient(p, "claude", 0)
		if c.Name() != "anthropic" {
			t.Errorf("expected 'anthropic', got %q", c.Name())
		}
	})
	t.Run("nil provider", func(t *testing.T) {
		c := &Client{}
		if c.Name() != "" {
			t.Errorf("expected empty name, got %q", c.Name())
		}
	})
}

func TestClient_ModelID(t *testing.T) {
	c := NewClient(nil, "claude-sonnet-4-6", 0)
	if c.ModelID() != "claude-sonnet-4-6" {
		t.Errorf("expected 'claude-sonnet-4-6', got %q", c.ModelID())
	}
}

func TestClient_Provider(t *testing.T) {
	p := newMockProvider("openai")
	c := NewClient(p, "gpt-5", 0)
	if c.Provider() != p {
		t.Error("Provider() returned wrong provider")
	}
}

func TestClient_InputLimit(t *testing.T) {
	t.Run("from models", func(t *testing.T) {
		p := newMockProvider("test")
		p.models = []ModelInfo{
			{ID: "test-model", InputTokenLimit: 200000, OutputTokenLimit: 8192},
		}
		c := NewClient(p, "test-model", 0)
		if c.InputLimit() != 200000 {
			t.Errorf("expected 200000, got %d", c.InputLimit())
		}
	})
	t.Run("model not found", func(t *testing.T) {
		p := newMockProvider("test")
		p.models = []ModelInfo{{ID: "other", InputTokenLimit: 100}}
		c := NewClient(p, "unknown", 0)
		if c.InputLimit() != 0 {
			t.Errorf("expected 0, got %d", c.InputLimit())
		}
	})
}

func TestClient_ThinkingEffort(t *testing.T) {
	c := NewClient(nil, "model", 0)
	if c.ThinkingEffort() != "" {
		t.Errorf("expected empty, got %q", c.ThinkingEffort())
	}
	c.SetThinkingEffort("high")
	if c.ThinkingEffort() != "high" {
		t.Errorf("expected 'high', got %q", c.ThinkingEffort())
	}
}

func TestClient_ResolveMaxTokens(t *testing.T) {
	tests := []struct {
		name       string
		maxTokens  int
		outputCap  int
		want       int
	}{
		{"explicit", 4096, 8192, 4096},
		{"from provider", 0, 8192, 8192},
		{"default fallback", 0, 0, defaultMaxTokens},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newMockProvider("test")
			if tt.outputCap > 0 {
				p.models = []ModelInfo{
					{ID: "model", OutputTokenLimit: tt.outputCap},
				}
			}
			got := resolveMaxTokens(tt.maxTokens, p, "model")
			if got != tt.want {
				t.Errorf("resolveMaxTokens() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestClient_Infer_ContextCancelled(t *testing.T) {
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			select {
			case ch <- makeTextChunk("slow response"):
			case <-ctx.Done():
				return
			}
		}()
		return ch
	}

	c := NewClient(p, "model", 0)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	ch, err := c.Infer(ctx, InferRequest{
		Messages: []Message{UserMessage("hi", nil)},
	})
	if err != nil {
		t.Fatalf("Infer() returned error: %v", err)
	}

	// Drain channel
	for range ch {
	}
}

func TestClient_Infer_ThinkingChunks(t *testing.T) {
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 3)
		go func() {
			defer close(ch)
			ch <- makeThinkingChunk("Let me think...")
			ch <- makeTextChunk("Here's the answer.")
			ch <- makeDoneChunk("Here's the answer.")
		}()
		return ch
	}

	c := NewClient(p, "model", 0)
	ch, _ := c.Infer(context.Background(), InferRequest{
		Messages: []Message{UserMessage("q", nil)},
	})

	var thinking string
	var text string
	for chunk := range ch {
		thinking += chunk.Thinking
		text += chunk.Text
	}
	if thinking != "Let me think..." {
		t.Errorf("expected thinking 'Let me think...', got %q", thinking)
	}
	if text != "Here's the answer." {
		t.Errorf("expected text 'Here's the answer.', got %q", text)
	}
}

func TestClient_Infer_MaxTokensFromProvider(t *testing.T) {
	p := newMockProvider("test")
	p.models = []ModelInfo{
		{ID: "model", OutputTokenLimit: 4096},
	}
	var gotMaxTokens int
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		gotMaxTokens = opts.MaxTokens
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			ch <- makeDoneChunk("ok")
		}()
		return ch
	}

	c := NewClient(p, "model", 0) // maxTokens=0 -> resolve from provider
	ch, _ := c.Infer(context.Background(), InferRequest{
		Messages: []Message{UserMessage("hi", nil)},
	})
	for range ch {
	}
	if gotMaxTokens != 4096 {
		t.Errorf("expected MaxTokens 4096 from provider, got %d", gotMaxTokens)
	}
}

// ── Standalone Complete Tests ──

func TestComplete(t *testing.T) {
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 2)
		go func() {
			defer close(ch)
			ch <- makeTextChunk("Hello")
			ch <- makeDoneChunk("Hello")
		}()
		return ch
	}

	resp, err := Complete(context.Background(), p, CompletionOptions{
		Model:    "model",
		Messages: []Message{UserMessage("hi", nil)},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.Content != "Hello" {
		t.Errorf("expected 'Hello', got %q", resp.Content)
	}
}

func TestComplete_WithTextAccumulation(t *testing.T) {
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			ch <- StreamChunk{
				Type: ChunkTypeDone,
				Response: &CompletionResponse{
					Content:    "",
					StopReason: "end_turn",
					Usage:      Usage{InputTokens: 0, OutputTokens: 0},
				},
			}
		}()
		return ch
	}

	resp, err := Complete(context.Background(), p, CompletionOptions{
		Model: "model",
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.Content != "" {
		t.Errorf("expected empty content, got %q", resp.Content)
	}
}

func TestComplete_Error(t *testing.T) {
	testErr := errors.New("provider failed")
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			ch <- makeErrorChunk(testErr)
		}()
		return ch
	}

	_, err := Complete(context.Background(), p, CompletionOptions{
		Model: "model",
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestComplete_NoDoneChunk(t *testing.T) {
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			ch <- makeTextChunk("text only")
		}()
		return ch
	}

	_, err := Complete(context.Background(), p, CompletionOptions{
		Model: "model",
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "stream closed without completion" {
		t.Errorf("expected 'stream closed without completion', got %q", err.Error())
	}
}

// ── Helper Tests ──

func TestUserMessage(t *testing.T) {
	msg := UserMessage("hello", nil)
	if msg.Role != RoleUser {
		t.Errorf("expected RoleUser, got %q", msg.Role)
	}
	if msg.Content != "hello" {
		t.Errorf("expected 'hello', got %q", msg.Content)
	}
}

func TestUserMessage_WithImages(t *testing.T) {
	images := []Image{{MediaType: "image/png", Data: "base64..."}}
	msg := UserMessage("describe this", images)
	if len(msg.Images) != 1 {
		t.Errorf("expected 1 image, got %d", len(msg.Images))
	}
}

func TestAssistantMessage(t *testing.T) {
	calls := []ToolCall{{ID: "1", Name: "read_file", Input: `{"path":"a.go"}`}}
	msg := AssistantMessage("result", "thinking...", calls)
	if msg.Role != RoleAssistant {
		t.Errorf("expected RoleAssistant, got %q", msg.Role)
	}
	if msg.Content != "result" {
		t.Errorf("expected 'result', got %q", msg.Content)
	}
	if msg.Thinking != "thinking..." {
		t.Errorf("expected 'thinking...', got %q", msg.Thinking)
	}
	if len(msg.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
}

func TestRoleConstants(t *testing.T) {
	if RoleUser != "user" {
		t.Errorf("RoleUser = %q, want 'user'", RoleUser)
	}
	if RoleAssistant != "assistant" {
		t.Errorf("RoleAssistant = %q, want 'assistant'", RoleAssistant)
	}
	if RoleTool != "tool_result" {
		t.Errorf("RoleTool = %q, want 'tool_result'", RoleTool)
	}
}

func TestStopReasonConstants(t *testing.T) {
	if StopEndTurn != "end_turn" {
		t.Errorf("StopEndTurn = %q", StopEndTurn)
	}
	if StopMaxTokens != "max_tokens" {
		t.Errorf("StopMaxTokens = %q", StopMaxTokens)
	}
	if StopToolUse != "tool_use" {
		t.Errorf("StopToolUse = %q", StopToolUse)
	}
	if StopMaxSteps != "max_steps" {
		t.Errorf("StopMaxSteps = %q", StopMaxSteps)
	}
	if StopCancelled != "cancelled" {
		t.Errorf("StopCancelled = %q", StopCancelled)
	}
}

func TestChunkTypeConstants(t *testing.T) {
	if ChunkTypeText != "text" {
		t.Errorf("ChunkTypeText = %q", ChunkTypeText)
	}
	if ChunkTypeThinking != "thinking" {
		t.Errorf("ChunkTypeThinking = %q", ChunkTypeThinking)
	}
	if ChunkTypeDone != "done" {
		t.Errorf("ChunkTypeDone = %q", ChunkTypeDone)
	}
	if ChunkTypeError != "error" {
		t.Errorf("ChunkTypeError = %q", ChunkTypeError)
	}
}

// ── Test helpers that verify LLM interface compliance ──

func TestClient_ImplementsLLM(t *testing.T) {
	var _ LLM = (*Client)(nil) // compile-time check
}

func TestMockProvider_ImplementsProvider(t *testing.T) {
	var _ Provider = (*mockProvider)(nil)
}

// ── Concurrent Access Tests ──

func TestClient_ConcurrentAccess(t *testing.T) {
	p := newMockProvider("test")
	p.streamFunc = func(ctx context.Context, opts CompletionOptions) <-chan StreamChunk {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			ch <- makeDoneChunk("ok")
		}()
		return ch
	}
	c := NewClient(p, "model", 0)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = c.Name()
			_ = c.ModelID()
			_ = c.ThinkingEffort()
			c.SetThinkingEffort(fmt.Sprintf("effort-%d", i))
			_ = c.InputLimit()
		}()
	}
	wg.Wait()
}
