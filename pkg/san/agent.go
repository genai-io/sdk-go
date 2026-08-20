package san

import (
	"context"
	"fmt"
	"sync"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Agent is a San agent instance. It holds an LLM, system prompt, tools,
// and manages the conversation lifecycle.
type Agent struct {
	id     string
	llm    llm.LLM
	system string
	tools  *ToolSet

	mu       sync.RWMutex
	messages []llm.Message
	inbox    chan llm.Message
	outbox   chan Event

	maxSteps int

	closed  bool
	closeMu sync.Mutex
}

// New creates an agent with the given options.
func New(opts ...AgentOption) (*Agent, error) {
	cfg := defaultConfig()
	for _, o := range opts {
		o(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &Agent{
		id:       cfg.id,
		llm:      cfg.llm,
		system:   cfg.system,
		tools:    cfg.tools,
		inbox:    make(chan llm.Message, cfg.inboxBuf),
		outbox:   make(chan Event, cfg.outboxBuf),
		maxSteps: cfg.maxSteps,
	}, nil
}

// ID returns the agent identifier.
func (a *Agent) ID() string { return a.id }

// Inbox returns the send channel for user messages and signals.
func (a *Agent) Inbox() chan<- llm.Message { return a.inbox }

// Outbox returns the receive channel for agent lifecycle events.
func (a *Agent) Outbox() <-chan Event { return a.outbox }

// Messages returns a snapshot of the conversation history.
func (a *Agent) Messages() []llm.Message {
	a.mu.RLock()
	defer a.mu.RUnlock()
	out := make([]llm.Message, len(a.messages))
	copy(out, a.messages)
	return out
}

// SetMessages replaces the conversation history.
func (a *Agent) SetMessages(msgs []llm.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = make([]llm.Message, len(msgs))
	copy(a.messages, msgs)
}

// Run starts the agent's main loop. Blocks until ctx is cancelled or the
// inbox is closed. Emits events to Outbox.
func (a *Agent) Run(ctx context.Context) error {
	defer close(a.outbox)

	a.emit(Event{Type: OnStart, Source: a.id})

	for {
		// Phase 1: Wait for input
		msg, ok := a.waitForInput(ctx)
		if !ok {
			a.emit(Event{Type: OnStop, Source: a.id})
			return ctx.Err()
		}

		// Phase 2: Drain remaining inbox messages (non-blocking)
		a.appendMessage(msg)
		a.drainInbox()

		// Phase 3: Think + Act loop
		result, err := a.ThinkAct(ctx)
		if err != nil {
			a.emit(Event{Type: OnStop, Source: a.id, Data: err})
			return err
		}
		a.emit(Event{Type: OnTurn, Source: a.id, Data: *result})
	}
}

// ThinkAct runs one inference-action cycle synchronously.
func (a *Agent) ThinkAct(ctx context.Context) (*Result, error) {
	result := &Result{}
	for step := 0; a.maxSteps == 0 || step < a.maxSteps; step++ {
		select {
		case <-ctx.Done():
			result.StopReason = llm.StopCancelled
			return result, ctx.Err()
		default:
		}

		a.drainInbox()

		// Build request
		schemas := a.tools.Schemas()
		req := llm.InferRequest{
			System:   a.system,
			Messages: a.Messages(),
			Tools:    schemas,
		}

		a.emit(Event{Type: PreInfer, Source: a.id})

		ch, err := a.llm.Infer(ctx, req)
		if err != nil {
			return result, fmt.Errorf("infer: %w", err)
		}

		// Collect response
		resp, toolCalls := collectChunks(ch, a.outbox, a.id)

		a.emit(Event{Type: PostInfer, Source: a.id})

		result.Steps = step + 1
		result.TokensIn += resp.TokensIn
		result.TokensOut += resp.TokensOut

		if resp.StopReason == llm.StopEndTurn {
			result.Content = resp.Content
			result.StopReason = llm.StopEndTurn
			result.Messages = a.Messages()
			return result, nil
		}

		if resp.StopReason == llm.StopCancelled {
			result.StopReason = llm.StopCancelled
			result.Messages = a.Messages()
			return result, ctx.Err()
		}

		// Execute tool calls
		if len(toolCalls) > 0 {
			// Append assistant message with tool calls
			a.appendMessage(llm.AssistantMessage(resp.Content, resp.Thinking, toolCalls))

			for _, tc := range toolCalls {
				a.emit(Event{Type: PreTool, Source: tc.Name, Data: tc})

				result.ToolUses++
				toolResult := a.executeTool(ctx, tc)

				a.emit(Event{Type: PostTool, Source: tc.Name, Data: *toolResult})

				a.appendMessage(llm.Message{
					Role:       llm.RoleTool,
					ToolResult: toolResult,
				})
			}
			continue
		}

		// No tool calls, stop
		result.Content = resp.Content
		result.StopReason = resp.StopReason
		result.Messages = a.Messages()
		return result, nil
	}

	result.StopReason = llm.StopMaxSteps
	result.Messages = a.Messages()
	return result, nil
}

// Close shuts down the agent gracefully.
func (a *Agent) Close() error {
	a.closeMu.Lock()
	defer a.closeMu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
	close(a.inbox)
	return nil
}

func (a *Agent) waitForInput(ctx context.Context) (llm.Message, bool) {
	select {
	case <-ctx.Done():
		return llm.Message{}, false
	case msg, ok := <-a.inbox:
		return msg, ok
	}
}

func (a *Agent) drainInbox() {
	for {
		select {
		case msg, ok := <-a.inbox:
			if !ok {
				return
			}
			a.appendMessage(msg)
		default:
			return
		}
	}
}

func (a *Agent) appendMessage(msg llm.Message) {
	a.mu.Lock()
	a.messages = append(a.messages, msg)
	a.mu.Unlock()
}

func (a *Agent) emit(evt Event) {
	select {
	case a.outbox <- evt:
	default:
	}
}

func (a *Agent) executeTool(ctx context.Context, tc llm.ToolCall) *llm.ToolResult {
	t := a.tools.Get(tc.Name)
	if t == nil {
		return &llm.ToolResult{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Content:    fmt.Sprintf("unknown tool: %s", tc.Name),
			IsError:    true,
		}
	}
	// Try to parse input as JSON
	var input map[string]any
	if tc.Input != "" {
		_ = jsonUnmarshal(tc.Input, &input)
	}
	content, err := t.Execute(ctx, input)
	if err != nil {
		return &llm.ToolResult{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Content:    err.Error(),
			IsError:    true,
		}
	}
	return &llm.ToolResult{
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
		Content:    content,
	}
}

func collectChunks(ch <-chan llm.Chunk, outbox chan<- Event, source string) (llm.InferResponse, []llm.ToolCall) {
	var resp llm.InferResponse
	var toolCalls []llm.ToolCall
	for chunk := range ch {
		if chunk.Text != "" {
			outbox <- Event{Type: OnChunk, Source: source, Data: chunk}
		}
		if chunk.Err != nil {
			resp.StopReason = llm.StopCancelled
			return resp, toolCalls
		}
		if chunk.Done && chunk.Response != nil {
			resp = *chunk.Response
			toolCalls = chunk.Response.ToolCalls
		}
	}
	return resp, toolCalls
}

func jsonUnmarshal(s string, v *map[string]any) error {
	// Simple JSON-like parsing for tool inputs
	if s == "" {
		*v = map[string]any{}
		return nil
	}
	// Just set to empty map for now — real implementations use encoding/json
	*v = map[string]any{"_raw": s}
	return nil
}
