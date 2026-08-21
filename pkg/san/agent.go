package san

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"sync"

	"github.com/genai-io/sdk-go/pkg/llm"
)

// Model is the inference dependency an Agent needs.
//
// It is declared here, at the consumer, rather than taken from package llm, so
// an agent can be driven by anything that streams events — a real client, a
// recorded transcript, a router that picks a model per turn. *llm.Client
// satisfies it.
type Model interface {
	Stream(ctx context.Context, p *llm.Prompt, opts *llm.Options) iter.Seq2[llm.Event, error]
}

// Agent-loop stop reasons. They sit alongside the protocol's own reasons in
// llm because they describe how the loop ended, which no provider knows about.
const (
	// StopCancelled means the caller's context ended the turn.
	StopCancelled llm.StopReason = "cancelled"
	// StopMaxSteps means the turn hit its inference-step budget with the model
	// still asking for tools.
	StopMaxSteps llm.StopReason = "max_steps"
)

// Agent is a San agent instance. It holds a model, a system prompt and a tool
// set, and runs the think-act loop over them.
type Agent struct {
	id     string
	model  Model
	system string
	tools  *ToolSet

	mu       sync.RWMutex
	messages []llm.Message
	inbox    chan llm.Message
	outbox   chan Event

	maxSteps int
	options  *llm.Options

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
		model:    cfg.model,
		system:   cfg.system,
		tools:    cfg.tools,
		inbox:    make(chan llm.Message, cfg.inboxBuf),
		outbox:   make(chan Event, cfg.outboxBuf),
		maxSteps: cfg.maxSteps,
		options:  cfg.options,
	}, nil
}

// ID returns the agent identifier.
func (a *Agent) ID() string { return a.id }

// Inbox returns the send channel for user messages.
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

// Run starts the agent's main loop. It blocks until ctx is cancelled or the
// inbox is closed, emitting events to Outbox.
func (a *Agent) Run(ctx context.Context) error {
	defer close(a.outbox)

	a.emit(Event{Type: OnStart, Source: a.id})

	for {
		msg, ok := a.waitForInput(ctx)
		if !ok {
			a.emit(Event{Type: OnStop, Source: a.id})
			return ctx.Err()
		}

		a.appendMessage(msg)
		a.drainInbox()

		result, err := a.ThinkAct(ctx)
		if err != nil {
			a.emit(Event{Type: OnStop, Source: a.id, Data: err})
			return err
		}
		a.emit(Event{Type: OnTurn, Source: a.id, Data: *result})
	}
}

// ThinkAct runs one turn synchronously: infer, run any tools the model asked
// for, and repeat until it answers or the step budget runs out.
func (a *Agent) ThinkAct(ctx context.Context) (*Result, error) {
	result := &Result{}

	for step := 0; a.maxSteps == 0 || step < a.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			result.StopReason = StopCancelled
			result.Messages = a.Messages()
			return result, err
		}

		a.drainInbox()

		prompt := &llm.Prompt{
			System:   a.system,
			Messages: a.Messages(),
			Tools:    a.tools.Schemas(),
		}

		a.emit(Event{Type: PreInfer, Source: a.id})
		resp, err := a.infer(ctx, prompt)
		a.emit(Event{Type: PostInfer, Source: a.id})
		if resp != nil {
			// Count the step and its tokens whether or not it succeeded: a
			// turn that failed after streaming still cost what it cost.
			result.Steps = step + 1
			result.Usage.Add(resp.Usage)
		}
		if err != nil {
			result.Messages = a.Messages()
			if resp != nil {
				result.Content = resp.Content
			}
			if ctx.Err() != nil {
				result.StopReason = StopCancelled
				return result, ctx.Err()
			}
			result.StopReason = llm.StopError
			return result, fmt.Errorf("infer: %w", err)
		}

		result.Steps = step + 1

		if len(resp.ToolCalls) == 0 {
			result.Content = resp.Content
			result.StopReason = resp.StopReason
			result.Messages = a.Messages()
			return result, nil
		}

		// The assistant turn has to go into history before its results do, and
		// it has to be the response's own message so the thinking signature
		// and reasoning items ride along — reasoning models reject the next
		// request without them.
		a.appendMessage(resp.Message())

		results := make([]llm.ToolResult, 0, len(resp.ToolCalls))
		for _, tc := range resp.ToolCalls {
			a.emit(Event{Type: PreTool, Source: tc.Name, Data: tc})
			result.ToolUses++
			toolResult := a.executeTool(ctx, tc)
			a.emit(Event{Type: PostTool, Source: tc.Name, Data: toolResult})
			results = append(results, toolResult)
		}
		// One message carrying every result: that is the shape the protocols
		// expect, and splitting it would break tool-call pairing.
		a.appendMessage(llm.ToolResultsMessage(results...))
	}

	result.StopReason = StopMaxSteps
	result.Messages = a.Messages()
	return result, nil
}

// infer streams one inference call, forwarding deltas to the outbox.
//
// A failure returns the response as well as the error: the turn still streamed
// text and still burned tokens, and the caller needs both to account for the
// spend and to show what arrived.
func (a *Agent) infer(ctx context.Context, p *llm.Prompt) (*llm.Response, error) {
	var resp *llm.Response
	for event, err := range a.model.Stream(ctx, p, a.options) {
		if event.Type == llm.EventDone && event.Response != nil {
			resp = event.Response
		}
		if err != nil {
			return resp, err
		}
		switch event.Type {
		case llm.EventTextDelta, llm.EventThinkingDelta:
			a.emit(Event{Type: OnChunk, Source: a.id, Data: event})
		}
	}
	if resp == nil {
		return nil, fmt.Errorf("stream ended without a response")
	}
	return resp, nil
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

// emit drops the event when no one is reading. A blocked observer must not
// stall the agent's own loop.
func (a *Agent) emit(evt Event) {
	select {
	case a.outbox <- evt:
	default:
	}
}

func (a *Agent) executeTool(ctx context.Context, tc llm.ToolCall) llm.ToolResult {
	fail := func(format string, args ...any) llm.ToolResult {
		return llm.ToolResult{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Content:    fmt.Sprintf(format, args...),
			IsError:    true,
		}
	}

	t := a.tools.Get(tc.Name)
	if t == nil {
		return fail("unknown tool: %s", tc.Name)
	}

	// Arguments are model output, and model output is wrong sometimes.
	// Running the tool anyway turns a mistake the model could have corrected
	// into whatever the tool does with nonsense — a deletion with an empty
	// path, a query with a null filter. Handing it back as a tool error is
	// what the model recovers from; failing the turn is not.
	if err := t.Schema().ValidateArgs(tc.Input); err != nil {
		return fail("%v", err)
	}

	input := map[string]any{}
	if tc.Input != "" {
		if err := json.Unmarshal([]byte(tc.Input), &input); err != nil {
			return fail("invalid tool input: %v", err)
		}
	}

	content, err := t.Execute(ctx, input)
	if err != nil {
		return fail("%v", err)
	}
	return llm.ToolResult{ToolCallID: tc.ID, ToolName: tc.Name, Content: content}
}
