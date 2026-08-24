package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Agent runs a conversation: it calls the model, executes the tools the model
// asks for, and reports everything it does as events.
//
//	go a.Run(ctx)
//
//	a.In() <- ai.UserMessage("hi")
//	for e := range a.Out() { … }
type Agent struct {
	client *ai.Client

	id       string
	name     string
	system   string
	messages []ai.Message
	tools    []Tool
	hooks    []Hook

	maxSteps    int
	maxAttempts int

	// firstChunk and idle bound how long a stream may say nothing. Zero
	// disables either one.
	firstChunk time.Duration
	idle       time.Duration

	inBuf  int
	outBuf int

	// interrupt cancels the turn in flight. Nil between turns.
	interrupt context.CancelFunc

	// turnCount is how many exchanges this agent has held. It counts the ones
	// it actually ran, so a restored conversation starts again at zero — what
	// came back from storage was someone else's counting.
	turnCount atomic.Int64
	running   bool

	// The two channels an agent is: what comes in, what goes out.
	in  chan ai.Message
	out chan Event

	mu sync.Mutex
}

const (
	// defaultInputBuffer is how many messages may wait before a sender blocks.
	defaultInputBuffer = 64
	// defaultEventBuffer is how far ahead of a reader the agent may get. Past
	// it the loop waits, which is the right thing to do: an event a session
	// needed is worse to lose than a paint is to delay.
	defaultEventBuffer = 256

	// A stream that says nothing is the one failure that looks like work.
	// These bound it; WithStreamTimeout replaces them.
	defaultFirstChunk = 5 * time.Minute
	defaultIdle       = time.Minute
)

// Option sets one thing an agent does not need in order to exist. New's
// parameter is what it cannot be built without; everything else is named at
// the site that wanted it, so a plain agent is agent.New(client).
type Option func(*Agent)

// WithID identifies the agent in the events it emits, for when more than one
// runs. Machines read this.
func WithID(id string) Option { return func(a *Agent) { a.id = id } }

// WithName gives the agent a name for a person to read. Separate from the id
// because a name can be renamed and an id cannot.
func WithName(name string) Option { return func(a *Agent) { a.name = name } }

// Name returns the agent's human-readable name, or its id if it has none.
func (a *Agent) Name() string {
	if a.name != "" {
		return a.name
	}
	return a.id
}

// WithSystem sets the system prompt, verbatim. A string, because assembling
// one is the application's business and this package needs no opinion on it.
// Change it later with SetSystem.
func WithSystem(prompt string) Option { return func(a *Agent) { a.system = prompt } }

// WithTools sets what the model may call. Change it later with SetTools.
func WithTools(tools ...Tool) Option {
	return func(a *Agent) { a.tools = slices.Clone(tools) }
}

// WithHooks adds hooks. Several may be registered — a permission gate and an
// audit log are two different concerns and should not have to be one function.
func WithHooks(hooks ...Hook) Option {
	return func(a *Agent) { a.hooks = append(a.hooks, hooks...) }
}

// WithMessages seeds the conversation, e.g. from a restored session. Change it
// later with SetMessages.
func WithMessages(msgs []ai.Message) Option {
	return func(a *Agent) { a.messages = slices.Clone(msgs) }
}

// WithMaxSteps caps model calls per exchange. Zero means no cap.
func WithMaxSteps(n int) Option { return func(a *Agent) { a.maxSteps = n } }

// WithStreamTimeout bounds how long a model stream may say nothing: first is
// how long the endpoint has to say anything at all, idle how long it may pause
// once it has started. Either at zero turns that half off.
//
// A stalled stream is the one failure that looks like work, so this is on by
// default — five minutes and one minute. A model that reasons silently for
// longer than idle needs a longer one, or none.
//
// Running out is reported as a network failure, because it is one, and is
// retried like any other.
func WithStreamTimeout(first, idle time.Duration) Option {
	return func(a *Agent) { a.firstChunk, a.idle = first, idle }
}

// WithBuffers sizes the two channels: how many messages may wait on In before
// a sender blocks, and how far ahead of a reader Out may get.
func WithBuffers(in, out int) Option {
	return func(a *Agent) {
		if in > 0 {
			a.inBuf = in
		}
		if out > 0 {
			a.outBuf = out
		}
	}
}

// WithMaxAttempts caps attempts per model call when a stream fails retryably.
func WithMaxAttempts(n int) Option {
	return func(a *Agent) {
		if n > 0 {
			a.maxAttempts = n
		}
	}
}

// New builds an agent on a model handle. The client is the parameter because
// an agent without one can do nothing, and it is bound to that model for its
// lifetime. A missing one is an error rather than a panic: a library that
// panics on bad configuration takes the caller's process down with it.
func New(client *ai.Client, opts ...Option) (*Agent, error) {
	if client == nil {
		return nil, errors.New("agent: a client is required")
	}
	a := &Agent{
		client:      client,
		maxAttempts: 3,
		firstChunk:  defaultFirstChunk,
		idle:        defaultIdle,
		inBuf:       defaultInputBuffer,
		outBuf:      defaultEventBuffer,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(a)
		}
	}
	a.in = make(chan ai.Message, a.inBuf)
	a.out = make(chan Event, a.outBuf)
	return a, nil
}

// Messages returns a snapshot of the conversation. The slice is a copy; the
// messages in it are shared and must not be mutated.
func (a *Agent) Messages() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.messages)
}

// SetMessages replaces the conversation. This is how compaction and session
// restore work: both hand over a history built somewhere else.
func (a *Agent) SetMessages(msgs []ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = slices.Clone(msgs)
}

func (a *Agent) Tools() []Tool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.tools)
}

// SetTools replaces what the model may call. It takes effect on the next
// inference, not mid-stream.
func (a *Agent) SetTools(tools []Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools = slices.Clone(tools)
}

// definitions is what the model is told about the toolset. Caller holds a.mu.
func (a *Agent) definitions() []ai.Tool {
	defs := make([]ai.Tool, len(a.tools))
	for i, t := range a.tools {
		defs[i] = t.Definition()
	}
	return defs
}

func (a *Agent) System() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.system
}

// AddHook registers another hook. It takes effect on the next inference, and
// hooks run in the order they were added.
func (a *Agent) AddHook(h Hook) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.hooks = append(a.hooks, h)
}

// SetSystem replaces the system prompt. It takes effect on the next inference,
// so an application that rebuilds its prompt when something changes — the
// working directory, the active skills — calls this and is done.
func (a *Agent) SetSystem(prompt string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.system = prompt
}

func (a *Agent) String() string {
	if n := a.Name(); n != "" {
		return fmt.Sprintf("agent(%s)", n)
	}
	return fmt.Sprintf("agent(%s)", a.client.Model().ID)
}

// In is where you put messages. Close it to say there is no more work; the
// agent finishes the exchange it is in and Run returns.
//
//	a.In() <- ai.UserMessage("hi")
func (a *Agent) In() chan<- ai.Message { return a.in }

// Out is where the agent reports what it does. Range over it: Run closes it
// when it stops, so the range ends on its own.
//
//	go a.Run(ctx)
//	for e := range a.Out() { … }
func (a *Agent) Out() <-chan Event { return a.out }

// request is what this agent would send. One lock, not three: a prompt read
// before SetTools and a toolset read after it would describe an agent that
// never existed.
func (a *Agent) request() *ai.Request {
	a.mu.Lock()
	defer a.mu.Unlock()

	return &ai.Request{
		System:   a.system,
		Messages: slices.Clone(a.messages),
		Tools:    a.definitions(),
	}
}

// toolNamed finds a tool by the name the model used.
func (a *Agent) toolNamed(name string) (Tool, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, t := range a.tools {
		if t.Definition().Name == name {
			return t, true
		}
	}
	return nil, false
}

// Interrupt ends the turn in flight without ending the run: the exchange stops
// with StopCanceled, and the agent goes back to waiting on In.
//
// This is what a user pressing escape asks for. Cancelling Run's own context
// is the other thing — it ends everything. Between turns there is nothing to
// interrupt and this does nothing; watch TurnEnd to know it landed.
func (a *Agent) Interrupt() {
	a.mu.Lock()
	cancel := a.interrupt
	a.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}
