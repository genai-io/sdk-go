package agent

import (
	"context"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Agent runs a conversation: it calls the model, executes the tools the model
// asks for, and reports everything it does as events.
//
//	for e, err := range a.Run(ctx, ai.UserMessage("hi")) { … }
type Agent struct {
	// Set at construction and read without the lock.
	maxSteps       int
	retries        int
	retryBackoff   time.Duration
	continueTries  int
	continuePrompt string
	// streamFirst and streamIdle are never zero; WithStreamTimeout maps "off"
	// to never.
	streamFirst time.Duration
	streamIdle  time.Duration
	// mintID is nil when messages are not named. Called without the lock: it
	// is the caller's code.
	mintID func() string

	// mu guards everything below it.
	mu       sync.Mutex
	client   *ai.Client
	system   string
	messages []ai.Message
	tools    []Tool
	hooks    []Hook
	// replaced and pending are applied by settle at the next step boundary.
	replaced bool
	pending  []ai.Message
	// stopTurn cancels the turn in flight; a no-op between turns.
	stopTurn context.CancelFunc

	// turnCount is how many turns this agent has run.
	turnCount atomic.Int64
	// running is held for the length of one turn: a second Run is ErrBusy.
	running atomic.Bool
}

const (
	// A stream that says nothing is the one failure that looks like work.
	defaultFirstChunk = 5 * time.Minute
	defaultIdle       = time.Minute
)

// Option configures an agent built by New.
type Option func(*Agent)

// WithSystem sets the system prompt, verbatim. Change it later with SetSystem.
func WithSystem(prompt string) Option { return func(a *Agent) { a.system = prompt } }

// WithTools sets what the model may call. Change it later with SetTools.
func WithTools(tools ...Tool) Option {
	return func(a *Agent) { a.tools = slices.Clone(tools) }
}

// WithHooks adds hooks, which run in the order added.
func WithHooks(hooks ...Hook) Option {
	return func(a *Agent) { a.hooks = append(a.hooks, hooks...) }
}

// WithMaxSteps caps model calls per turn. Zero means no cap.
func WithMaxSteps(n int) Option { return func(a *Agent) { a.maxSteps = n } }

// WithMessageIDs names every message that enters the conversation without an
// ID, the loop's own answers, tool results and continuation prompts included,
// by calling next. A message that already has an ID keeps it. The format and
// uniqueness are yours. Off by default, and then IDs are left empty. IDs are
// never sent to the model.
//
//	agent.WithMessageIDs(func() string { return uuid.NewString() })
func WithMessageIDs(next func() string) Option {
	return func(a *Agent) { a.mintID = next }
}

// WithStreamTimeout bounds how long a model stream may say nothing: first
// before the endpoint says anything, idle between events once it has. Zero
// turns that half off. The defaults are five minutes and one minute. Running
// out is an ai.KindNetwork failure.
func WithStreamTimeout(first, idle time.Duration) Option {
	return func(a *Agent) { a.streamFirst, a.streamIdle = orNever(first), orNever(idle) }
}

// orNever maps "off" to a duration no timer reaches.
func orNever(d time.Duration) time.Duration {
	if d <= 0 {
		return never
	}
	return d
}

// WithContinuation asks the model to carry on when the output cap cut its
// answer off, at most attempts times in one turn, by adding prompt to the
// conversation (reported as MessageAdded) and taking another step. Running out
// of attempts ends the turn with StopMaxTokens. Off by default.
//
//	agent.WithContinuation(2, "Your answer was cut off by the output limit. "+
//	    "Carry on from exactly where you stopped, and do not repeat anything.")
func WithContinuation(attempts int, prompt string) Option {
	return func(a *Agent) {
		if attempts > 0 && prompt != "" {
			a.continueTries, a.continuePrompt = attempts, prompt
		}
	}
}

// WithRetry replays a call that failed retryably, up to retries more times,
// waiting backoff before the first replay and doubling after each further
// failure. A stream that already yielded output is replayed as a new message,
// and a stalled one is ended and replayed. Off by default; no driver retries
// on its own.
func WithRetry(retries int, backoff time.Duration) Option {
	return func(a *Agent) {
		if retries > 0 {
			a.retries = retries
		}
		if backoff >= 0 {
			a.retryBackoff = backoff
		}
	}
}

// New builds an agent that calls client. A nil client is an error.
func New(client *ai.Client, opts ...Option) (*Agent, error) {
	if client == nil {
		return nil, errors.New("agent: a client is required")
	}
	a := &Agent{
		client:       client,
		retryBackoff: time.Second,
		streamFirst:  defaultFirstChunk,
		streamIdle:   defaultIdle,
		stopTurn:     func() {},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(a)
		}
	}
	return a, nil
}

// Messages returns a snapshot. The slice is a copy; the messages in it are
// shared and must not be mutated.
func (a *Agent) Messages() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.messages)
}

// SetMessages replaces the conversation, for compaction or restoring a saved
// one. It is announced as MessagesReplaced at the next step boundary. To add
// messages, use AddMessages: reading Messages and setting it back races with
// a running turn.
func (a *Agent) SetMessages(msgs []ai.Message) {
	msgs = a.named(msgs)

	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.messages) > 0 || len(msgs) > 0 {
		a.replaced = true
	}
	a.messages = slices.Clone(msgs)
}

// named returns msgs with missing IDs filled in, copying rather than editing
// the caller's slice. Called off the lock.
func (a *Agent) named(msgs []ai.Message) []ai.Message {
	if a.mintID == nil {
		return msgs
	}
	out := slices.Clone(msgs)
	for i := range out {
		if out[i].ID == "" {
			out[i].ID = a.mintID()
		}
	}
	return out
}

// takeReplaced reports a replacement since the last boundary, and what it left.
func (a *Agent) takeReplaced() (bool, []ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.replaced {
		return false, nil
	}
	a.replaced = false
	return true, slices.Clone(a.messages)
}

// turnNow is the turn being run. Only turn advances it, so every event
// in one agrees.
func (a *Agent) turnNow() int { return int(a.turnCount.Load()) }

// AddMessages queues messages to enter the conversation at the next step
// boundary, ahead of the next turn's own input if no turn is running. Use it
// for input that arrives while a turn runs; between turns, pass input to Run.
func (a *Agent) AddMessages(msgs ...ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pending = append(a.pending, msgs...)
}

// Pending is how many added messages have not entered the conversation yet.
// A turn does not end while any wait, so a non-zero count between turns means
// one arrived after the last ended and another is owed.
func (a *Agent) Pending() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.pending)
}

// takePending empties the queue AddMessages fills.
func (a *Agent) takePending() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()

	msgs := a.pending
	a.pending = nil
	return msgs
}

// toolset is what the model may call now.
func (a *Agent) toolset() []Tool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.tools)
}

// SetTools replaces what the model may call, from the next inference on. The
// batch of calls a model already asked for is answered by the toolset it was
// offered, not this one.
func (a *Agent) SetTools(tools ...Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools = slices.Clone(tools)
}

// definitions is what the model is told about the toolset. Caller holds a.mu.
func (a *Agent) definitions() []ai.Schema {
	defs := make([]ai.Schema, len(a.tools))
	for i, t := range a.tools {
		defs[i] = t.Schema()
	}
	return defs
}

// clientFor is where inf goes: its own client, or the agent's.
func (a *Agent) clientFor(inf *Inference) *ai.Client {
	if inf.Client != nil {
		return inf.Client
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.client
}

// SetSystem replaces the system prompt from the next inference on.
func (a *Agent) SetSystem(prompt string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.system = prompt
}

// inference is the call this agent would make. One lock, not three: a prompt
// read before SetTools and a toolset read after it would describe an agent
// that never existed.
func (a *Agent) inference() (*Inference, []Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	return &Inference{
		Client:   a.client,
		System:   a.system,
		Messages: slices.Clone(a.messages),
		Tools:    a.definitions(),
	}, slices.Clone(a.tools)
}

// toolNamed finds a tool by the name the model used, in the set the batch is
// being answered against rather than whatever SetTools has left since.
func toolNamed(tools []Tool, name string) (Tool, bool) {
	for _, t := range tools {
		if t.Schema().Name == name {
			return t, true
		}
	}
	return nil, false
}

// Interrupt ends the turn in flight: it stops with StopCanceled, Run returns,
// and the next turn starts clean. Cancelling Run's context instead ends
// everything. A running tool is asked to stop through its context. Between
// turns it does nothing.
func (a *Agent) Interrupt() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.stopTurn()
}
