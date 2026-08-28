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
//	for e, err := range a.Run(ctx, ai.UserMessage("hi")) { … }
type Agent struct {
	client *ai.Client

	system   string
	messages []ai.Message
	tools    []Tool
	hooks    []Hook

	maxSteps int
	// maxAttempts is 1 unless WithRetry asked for more: the client owns retry,
	// and stacking a second budget on it multiplies rather than adds.
	maxAttempts  int
	retryBackoff time.Duration

	// streamFirst and streamIdle bound how long a model stream may say
	// nothing. Zero disables either one.
	streamFirst time.Duration
	streamIdle  time.Duration

	// stopTurn ends the turn in flight. Never nil: between turns it is the
	// last turn's, already spent, so calling it is the no-op it should be.
	stopTurn context.CancelFunc

	// turnCount is how many exchanges this agent has held. It counts the ones
	// it actually ran, so a restored conversation starts again at zero — what
	// came back from storage was someone else's counting.
	turnCount atomic.Int64
	// running is held for the length of one exchange and released after it;
	// a second Run while one is in flight is ErrBusy, not a queue.
	running atomic.Bool

	// replaced records that SetMessages threw the conversation away, so the
	// next exchange can say so. Nothing else changes the conversation without
	// announcing it.
	replaced bool

	// pending is what AddMessages queued, taken at the next step boundary.
	pending []ai.Message

	mu sync.Mutex
}

const (
	// A stream that says nothing is the one failure that looks like work.
	// These bound it; WithStreamTimeout replaces them.
	defaultFirstChunk = 5 * time.Minute
	defaultIdle       = time.Minute
)

// Option sets one thing an agent does not need in order to exist. New's
// parameter is what it cannot be built without; everything else is named at
// the site that wanted it, so a plain agent is agent.New(client).
type Option func(*Agent)

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
//
// Seeded messages are announced as MessagesReplaced at the start of the first
// exchange, for the same reason SetMessages is: they entered the conversation
// without ever being appended to it, so nothing folding what was appended
// would otherwise have them.
func WithMessages(msgs []ai.Message) Option {
	return func(a *Agent) {
		a.messages = slices.Clone(msgs)
		a.replaced = len(msgs) > 0
	}
}

// WithMaxSteps caps model calls per exchange. Zero means no cap.
func WithMaxSteps(n int) Option { return func(a *Agent) { a.maxSteps = n } }

// WithStreamTimeout bounds how long a model stream may say nothing: first is
// how long the endpoint has to say anything at all, idle how long it may pause
// once it has started. Either at zero is that half turned off, independently.
//
// A stalled stream is the one failure that looks like work, so this is on by
// default — five minutes and one minute. A model that reasons silently for
// longer than idle needs a longer one, or none.
//
// Running out is reported as a network failure, because it is one, and is
// retried like any other.
func WithStreamTimeout(first, idle time.Duration) Option {
	return func(a *Agent) { a.streamFirst, a.streamIdle = orNever(first), orNever(idle) }
}

// orNever turns "off" into a duration no timer reaches, so that a disabled
// timeout is a timer that does not fire rather than a branch every reader of
// these fields has to remember.
func orNever(d time.Duration) time.Duration {
	if d <= 0 {
		return never
	}
	return d
}

// WithRetry has the agent replay a failed model call, at most attempts times
// in total, waiting backoff before the second and doubling it after each
// further failure — the same rule, and the same arguments, as ai.Retry.
//
// It is off by default, and that is the point: retry belongs on the client,
// where ai.Retry already implements it, and two budgets do not compose. An
// agent set to three attempts on a client wrapped in ai.Retry(3, …) is nine
// model calls for one step, and neither loop can see the other's count.
//
//	client := ai.NewClientWithDriver(ai.Wrap(driver, ai.Retry(3, time.Second)), model)
//
// Turn this on for what the client cannot retry: ai.Retry gives up once a
// stream has yielded output, because its caller has already seen it, where
// this loop discards the attempt and opens a new message. A stalled stream is
// the same case — ending one cancels the context ai.Retry would wait on.
func WithRetry(attempts int, backoff time.Duration) Option {
	return func(a *Agent) {
		if attempts > 0 {
			a.maxAttempts = attempts
		}
		if backoff >= 0 {
			a.retryBackoff = backoff
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
		client:       client,
		maxAttempts:  1,
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

// Messages returns a snapshot of the conversation. The slice is a copy; the
// messages in it are shared and must not be mutated.
func (a *Agent) Messages() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.messages)
}

// SetMessages replaces the conversation. This is how compaction and session
// restore work: both hand over a history built somewhere else. To add to it
// rather than replace it, use AddMessages — building on Messages and setting
// the result back races with the exchange that may be running.
//
// Nothing on the event stream says this happened, because nothing here knows
// it did until it is done — and whoever calls it does. A consumer folding
// MessageAdded would go on holding what the agent threw away, so tell it:
// session.Recorder.Snapshot is that for a session, and a caller keeping its
// own view of the conversation resets it the same way.
func (a *Agent) SetMessages(msgs []ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = slices.Clone(msgs)
	// Replacing a conversation is not appending to it, and anything watching
	// has to be told which happened. There is nowhere to say so from here —
	// no exchange is running — so it is said at the start of the next one.
	a.replaced = true
}

// takeReplaced reports whether the conversation was replaced since the last
// exchange, and hands back what it was replaced with.
func (a *Agent) takeReplaced() (bool, []ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.replaced {
		return false, nil
	}
	a.replaced = false
	return true, slices.Clone(a.messages)
}

// turnNow is the exchange being held, for events that say which one they
// belong to. Only turn advances it, so every event in one agrees.
func (a *Agent) turnNow() int { return int(a.turnCount.Load()) }

// AddMessages puts messages into the conversation from outside an exchange —
// something that arrived while one is already running, routed in from
// elsewhere or typed while the model worked.
//
// They enter at the next step boundary: the model sees them at its next call,
// and each is reported as MessageAdded there. Changing what the model is about
// to see is safe exactly once per call, and that is where — which is also the
// first moment there is a goroutine allowed to report it.
//
// Between exchanges, pass them to Run instead. This is for the ones that could
// not wait.
func (a *Agent) AddMessages(msgs ...ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pending = append(a.pending, msgs...)
}

// taken empties the queue AddMessages fills.
func (a *Agent) taken() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()

	msgs := a.pending
	a.pending = nil
	return msgs
}

func (a *Agent) Tools() []Tool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.tools)
}

// SetTools replaces what the model may call. It takes effect on the next
// inference, not mid-stream.
func (a *Agent) SetTools(tools ...Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools = slices.Clone(tools)
}

// definitions is what the model is told about the toolset. Caller holds a.mu.
func (a *Agent) definitions() []ai.Tool {
	defs := make([]ai.Tool, len(a.tools))
	for i, t := range a.tools {
		defs[i] = ai.Tool{Schema: t.Schema()}
	}
	return defs
}

func (a *Agent) System() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.system
}

// AddHooks registers more hooks. They take effect on the next inference, and
// hooks run in the order they were added.
func (a *Agent) AddHooks(hooks ...Hook) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.hooks = append(a.hooks, hooks...)
}

// SetSystem replaces the system prompt. It takes effect on the next inference,
// so an application that rebuilds its prompt when something changes — the
// working directory, the active skills — calls this and is done.
func (a *Agent) SetSystem(prompt string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.system = prompt
}

// String names the agent by the model it is bound to, which is the one thing
// about it that is both stable and worth reading in a log.
func (a *Agent) String() string { return fmt.Sprintf("agent(%s)", a.client.Model().ID) }

// inference is the call this agent would make. One lock, not three: a prompt
// read before SetTools and a toolset read after it would describe an agent
// that never existed.
func (a *Agent) inference() *Inference {
	a.mu.Lock()
	defer a.mu.Unlock()

	return &Inference{
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
		if t.Schema().Name == name {
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
	defer a.mu.Unlock()
	a.stopTurn()
}
