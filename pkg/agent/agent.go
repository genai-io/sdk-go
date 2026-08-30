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

	// Settled at construction and never written again, so they are read
	// without the lock. Anything given a setter has to move to the group
	// below and take the lock with it.
	maxSteps      int
	retryAttempts int
	retryBackoff  time.Duration
	// resumeTries and resumePrompt carry WithContinuation: how many times a
	// cut-off answer may be asked to carry on, and what to ask with.
	resumeTries  int
	resumePrompt string
	// streamFirst bounds how long the endpoint may say nothing at all,
	// streamIdle how long it may pause once it has started. WithStreamTimeout
	// normalises "off" to never, so neither is ever zero here and nothing
	// reading them needs a case for it.
	streamFirst time.Duration
	streamIdle  time.Duration

	// Everything a caller can change while the agent runs, and the lock that
	// makes that safe. Read or write one of these and you hold mu.
	mu       sync.Mutex
	system   string
	messages []ai.Message
	tools    []Tool
	hooks    []Hook
	// replaced and pending are the two ways the conversation changes from
	// outside an exchange, waiting for one to announce them. They are
	// separate because they are announced at different moments: a
	// replacement at the start of the next exchange, since everything before
	// it is gone, and queued messages at the next step boundary, which is the
	// one place it is safe to change what the model is about to see.
	replaced bool
	pending  []ai.Message
	// stopTurn ends the turn in flight. Never nil: between turns it is a
	// no-op, so Interrupt needs no case for having nothing to interrupt.
	stopTurn context.CancelFunc
	// stopped closes when the exchange in flight has finished — after the
	// last event, and after the agent is free to run another. Between
	// exchanges it is an already-closed channel, for the same reason stopTurn
	// is a no-op there.
	stopped chan struct{}

	// Their own synchronisation, because they are read outside mu.
	//
	// turnCount is how many exchanges this agent has held. It counts the ones
	// it actually ran, so a restored conversation starts again at zero — what
	// came back from storage was someone else's counting.
	turnCount atomic.Int64
	// running is held for the length of one exchange and released after it: a
	// second Run while one is in flight is ErrBusy, not a queue.
	running atomic.Bool
}

const (
	// A stream that says nothing is the one failure that looks like work.
	defaultFirstChunk = 5 * time.Minute
	defaultIdle       = time.Minute
)

// Option sets one thing an agent does not need in order to exist — New's
// parameter is what it cannot be built without, so a plain agent is
// agent.New(client).
type Option func(*Agent)

// WithSystem sets the system prompt, verbatim. Assembling one is the
// application's business. Change it later with SetSystem.
func WithSystem(prompt string) Option { return func(a *Agent) { a.system = prompt } }

// WithTools sets what the model may call. Change it later with SetTools.
func WithTools(tools ...Tool) Option {
	return func(a *Agent) { a.tools = slices.Clone(tools) }
}

// WithHooks adds hooks. Several may be registered: a permission gate and an
// audit log should not have to be one function.
func WithHooks(hooks ...Hook) Option {
	return func(a *Agent) { a.hooks = append(a.hooks, hooks...) }
}

// WithMessages seeds the conversation, e.g. from a restored session. Change it
// later with SetMessages.
//
// The first exchange announces these as MessagesReplaced: they entered without
// ever being appended, so a fold over what was appended would not have them.
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
// once it has started. Either at zero turns that half off.
//
// A stalled stream is the one failure that looks like work, so this is on by
// default at five minutes and one minute; a model that reasons silently for
// longer needs a longer idle, or none. Running out is a network failure,
// because it is one.
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

// WithContinuation asks the model to carry on when the output cap cut its
// answer off, at most attempts times in one exchange, by putting prompt into
// the conversation and taking another step.
//
// A model stopped by max_tokens did not finish: it was interrupted. The loop
// knows when that happened and has budget left, but what to say about it — and
// whether to pay for more tokens at all — is the application's, so this is off
// by default and the words are yours.
//
//	agent.WithContinuation(2, "Your answer was cut off by the output limit. "+
//	    "Carry on from exactly where you stopped, and do not repeat anything.")
//
// The prompt enters the conversation as an ordinary message and is reported as
// MessageAdded, so a session records what was asked and a restore replays it.
// Running out of attempts ends the exchange with StopMaxTokens, the same as
// never asking: the answer is still cut off.
func WithContinuation(attempts int, prompt string) Option {
	return func(a *Agent) {
		if attempts > 0 && prompt != "" {
			a.resumeTries, a.resumePrompt = attempts, prompt
		}
	}
}

// WithRetry replays a failed model call, at most attempts times, waiting
// backoff before the second and doubling after each further failure — the same
// rule and arguments as ai.Retry.
//
// Off by default, and that is the point: retry belongs on the client, and two
// budgets multiply rather than add. Three attempts here on a client wrapped in
// ai.Retry(3, …) is nine model calls for one step, with neither loop able to
// see the other's count.
//
// Turn it on for what the client cannot replay: a stream that already yielded
// output, which ai.Retry gives up on because its caller has seen it, and a
// stalled one, since ending a stall cancels the context ai.Retry would wait on.
func WithRetry(attempts int, backoff time.Duration) Option {
	return func(a *Agent) {
		if attempts > 0 {
			a.retryAttempts = attempts
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
		client:        client,
		retryAttempts: 1,
		retryBackoff:  time.Second,
		streamFirst:   defaultFirstChunk,
		streamIdle:    defaultIdle,
		stopTurn:      func() {},
		stopped:       closed,
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

// SetMessages replaces the conversation — how compaction and session restore
// both work. To add to it instead use AddMessages: reading Messages and
// setting the result back races with the exchange that may be running.
//
// The next exchange announces this as MessagesReplaced.
func (a *Agent) SetMessages(msgs []ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.messages = slices.Clone(msgs)
	// Replacing a conversation is not appending to it, and anything watching
	// has to be told which happened. There is nowhere to say so from here —
	// no exchange is running — so it is said at the start of the next one.
	a.replaced = true
}

// takeReplaced reports a replacement since the last exchange, and what it left.
func (a *Agent) takeReplaced() (bool, []ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.replaced {
		return false, nil
	}
	a.replaced = false
	return true, slices.Clone(a.messages)
}

// turnNow is the exchange being held. Only turn advances it, so every event
// in one agrees.
func (a *Agent) turnNow() int { return int(a.turnCount.Load()) }

// AddMessages puts messages into the conversation from outside an exchange —
// typed while the model worked, or routed in from elsewhere. They enter at the
// next step boundary, which is the one place it is safe to change what the
// model is about to see. Between exchanges, pass them to Run instead.
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

// Interrupt ends the exchange in flight — it stops with StopCanceled, Run
// returns, and the next one starts clean. This is what a user pressing escape
// asks for; cancelling Run's own context is the other thing, and ends
// everything.
//
// The returned channel closes when that exchange has actually finished: after
// its last event, and after the agent is free to run another. Wait on it when
// something has to happen once the agent has stopped touching the
// conversation, from a goroutine that is not the one ranging over Run — the
// keystroke handler that asked for the interrupt, typically, which cannot see
// the range end.
//
//	<-a.Interrupt()      // the turn is over; the agent is idle
//	a.SetMessages(fresh)
//
// Between exchanges there is nothing to interrupt: the channel is already
// closed and this does nothing.
func (a *Agent) Interrupt() <-chan struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.stopTurn()
	return a.stopped
}

// closed stands in for the exchange that is not running, so that Interrupt
// between two of them returns something a caller can wait on without a case
// for there being nothing to wait for.
var closed = func() chan struct{} {
	c := make(chan struct{})
	close(c)
	return c
}()
