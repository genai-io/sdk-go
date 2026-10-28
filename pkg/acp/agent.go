package acp

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"

	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

// AgentHandlers implement the stable receiving-side ACP core. Callbacks run
// concurrently across sessions and must cooperate with cancellation. Returned
// values transfer ownership to Agent; do not mutate them after returning.
// Storage, runtime cleanup, authorization, capability-dependent content and
// MCP server handling belong to the application. Only advertise capabilities
// implemented by these callbacks; optional protocol methods are not dispatched.
type AgentHandlers struct {
	Initialize    func(context.Context, protocol.InitializeRequest) (protocol.InitializeResponse, error)
	CreateSession func(context.Context, protocol.NewSessionRequest) (protocol.NewSessionResponse, error)
	Prompt        func(context.Context, *Turn, protocol.PromptRequest) (protocol.PromptResponse, error)
}

// AgentOptions bounds transport and connection-local session bookkeeping.
// MaxSessions defaults to 64. There is no eviction or session-close protocol:
// successful sessions occupy a slot until the connection ends. Concurrent
// creations reserve slots before invoking application code.
type AgentOptions struct {
	Limits      Limits
	MaxSessions int
}

// Agent is a typed ACP receiving endpoint. It owns its Conn and streams, not
// application session resources. Use Close followed by Wait before releasing
// those resources. An Agent must not be copied.
type Agent struct {
	conn                      *Conn
	handlers                  AgentHandlers
	maxSessions               int
	mu                        sync.Mutex
	initializing              bool
	initialized               bool
	initRequest, initResponse json.RawMessage
	sessions                  map[protocol.SessionID]*agentSession
	slots                     int
}

type agentSession struct {
	ready bool
	turn  *Turn
}

// NewAgent starts the typed endpoint. All three callbacks are required. The
// supported protocol version is 1; other versions are rejected without calling
// Initialize. Invalid/failed initialization may be retried; successful
// initialization is once per connection. Requests pipelined before its response
// is committed are rejected, not queued behind application initialization.
func NewAgent(ctx context.Context, input io.ReadCloser, output io.WriteCloser, handlers AgentHandlers, options AgentOptions) (*Agent, error) {
	if handlers.Initialize == nil || handlers.CreateSession == nil || handlers.Prompt == nil {
		return nil, errors.New("acp: all agent callbacks are required")
	}
	if options.MaxSessions < 0 {
		return nil, errors.New("acp: negative session limit")
	}
	if options.MaxSessions == 0 {
		options.MaxSessions = 64
	}
	a := &Agent{handlers: handlers, maxSessions: options.MaxSessions, sessions: make(map[protocol.SessionID]*agentSession)}
	c, err := newConn(ctx, input, output, Handlers{}, options.Limits, a)
	if err != nil {
		return nil, err
	}
	a.conn = c
	return a, nil
}

// Close begins shutdown; it does not wait for application callbacks.
func (a *Agent) Close() error { return a.conn.Close() }

// Done signals shutdown beginning.
func (a *Agent) Done() <-chan struct{} { return a.conn.Done() }

// Err returns the connection's shutdown cause.
func (a *Agent) Err() error { return a.conn.Err() }

// Wait joins transport, callbacks and admitted turn operations. Its context
// bounds waiting, not execution: uncooperative application work cannot be killed.
func (a *Agent) Wait(ctx context.Context) error { return a.conn.Wait(ctx) }

// Initialization returns independent snapshots of the negotiated initialization
// messages, including both capability sets. Before success it returns false.
// This records capabilities; it does not implement application policy for them.
func (a *Agent) Initialization() (protocol.InitializeRequest, protocol.InitializeResponse, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	var req protocol.InitializeRequest
	var res protocol.InitializeResponse
	if !a.initialized {
		return req, res, false
	}
	_ = json.Unmarshal(a.initRequest, &req)
	_ = json.Unmarshal(a.initResponse, &res)
	return req, res, true
}

func agentReject(code int64, message string) (func() (any, error), func()) {
	return func() (any, error) { return nil, &Error{Code: code, Message: message} }, nil
}

// admit runs only on the reader, after transport capacity/ID reservation. It
// performs bounded decoding and bookkeeping, never application callbacks or I/O.
// commit runs on the writer before a response becomes observable.
func (a *Agent) admit(c *Conn, m message) (func() (any, error), func()) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if m.method == "initialize" {
		var req protocol.InitializeRequest
		if json.Unmarshal(m.params, &req) != nil || req.ProtocolVersion != 1 {
			return agentReject(-32602, "Invalid initialization")
		}
		if a.initializing || a.initialized {
			return agentReject(-32600, "Already initialized or initializing")
		}
		a.initializing = true
		var encoded json.RawMessage
		return func() (any, error) {
				res, err := a.handlers.Initialize(c.ctx, req)
				if err != nil {
					return nil, err
				}
				if res.ProtocolVersion != 1 {
					return nil, errors.New("acp: unsupported initialization response version")
				}
				b, err := json.Marshal(res)
				if err == nil {
					encoded = b
				}
				return json.RawMessage(b), err
			}, func() {
				a.mu.Lock()
				defer a.mu.Unlock()
				a.initializing = false
				if encoded != nil {
					a.initialized = true
					a.initRequest = append(json.RawMessage(nil), m.params...)
					a.initResponse = encoded
				}
			}
	}
	if m.method != "session/new" && m.method != "session/prompt" {
		return agentReject(-32601, "Method not found")
	}
	if !a.initialized {
		return agentReject(-32600, "Not initialized")
	}
	if m.method == "session/new" {
		var req protocol.NewSessionRequest
		if json.Unmarshal(m.params, &req) != nil {
			return agentReject(-32602, "Invalid session parameters")
		}
		if a.slots >= a.maxSessions {
			return agentReject(-32001, "Session limit reached")
		}
		a.slots++
		var session *agentSession
		return func() (any, error) {
				res, err := a.handlers.CreateSession(c.ctx, req)
				if err != nil {
					return nil, err
				}
				b, err := json.Marshal(res)
				if err != nil {
					return nil, err
				}
				a.mu.Lock()
				defer a.mu.Unlock()
				if res.SessionID == "" || a.sessions[res.SessionID] != nil {
					return nil, errors.New("acp: empty or duplicate session ID")
				}
				session = &agentSession{}
				a.sessions[res.SessionID] = session
				return json.RawMessage(b), nil
			}, func() {
				a.mu.Lock()
				defer a.mu.Unlock()
				if session == nil {
					a.slots--
				} else {
					session.ready = true
				}
			}
	}
	var req protocol.PromptRequest
	if json.Unmarshal(m.params, &req) != nil {
		return agentReject(-32602, "Invalid prompt parameters")
	}
	session := a.sessions[req.SessionID]
	if session == nil || !session.ready {
		return agentReject(-32002, "Unknown session")
	}
	if session.turn != nil {
		return agentReject(-32001, "Prompt already active")
	}
	ctx, cancel := context.WithCancel(c.ctx)
	turn := &Turn{conn: c, sessionID: req.SessionID, ctx: ctx, cancel: cancel}
	session.turn = turn
	return func() (any, error) {
			defer turn.finish()
			res, err := a.handlers.Prompt(ctx, turn, req)
			if ctx.Err() != nil && (err == nil || errors.Is(err, context.Canceled)) {
				res = protocol.PromptResponse{StopReason: protocol.StopReasonCancelled}
				err = nil
			}
			return res, err
		}, func() {
			a.mu.Lock()
			defer a.mu.Unlock()
			session.turn = nil
		}
}

func (a *Agent) notification(m message) error {
	if m.method != "session/cancel" {
		return nil
	}
	var req protocol.CancelNotification
	if err := json.Unmarshal(m.params, &req); err != nil {
		return err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if session := a.sessions[req.SessionID]; session != nil && session.turn != nil {
		session.turn.cancel()
	}
	return nil
}

// ErrTurnFinished means the callback has returned and its outbound-work gate is
// closed. Retaining a Turn never grants access to a successor prompt.
var ErrTurnFinished = errors.New("acp: turn finished")

// Turn scopes outbound updates and permissions to one admitted prompt. Methods
// may run concurrently. Prompt must join its own application goroutines before
// returning; Agent joins admitted outbound calls, not arbitrary application work.
// Once Prompt returns, new outbound calls fail with ErrTurnFinished.
type Turn struct {
	conn      *Conn
	sessionID protocol.SessionID
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.Mutex
	finished  bool
	work      sync.WaitGroup
}

func (t *Turn) begin() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.finished {
		return ErrTurnFinished
	}
	t.work.Add(1)
	return nil
}

func (t *Turn) finish() {
	t.mu.Lock()
	t.finished = true
	t.cancel()
	t.mu.Unlock()
	t.work.Wait()
}

// SendSessionUpdate binds the notification to this session (the supplied
// SessionID is ignored). It remains usable during cooperative cancellation for
// final updates. Writes use connection lifetime, not turn cancellation; a stuck
// peer requires connection shutdown. Completion means written, not rendered.
func (t *Turn) SendSessionUpdate(update protocol.SessionNotification) error {
	if err := t.begin(); err != nil {
		return err
	}
	defer t.work.Done()
	update.SessionID = t.sessionID
	return t.conn.SendSessionUpdate(t.conn.ctx, update)
}

// RequestPermission binds the request to this session and waits with the turn's
// context. Cancellation stops waiting only after the entire request frame has
// been written; it cannot interrupt an active write and corrupt the connection.
// A blocked writer therefore requires connection shutdown to unblock. No remote
// permission cancellation is sent; late replies are discarded. The application
// must validate selected options and enforce authorization itself.
func (t *Turn) RequestPermission(request protocol.RequestPermissionRequest) (protocol.RequestPermissionResponse, error) {
	var res protocol.RequestPermissionResponse
	if err := t.begin(); err != nil {
		return res, err
	}
	defer t.work.Done()
	request.SessionID = t.sessionID
	params, err := marshalProtocol("session/request_permission", request)
	if err != nil {
		return res, err
	}
	var wire json.RawMessage
	if err := t.conn.call(t.ctx, t.conn.ctx, "session/request_permission", params, &wire); err != nil {
		return res, err
	}
	err = json.Unmarshal(wire, &res)
	return res, err
}
