package acp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"
	"sync/atomic"
)

var (
	ErrClosed           = errors.New("acp: connection closed")
	ErrLimit            = errors.New("acp: connection limit exceeded")
	ErrFrameTooLarge    = errors.New("acp: frame too large")
	ErrInvalidMessage   = errors.New("acp: invalid message")
	ErrNotificationCall = errors.New("acp: synchronous call from notification handler")
)

// Limits bounds connection-owned work. Defaults are 1 MiB per frame,
// 8 requests, 32 pending calls, 64 notifications and 16 writes.
// Zero selects a default; negative values fail.
type Limits struct {
	FrameBytes    int
	Requests      int
	PendingCalls  int
	Notifications int
	Writes        int
}

func (l Limits) defaults() (Limits, error) {
	fields := []*int{&l.FrameBytes, &l.Requests, &l.PendingCalls, &l.Notifications, &l.Writes}
	defaults := []int{1 << 20, 8, 32, 64, 16}
	for i, field := range fields {
		if *field < 0 {
			return l, fmt.Errorf("%w: negative limit", ErrLimit)
		}
		if *field == 0 {
			*field = defaults[i]
		}
	}
	if l.FrameBytes > 64<<20 {
		return l, fmt.Errorf("%w: frame limit exceeds 64 MiB", ErrLimit)
	}
	return l, nil
}

// Handlers receive connection-bound contexts. Requests may call the peer back.
// Notifications run serially and must return promptly; they must not Call using
// their notification context, since responses wait for preceding notifications.
type Handlers struct {
	Request      func(context.Context, *Conn, string, json.RawMessage) (any, error)
	Notification func(context.Context, *Conn, string, json.RawMessage) error
}

type reply struct {
	message
	through uint64
}

type notification struct {
	message
	sequence uint64
}

type incoming struct{ release sync.Once }

type write struct {
	ctx    context.Context
	data   []byte
	done   chan error
	commit func()
	state  atomic.Int32 // queued, writing, cancelled, finished
}

// Conn is a bidirectional, newline-delimited JSON-RPC connection for ACP.
// Invalid frames and exhausted inbound/write capacity terminate the connection.
type Conn struct {
	ctx      context.Context
	cancel   context.CancelCauseFunc
	input    io.ReadCloser
	output   io.WriteCloser
	handlers Handlers
	limits   Limits

	writes        chan *write
	notifications chan notification
	requests      chan struct{}
	wg            sync.WaitGroup
	joined        chan struct{}
	closed        chan struct{}
	once          sync.Once
	next          atomic.Uint64

	mu        sync.Mutex
	pending   map[string]chan reply
	inbound   map[string]*incoming
	delivered uint64
	progress  chan struct{}
}

// New takes ownership of input/output on success. Closing either stream must
// interrupt its active I/O. No process stdio, logger, or global state is changed.
func New(ctx context.Context, input io.ReadCloser, output io.WriteCloser, handlers Handlers, limits Limits) (*Conn, error) {
	if ctx == nil || input == nil || output == nil {
		return nil, errors.New("acp: context and closable streams are required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limits, err := limits.defaults()
	if err != nil {
		return nil, err
	}
	life, cancel := context.WithCancelCause(context.Background())
	c := &Conn{
		ctx: life, cancel: cancel, input: input, output: output,
		handlers: handlers, limits: limits,
		writes: make(chan *write, limits.Writes), notifications: make(chan notification, limits.Notifications),
		requests: make(chan struct{}, limits.Requests), joined: make(chan struct{}), closed: make(chan struct{}),
		pending: make(map[string]chan reply), inbound: make(map[string]*incoming), progress: make(chan struct{}),
	}
	stopParent := context.AfterFunc(ctx, func() { c.shutdown(context.Cause(ctx)) })
	c.wg.Add(3)
	go c.read()
	go c.write()
	go c.notify()
	go func() {
		c.wg.Wait()
		<-c.closed
		stopParent()
		close(c.joined)
	}()
	return c, nil
}

// Call waits for a response and its preceding notifications. Cancelling the
// context stops this wait, not remote application work: send session/cancel for
// a prompt, or Close to terminate the peer connection.
func (c *Conn) Call(ctx context.Context, method string, params any, result any) error {
	if ctx.Value(notificationContext{}) == c {
		return ErrNotificationCall
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	id, _ := json.Marshal(strconv.FormatUint(c.next.Add(1), 10))
	frame, err := requestFrame(id, method, params)
	if err != nil {
		return err
	}
	key, _ := idKey(id)
	ch := make(chan reply, 1)
	c.mu.Lock()
	if len(c.pending) >= c.limits.PendingCalls {
		c.mu.Unlock()
		return ErrLimit
	}
	c.pending[key] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pending, key)
		c.mu.Unlock()
	}()
	if err := c.send(ctx, frame); err != nil {
		return err
	}
	var response reply
	select {
	case response = <-ch:
	case <-ctx.Done():
		return ctx.Err()
	case <-c.Done():
		select {
		case response = <-ch:
		default:
			return c.Err()
		}
	}
	if err := c.waitNotifications(ctx, response.through); err != nil {
		return err
	}
	if response.err != nil {
		return response.err
	}
	if result != nil {
		return json.Unmarshal(response.result, result)
	}
	return nil
}

// Notify sends a notification. Cancellation after its write begins terminates
// the connection: a partially written JSON frame cannot safely be reused.
func (c *Conn) Notify(ctx context.Context, method string, params any) error {
	frame, err := requestFrame(nil, method, params)
	if err != nil {
		return err
	}
	return c.send(ctx, frame)
}

func (c *Conn) send(ctx context.Context, frame []byte) error {
	return c.sendCommit(ctx, frame, nil)
}

func (c *Conn) sendCommit(ctx context.Context, frame []byte, commit func()) error {
	if len(frame) > c.limits.FrameBytes {
		return ErrFrameTooLarge
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := c.Err(); err != nil {
		return err
	}
	job := &write{ctx: ctx, data: append(frame, '\n'), done: make(chan error, 1), commit: commit}
	select {
	case c.writes <- job:
	default:
		c.shutdown(ErrLimit)
		return ErrLimit
	}
	select {
	case err := <-job.done:
		return err
	case <-ctx.Done():
		if !job.state.CompareAndSwap(0, 2) && job.state.Load() == 1 {
			c.shutdown(ctx.Err())
		}
		return ctx.Err()
	case <-c.Done():
		return c.Err()
	}
}

func (c *Conn) write() {
	defer c.wg.Done()
	for {
		select {
		case <-c.Done():
			return
		case job := <-c.writes:
			if !job.state.CompareAndSwap(0, 1) {
				continue
			}
			if err := job.ctx.Err(); err != nil {
				job.done <- err
				job.state.Store(3)
				continue
			}
			// Retire request ownership before the peer can observe its response.
			if job.commit != nil {
				job.commit()
			}
			n, err := c.output.Write(job.data)
			if err == nil && n != len(job.data) {
				err = io.ErrShortWrite
			}
			job.done <- err
			job.state.Store(3)
			if err != nil {
				c.shutdown(err)
				return
			}
		}
	}
}

func (c *Conn) read() {
	defer c.wg.Done()
	r := bufio.NewReaderSize(c.input, min(4096, c.limits.FrameBytes+1))
	var sequence uint64
	for {
		frame, err := readFrame(r, c.limits.FrameBytes)
		if err != nil {
			c.shutdown(err)
			return
		}
		m, err := decode(frame)
		if err != nil {
			c.shutdown(err)
			return
		}
		switch {
		case m.method == "":
			c.mu.Lock()
			ch := c.pending[m.key]
			c.mu.Unlock()
			if ch != nil {
				select {
				case ch <- reply{message: m, through: sequence}:
				default:
				}
			}
		case len(m.id) == 0:
			sequence++
			select {
			case c.notifications <- notification{message: m, sequence: sequence}:
			default:
				c.shutdown(ErrLimit)
				return
			}
		default:
			if err := c.dispatch(m); err != nil {
				c.shutdown(err)
				return
			}
		}
	}
}

func (c *Conn) dispatch(m message) error {
	entry := &incoming{}
	c.mu.Lock()
	if _, duplicate := c.inbound[m.key]; duplicate {
		c.mu.Unlock()
		return ErrInvalidMessage
	}
	select {
	case c.requests <- struct{}{}:
		c.inbound[m.key] = entry
	default:
		c.mu.Unlock()
		return ErrLimit
	}
	c.mu.Unlock()
	release := func() {
		entry.release.Do(func() {
			c.mu.Lock()
			if c.inbound[m.key] == entry {
				delete(c.inbound, m.key)
			}
			<-c.requests
			c.mu.Unlock()
		})
	}
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		defer release()
		result, failure := c.invoke(m)
		frame, err := responseFrame(m.id, result, failure)
		if err != nil {
			frame, _ = responseFrame(m.id, nil, &Error{Code: -32603, Message: "Internal error"})
		}
		if err := c.sendCommit(c.ctx, frame, release); err != nil {
			c.shutdown(err)
		}
	}()
	return nil
}

func (c *Conn) invoke(m message) (result any, failure *Error) {
	defer func() {
		if recover() != nil {
			result, failure = nil, &Error{Code: -32603, Message: "Internal error"}
		}
	}()
	if c.handlers.Request == nil {
		return nil, &Error{Code: -32601, Message: "Method not found"}
	}
	value, err := c.handlers.Request(c.ctx, c, m.method, m.params)
	if err == nil {
		return value, nil
	}
	if errors.As(err, &failure) {
		return nil, failure
	}
	return nil, &Error{Code: -32603, Message: "Internal error"}
}

type notificationContext struct{}

func (c *Conn) notify() {
	defer c.wg.Done()
	defer func() {
		if recover() != nil {
			c.shutdown(errors.New("acp: notification handler panicked"))
		}
	}()
	ctx := context.WithValue(c.ctx, notificationContext{}, c)
	for {
		select {
		case <-c.Done():
			return
		case n := <-c.notifications:
			if c.handlers.Notification != nil {
				if err := c.handlers.Notification(ctx, c, n.method, n.params); err != nil {
					c.shutdown(err)
					return
				}
			}
			c.mu.Lock()
			c.delivered = n.sequence
			close(c.progress)
			c.progress = make(chan struct{})
			c.mu.Unlock()
		}
	}
}

func (c *Conn) waitNotifications(ctx context.Context, through uint64) error {
	for {
		c.mu.Lock()
		if c.delivered >= through {
			c.mu.Unlock()
			return nil
		}
		progress := c.progress
		c.mu.Unlock()
		select {
		case <-progress:
		case <-ctx.Done():
			return ctx.Err()
		case <-c.Done():
			return c.Err()
		}
	}
}

func (c *Conn) shutdown(cause error) {
	c.once.Do(func() {
		c.cancel(cause)
		_ = c.input.Close()
		_ = c.output.Close()
		close(c.closed)
	})
}

func (c *Conn) Close() error {
	c.shutdown(ErrClosed)
	return nil
}

// Done signals shutdown beginning; Wait also joins handlers and I/O workers.
func (c *Conn) Done() <-chan struct{} { return c.ctx.Done() }
func (c *Conn) Err() error            { return context.Cause(c.ctx) }

func (c *Conn) Wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-c.joined:
		return c.Err()
	}
}
