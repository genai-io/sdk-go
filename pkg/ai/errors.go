package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"time"
)

// ErrorKind is the provider-agnostic failure category. It is what a caller
// needs in order to decide between retrying, compacting the prompt, asking for
// a credential, and giving up — none of which can be decided from a message
// string.
type ErrorKind string

const (
	// KindUnknown is a failure with no typed signal. Treat it as fatal unless
	// it terminated a stream, where the transport commonly loses the type.
	KindUnknown ErrorKind = "unknown"
	// KindAuth is a rejected or missing credential (401/403).
	KindAuth ErrorKind = "auth"
	// KindInvalidRequest is a malformed or unacceptable request (400/404/422).
	KindInvalidRequest ErrorKind = "invalid_request"
	// KindContextExceeded means the prompt is too large to accept: past the
	// context window, or a request body too big to send (413, or a 408 timed
	// out reading it). Retrying the same prompt cannot help; compacting it can.
	KindContextExceeded ErrorKind = "context_exceeded"
	// KindRateLimit is a 429. RetryAfter carries the provider's hint when it
	// sent one.
	KindRateLimit ErrorKind = "rate_limit"
	// KindOverloaded is a transient server-side failure (408, 409, 5xx,
	// including Anthropic's 529).
	KindOverloaded ErrorKind = "overloaded"
	// KindNetwork is a transport failure: dropped, refused or reset
	// connection, or a timeout.
	KindNetwork ErrorKind = "network"
	// KindCanceled means the caller's context ended the call.
	KindCanceled ErrorKind = "canceled"
	// KindUnsupported means the request asks the model for something it cannot
	// do. It is caught before the network, so nothing was spent.
	KindUnsupported ErrorKind = "unsupported"
)

// Error is a provider failure, classified.
type Error struct {
	// Driver is the wire protocol that produced the error.
	Driver string
	// Kind is the failure category.
	Kind ErrorKind
	// Status is the HTTP status, or 0 when there was none.
	Status int
	// Code is the provider's machine-readable error code, when it sent one.
	Code string
	// Message is the provider's human-readable message.
	Message string
	// RetryAfter is the provider's rate-limit hint; 0 when absent.
	RetryAfter time.Duration
	// Err is the underlying SDK or transport error.
	Err error
}

// Error renders whichever of the four parts are set. Each separator belongs to
// the part that follows it, so an error carrying no kind reads as an error
// rather than as "driver: : message".
func (e *Error) Error() string {
	var sb strings.Builder
	if e.Driver != "" {
		sb.WriteString(e.Driver)
	}
	if e.Kind != "" {
		if sb.Len() > 0 {
			sb.WriteString(": ")
		}
		sb.WriteString(string(e.Kind))
	}
	if e.Status != 0 {
		if sb.Len() > 0 {
			sb.WriteByte(' ')
		}
		fmt.Fprintf(&sb, "(http %d)", e.Status)
	}
	msg := e.Message
	if msg == "" && e.Err != nil {
		msg = e.Err.Error()
	}
	if msg != "" {
		if sb.Len() > 0 {
			sb.WriteString(": ")
		}
		sb.WriteString(msg)
	}
	if sb.Len() == 0 {
		return "ai: unspecified error"
	}
	return sb.String()
}

func (e *Error) Unwrap() error { return e.Err }

// IsRetryable reports whether err is a transient failure worth retrying.
//
// A classified error answers from its kind. Anything else is put to the same
// transport test a driver applies: a dropped connection or a timeout is worth
// another attempt whether or not it passed through a driver on the way here,
// and an application should not have to reconstruct an *Error to be told so.
func IsRetryable(err error) bool {
	kind, ok := ErrorKind(""), true
	if e, isErr := errors.AsType[*Error](err); isErr {
		kind = e.Kind
	} else {
		kind, ok = classifyTransport(err)
	}
	return ok && (kind == KindRateLimit || kind == KindOverloaded || kind == KindNetwork)
}

// IsKind reports whether err is a provider error of the given kind.
func IsKind(err error, kind ErrorKind) bool {
	e, ok := errors.AsType[*Error](err)
	return ok && e.Kind == kind
}

// IsAuth reports whether err is a credential failure.
func IsAuth(err error) bool { return IsKind(err, KindAuth) }

// IsContextExceeded reports whether err says the prompt exceeds the model's
// context window — the signal to compact and retry rather than to give up.
func IsContextExceeded(err error) bool { return IsKind(err, KindContextExceeded) }

// RetryAfter returns the provider's rate-limit hint, or 0.
func RetryAfter(err error) time.Duration {
	if e, ok := errors.AsType[*Error](err); ok {
		return e.RetryAfter
	}
	return 0
}

// classifyTransport classifies an error carrying no HTTP status: cancellation,
// a dropped connection, or a timeout. It reports false when err is none of
// those, leaving the driver to fall back to its own signals.
func classifyTransport(err error) (ErrorKind, bool) {
	switch {
	case errors.Is(err, context.Canceled):
		return KindCanceled, true
	case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
		// A mid-stream cutoff. Worth another attempt.
		return KindNetwork, true
	}
	// net.Error also matches a per-request timeout — including
	// context.DeadlineExceeded — which is likewise worth retrying. A user
	// interrupt was already caught above and stays fatal.
	if _, ok := errors.AsType[net.Error](err); ok {
		return KindNetwork, true
	}
	return KindUnknown, false
}
