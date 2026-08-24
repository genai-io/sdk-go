package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
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
	// KindContextExceeded means the prompt is larger than the context window.
	// Retrying the same prompt cannot help; compacting it can.
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

// IsUnsupported reports whether err is a request the model cannot serve —
// images to a text-only endpoint, tools to a model without them, a retired
// model. Retrying or switching credentials cannot help; changing the request
// or the model can.
func IsUnsupported(err error) bool { return IsKind(err, KindUnsupported) }

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

func (e *Error) Error() string {
	var sb strings.Builder
	if e.Driver != "" {
		sb.WriteString(e.Driver)
		sb.WriteString(": ")
	}
	sb.WriteString(string(e.Kind))
	if e.Status != 0 {
		fmt.Fprintf(&sb, " (http %d)", e.Status)
	}
	msg := e.Message
	if msg == "" && e.Err != nil {
		msg = e.Err.Error()
	}
	if msg != "" {
		sb.WriteString(": ")
		sb.WriteString(msg)
	}
	return sb.String()
}

func (e *Error) Unwrap() error { return e.Err }

// Retryable reports whether trying the same request again could succeed.
func (e *Error) Retryable() bool {
	switch e.Kind {
	case KindRateLimit, KindOverloaded, KindNetwork:
		return true
	default:
		return false
	}
}

// IsRetryable reports whether err is a transient failure worth retrying.
func IsRetryable(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Retryable()
}

// IsKind reports whether err is a provider error of the given kind.
func IsKind(err error, kind ErrorKind) bool {
	var e *Error
	return errors.As(err, &e) && e.Kind == kind
}

// IsAuth reports whether err is a credential failure.
func IsAuth(err error) bool { return IsKind(err, KindAuth) }

// IsContextExceeded reports whether err says the prompt exceeds the model's
// context window — the signal to compact and retry rather than to give up.
func IsContextExceeded(err error) bool { return IsKind(err, KindContextExceeded) }

// RetryAfter returns the provider's rate-limit hint, or 0.
func RetryAfter(err error) time.Duration {
	var e *Error
	if errors.As(err, &e) {
		return e.RetryAfter
	}
	return 0
}

// contextExceededSignatures are the ways providers say "this prompt exceeds
// the context window". Matching is on message text because no provider
// distinguishes it from other 400s with a machine-readable code.
var contextExceededSignatures = []string{
	"prompt is too long",                // Anthropic
	"prompt_too_long",                   // Anthropic error type
	"maximum context length",            // OpenAI and compatibles
	"context_length_exceeded",           // OpenAI error code
	"reduce the length of the messages", // OpenAI remediation text
	"input token count",                 // Google Gemini
	"exceeds the maximum number of tokens",
	"context length exceeded",
	"too many tokens",
}

// notContextExceededSignatures are messages that match a context-exceeded
// signature but mean something else entirely.
var notContextExceededSignatures = []string{
	"rate limit",
	"too many requests",
	"throttling",
	"quota exceeded",
	"please wait before",
}

// ClassifyMessage reports whether a provider message says the prompt exceeded
// the context window. Drivers call it before status classification, because
// the condition arrives as an ordinary 400.
func ClassifyMessage(msg string) (ErrorKind, bool) {
	lower := strings.ToLower(msg)
	for _, sig := range notContextExceededSignatures {
		if strings.Contains(lower, sig) {
			return KindUnknown, false
		}
	}
	for _, sig := range contextExceededSignatures {
		if strings.Contains(lower, sig) {
			return KindContextExceeded, true
		}
	}
	return KindUnknown, false
}

// classifyStatus maps an HTTP status onto a Kind, extracting Retry-After from
// resp for a 429. resp may be nil.
func classifyStatus(status int, resp *http.Response) (ErrorKind, time.Duration) {
	switch {
	case status == http.StatusTooManyRequests:
		return KindRateLimit, parseRetryAfter(resp)
	case status == http.StatusRequestTimeout,
		status == http.StatusConflict,
		status >= 500:
		return KindOverloaded, 0
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return KindAuth, 0
	case status >= 400:
		return KindInvalidRequest, 0
	default:
		return KindUnknown, 0
	}
}

// classifyTransport classifies an error carrying no HTTP status: cancellation,
// a dropped connection, or a timeout. It reports false when err is none of
// those, leaving the driver to fall back to its own signals.
func classifyTransport(err error) (ErrorKind, bool) {
	switch {
	case err == nil:
		return KindUnknown, false
	case errors.Is(err, context.Canceled):
		return KindCanceled, true
	case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
		// A mid-stream cutoff. Worth another attempt.
		return KindNetwork, true
	}
	// net.Error also matches a per-request timeout — including
	// context.DeadlineExceeded — which is likewise worth retrying. A user
	// interrupt was already caught above and stays fatal.
	var netErr net.Error
	if errors.As(err, &netErr) {
		return KindNetwork, true
	}
	return KindUnknown, false
}

// parseRetryAfter reads a Retry-After header in either allowed form,
// delta-seconds or an HTTP date. It returns 0 when absent or unparseable.
func parseRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	v := strings.TrimSpace(resp.Header.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs <= 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// Classify assembles an *Error from what a driver has to hand. It applies the
// checks in the order that keeps each from masking the next:
func Classify(driver string, status int, resp *http.Response, code, message string, err error) *Error {
	var typed *Error
	if errors.As(err, &typed) {
		return typed
	}
	out := &Error{Driver: driver, Status: status, Code: code, Message: message, Err: err}
	if out.Message == "" && err != nil {
		out.Message = err.Error()
	}

	if err != nil && errors.Is(err, context.Canceled) {
		out.Kind = KindCanceled
		return out
	}
	if kind, ok := ClassifyMessage(out.Message); ok {
		out.Kind = kind
		return out
	}
	if status != 0 {
		out.Kind, out.RetryAfter = classifyStatus(status, resp)
		return out
	}
	if kind, ok := classifyTransport(err); ok {
		out.Kind = kind
		return out
	}
	out.Kind = KindUnknown
	return out
}

// StreamError is Classify for an error that terminated a stream.
func StreamError(driver string, status int, resp *http.Response, code, message string, err error) *Error {
	out := Classify(driver, status, resp, code, message, err)
	if out.Kind == KindUnknown {
		out.Kind = KindNetwork
	}
	return out
}
