// Package errs turns one protocol's failure into a classified ai.Error.
//
// Reading a failure is protocol-specific — each SDK has its own error type, and
// the Gemini driver has no SDK at all — but nothing after that differs: the
// status, code, message and response go to Classify in the same order every
// time. Only the reading is worth writing once per protocol, so only the
// reading lives in the drivers.
//
// It sits under driver/ rather than in ai so that the compiler, not a
// convention, limits it to the packages that implement a wire protocol.
package errs

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Details is what a protocol's own error type says about a failed call. The
// zero value is a failure it did not produce, which leaves classification to
// fall back to the transport.
type Details struct {
	// Status is the HTTP status the endpoint answered with, or 0.
	Status int
	// Code is the provider's machine-readable error code.
	Code string
	// Message is the provider's human-readable message.
	Message string
	// Response is kept so a 429's Retry-After is honoured rather than replaced
	// by the caller's own backoff.
	Response *http.Response
}

// Classifier is one driver's pair of error constructors, bound to the name
// that will appear as ai.Error.Driver.
type Classifier struct {
	driver string
	read   func(error) Details
}

// For binds a driver name to the reader for its protocol's error type.
func For(driver string, read func(error) Details) Classifier {
	return Classifier{driver: driver, read: read}
}

// Wrap classifies a failure from a non-streaming call.
func (c Classifier) Wrap(err error) error {
	if err == nil {
		return nil
	}
	d := c.read(err)
	return Classify(c.driver, d.Status, d.Response, d.Code, d.Message, err)
}

// WrapStream classifies a failure that ended a stream, where the transport
// commonly loses the protocol's typed error.
func (c Classifier) WrapStream(err error) error {
	if err == nil {
		return nil
	}
	d := c.read(err)
	return ClassifyStream(c.driver, d.Status, d.Response, d.Code, d.Message, err)
}

// Classify assembles an *ai.Error from what a driver has to hand. It applies
// the checks in the order that keeps each from masking the next: an already
// classified error, then a cancel, then the message, then the status, with the
// transport as the last word.
func Classify(driver string, status int, resp *http.Response, code, message string, err error) *ai.Error {
	var typed *ai.Error
	if errors.As(err, &typed) {
		return typed
	}
	out := &ai.Error{Driver: driver, Status: status, Code: code, Message: message, Err: err}
	if out.Message == "" && err != nil {
		out.Message = err.Error()
	}

	switch {
	case err != nil && errors.Is(err, context.Canceled):
		out.Kind = ai.KindCanceled
	// A context overflow arrives as an ordinary 400 with no machine-readable
	// code, so the message is the only place to read it from — but only on a
	// 400: a 401 or 503 mentioning tokens is a key or a busy server, not this.
	case (status == 0 || status == http.StatusBadRequest) && IsContextExceededMessage(out.Message):
		out.Kind = ai.KindContextExceeded
	case status != 0:
		out.Kind, out.RetryAfter = classifyStatus(status, resp)
	case ai.IsRetryable(err):
		// A bare transport failure: a dropped connection or a timeout.
		out.Kind = ai.KindNetwork
	default:
		out.Kind = ai.KindUnknown
	}
	return out
}

// ClassifyStream is Classify for an error that terminated a stream.
func ClassifyStream(driver string, status int, resp *http.Response, code, message string, err error) *ai.Error {
	out := Classify(driver, status, resp, code, message, err)
	if out.Kind == ai.KindUnknown {
		out.Kind = ai.KindNetwork
	}
	return out
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

// IsContextExceededMessage reports whether a provider message says the prompt
// exceeded the context window.
func IsContextExceededMessage(msg string) bool {
	lower := strings.ToLower(msg)
	for _, sig := range notContextExceededSignatures {
		if strings.Contains(lower, sig) {
			return false
		}
	}
	for _, sig := range contextExceededSignatures {
		if strings.Contains(lower, sig) {
			return true
		}
	}
	return false
}

// classifyStatus maps an HTTP status onto a Kind, extracting Retry-After from
// resp for a 429. resp may be nil.
func classifyStatus(status int, resp *http.Response) (ai.ErrorKind, time.Duration) {
	switch {
	case status == http.StatusTooManyRequests:
		return ai.KindRateLimit, parseRetryAfter(resp)
	case status == http.StatusRequestTimeout,
		status == http.StatusConflict,
		status >= 500:
		return ai.KindOverloaded, 0
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return ai.KindAuth, 0
	case status >= 400:
		return ai.KindInvalidRequest, 0
	default:
		return ai.KindUnknown, 0
	}
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
