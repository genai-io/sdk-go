// Package errs turns one protocol's failure into a classified ai.Error.
//
// Reading a failure is protocol-specific — each SDK has its own error type, and
// the Gemini driver has no SDK at all — but nothing after that differs: the
// status, code, message and response go to ai.Classify in the same order every
// time. Only the reading is worth writing once per protocol, so only the
// reading lives in the drivers.
//
// It sits under driver/ rather than in ai so that the compiler, not a
// convention, limits it to the packages that implement a wire protocol.
package errs

import (
	"net/http"

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

// Reader pulls Details out of one protocol's error type.
type Reader func(error) Details

// Classifier is one driver's pair of error constructors, bound to the name
// that will appear as ai.Error.Driver.
type Classifier struct {
	driver string
	read   Reader
}

// For binds a driver name to the reader for its protocol's error type.
func For(driver string, read Reader) Classifier {
	return Classifier{driver: driver, read: read}
}

// Wrap classifies a failure from a non-streaming call.
func (c Classifier) Wrap(err error) error {
	if err == nil {
		return nil
	}
	d := c.read(err)
	return ai.Classify(c.driver, d.Status, d.Response, d.Code, d.Message, err)
}

// WrapStream classifies a failure that ended a stream, where the transport
// commonly loses the protocol's typed error.
func (c Classifier) WrapStream(err error) error {
	if err == nil {
		return nil
	}
	d := c.read(err)
	return ai.StreamError(c.driver, d.Status, d.Response, d.Code, d.Message, err)
}
