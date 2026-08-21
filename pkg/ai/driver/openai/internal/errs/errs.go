// Package errs reads a failure out of the OpenAI Go SDK.
//
// The chat and responses drivers implement two different protocols, but they
// share one vendor SDK and therefore one error shape.
// Reading it lives here rather than in both drivers, where two copies would
// drift the first time the SDK moved a field — and drift in error
// classification is silent: a rate limit that stops being recognised simply
// stops being retried.
//
// It is internal to ai/driver/openai, so the compiler — not a convention —
// keeps the Anthropic and Gemini drivers away from the OpenAI SDK's types.
package errs

import (
	"errors"
	"net/http"
	"strings"

	sdk "github.com/openai/openai-go/v3"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Details reports what the SDK knows about a failed call. Everything is zero
// when err did not come from the SDK, which leaves ai.Classify to fall back
// to the transport.
func Details(err error) (status int, code, message string, resp *http.Response) {
	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		return 0, "", "", nil
	}
	message = strings.TrimSpace(apiErr.Message)
	if message == "" {
		// A body the SDK could not parse into a message still says more than
		// an empty string does.
		message = strings.TrimSpace(apiErr.RawJSON())
	}
	return apiErr.StatusCode, apiErr.Code, message, apiErr.Response
}

// Wrap classifies a failure from a non-streaming call.
func Wrap(driver string, err error) error {
	status, code, msg, resp := Details(err)
	return ai.Classify(driver, status, resp, code, msg, err)
}

// WrapStream classifies a failure that terminated a stream, where the
// transport commonly loses the SDK's typed error.
func WrapStream(driver string, err error) error {
	status, code, msg, resp := Details(err)
	return ai.StreamError(driver, status, resp, code, msg, err)
}
