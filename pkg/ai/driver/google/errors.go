package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Classifying this protocol's failures into ai.Error kinds.

// readAPIError turns a failed response into the driver's own error, keeping
// the response so a 429's Retry-After is honoured.
func readAPIError(res *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	var envelope apiError
	_ = json.Unmarshal(raw, &envelope)
	message := strings.TrimSpace(envelope.Error.Message)
	if message == "" {
		message = strings.TrimSpace(string(raw))
	}
	return &statusError{status: res.StatusCode, code: envelope.Error.Status, message: message, response: res}
}

// statusError carries what the driver knows about a failed response through to
// classification.
type statusError struct {
	status   int
	code     string
	message  string
	response *http.Response
}

func (e *statusError) Error() string {
	if e.code != "" {
		return fmt.Sprintf("%s: %s", e.code, e.message)
	}
	return e.message
}

func (d *Driver) wrap(err error) error {
	status, resp, code, msg := errorDetails(err)
	return ai.Classify(Name, status, resp, code, msg, err)
}

func (d *Driver) wrapStream(err error) error {
	status, resp, code, msg := errorDetails(err)
	return ai.StreamError(Name, status, resp, code, msg, err)
}

// errorDetails reads what the endpoint reported. Unlike the SDK's error type,
// the response is kept, so a 429 that carries Retry-After is honoured rather
// than falling back to the caller's own backoff.
func errorDetails(err error) (status int, resp *http.Response, code, message string) {
	var se *statusError
	if errors.As(err, &se) {
		return se.status, se.response, se.code, se.message
	}
	return 0, nil, "", ""
}
