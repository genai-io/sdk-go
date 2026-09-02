package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
)

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

// fail classifies this protocol's failures.
var fail = errs.For(Name, details)

// details reads what the endpoint reported. There is no vendor SDK here, so
// the response travels on the driver's own error type and a 429 that carries
// Retry-After is honoured rather than falling back to the caller's backoff.
func details(err error) errs.Details {
	var se *statusError
	if errors.As(err, &se) {
		return errs.Details{Status: se.status, Code: se.code, Message: se.message, Response: se.response}
	}
	return errs.Details{}
}
