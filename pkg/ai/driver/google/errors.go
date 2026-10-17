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
	var envelope errorEnvelope
	_ = json.Unmarshal(raw, &envelope)
	message := strings.TrimSpace(envelope.Error.Message)
	if message == "" {
		message = strings.TrimSpace(string(raw))
	}
	return &statusError{errs.Details{Status: res.StatusCode, Code: envelope.Error.Status, Message: message, Response: res}}
}

// statusError carries what the driver knows about a failed response through to
// classification.
type statusError struct{ errs.Details }

func (e *statusError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return e.Message
}

// details reads what the endpoint reported. There is no vendor SDK here, so
// the response travels on the driver's own error type and a 429 that carries
// Retry-After is honoured rather than falling back to the caller's backoff.
func details(err error) errs.Details {
	var se *statusError
	if errors.As(err, &se) {
		return se.Details
	}
	return errs.Details{}
}

// errorEnvelope is the parts of the Google API error envelope the driver reads:
// {"error":{message,status}}. The numeric code repeats the HTTP status.
type errorEnvelope struct {
	Error struct {
		Message string `json:"message,omitempty"`
		Status  string `json:"status,omitempty"`
	} `json:"error"`
}
