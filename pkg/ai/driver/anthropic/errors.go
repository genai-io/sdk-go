package anthropic

import (
	"encoding/json"
	"errors"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
)

// errorEnvelope is how this protocol reports a failure:
// {"type":"error","error":{"type":"invalid_request_error","message":"…"}}.
type errorEnvelope struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// details reads a failure out of the SDK's error type. The body is parsed
// rather than taken from the SDK's Error method, which renders the request
// line, the status, the request ID and the raw JSON all at once: that makes a
// message no caller wants to show, and it leaves the error type — the one
// machine-readable part, and what prompt_too_long arrives as — unread.
func details(err error) errs.Details {
	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		return errs.Details{}
	}
	raw := apiErr.RawJSON()
	var envelope errorEnvelope
	_ = json.Unmarshal([]byte(raw), &envelope)

	message := strings.TrimSpace(envelope.Error.Message)
	if message == "" {
		// A body that would not parse still says more than an empty string.
		message = strings.TrimSpace(raw)
	}
	return errs.Details{
		Status:   apiErr.StatusCode,
		Code:     envelope.Error.Type,
		Message:  message,
		Response: apiErr.Response,
	}
}
