package anthropic

import (
	"errors"
	"net/http"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Classifying this protocol's failures into ai.Error kinds.
//
// A driver is the only place the mapping is reliable, because it is the only
// place the SDK's own typed errors and the HTTP response are both in hand.
// Everything above the driver reads Kind and nothing else.

func (d *Driver) wrap(err error) error {
	if err == nil {
		return nil
	}
	status, code, msg, resp := errorDetails(err)
	return ai.Classify(Name, status, resp, code, msg, err)
}

func (d *Driver) wrapStream(err error) error {
	if err == nil {
		return nil
	}
	status, code, msg, resp := errorDetails(err)
	return ai.StreamError(Name, status, resp, code, msg, err)
}

// errorDetails pulls everything the classifier needs out of one assertion,
// the shape the shared OpenAI helper already uses.
func errorDetails(err error) (status int, code, message string, resp *http.Response) {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode, "", strings.TrimSpace(apiErr.Error()), apiErr.Response
	}
	return 0, "", "", nil
}
