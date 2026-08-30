package responses

import (
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/errs"
	wire "github.com/openai/openai-go/v3/responses"
)

// responseError converts an in-band API failure. These arrive inside a 200
// response, so there is no status to classify from — the error code is the
// only signal for whether another attempt could work.
func (d *Driver) responseError(code, message string) error {
	if message == "" {
		message = "responses API failed"
	}
	err := &ai.Error{Driver: Name, Code: code, Message: message}
	if kind, ok := ai.ClassifyMessage(message); ok {
		err.Kind = kind
		return err
	}
	switch code {
	case string(wire.ResponseErrorCodeServerError),
		string(wire.ResponseErrorCodeRateLimitExceeded),
		string(wire.ResponseErrorCodeVectorStoreTimeout):
		err.Kind = ai.KindOverloaded
	default:
		err.Kind = ai.KindInvalidRequest
	}
	return err
}

func wrap(err error) error       { return errs.Wrap(Name, err) }
func wrapStream(err error) error { return errs.WrapStream(Name, err) }
