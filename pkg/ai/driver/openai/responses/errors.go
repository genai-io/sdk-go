package responses

import (
	wire "github.com/openai/openai-go/v3/responses"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/oai"
)

// fail classifies a transport or HTTP failure. The in-band kind below is this
// protocol's own, and arrives inside a 200.
var fail = errs.For(Name, oai.Details)

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

func wrap(err error) error       { return fail.Wrap(err) }
func wrapStream(err error) error { return fail.WrapStream(err) }
