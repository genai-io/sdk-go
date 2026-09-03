// Package oai holds what the two OpenAI protocols share: the client they are
// both built on, the shape of a failure from the SDK they both use, and the
// framing an inline image takes. It is internal to openai/ so that the
// compiler, not a convention, keeps the other drivers out of it.
package oai

import (
	"errors"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
)

// NewClient builds the SDK client for a Config.
//
// It assembles the client itself rather than calling sdk.NewClient, which
// prepends DefaultClientOptions — and that reads OPENAI_API_KEY,
// OPENAI_BASE_URL, OPENAI_ORG_ID and OPENAI_PROJECT_ID out of the process
// environment. A Config is meant to be the whole truth about where a request
// goes and what it presents, so none of them is ever read.
func NewClient(cfg ai.Config) sdk.Client {
	opts := clientOptions(cfg)
	// Only the three services the two drivers use are wired; a fourth would find
	// a client carrying no options and fail loudly on its first call.
	client := sdk.Client{Options: opts}
	client.Chat = sdk.NewChatService(opts...)
	client.Models = sdk.NewModelService(opts...)
	client.Responses = responses.NewResponseService(opts...)
	return client
}

func clientOptions(cfg ai.Config) []option.RequestOption {
	opts := []option.RequestOption{
		// The SDK's own host, which its constructor would otherwise be the only
		// thing supplying.
		option.WithEnvironmentProduction(),
		// Retries belong to the caller, which alone knows the budget for the
		// whole turn. An SDK retrying underneath would multiply it silently.
		option.WithMaxRetries(0),
	}
	if url := cfg.URL(); url != "" {
		opts = append(opts, option.WithBaseURL(url))
	}
	// Keyless endpoints exist — a local Ollama ignores the header entirely — so
	// no key means no credential header rather than a placeholder.
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	for k, v := range cfg.MergedHeaders() {
		opts = append(opts, option.WithHeader(k, v))
	}
	return opts
}

// RequestOptions are the headers one call carries beyond the ones the client
// was built with. They are applied after those, so a name the caller gave wins
// for this request.
func RequestOptions(req *ai.Request) []option.RequestOption {
	if len(req.Headers) == 0 {
		return nil
	}
	opts := make([]option.RequestOption, 0, len(req.Headers))
	for k, v := range req.Headers {
		opts = append(opts, option.WithHeader(k, v))
	}
	return opts
}

// Details reports what the SDK knows about a failed call. Everything is zero
// when err did not come from the SDK, which leaves classification to fall back
// to the transport.
func Details(err error) errs.Details {
	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		return errs.Details{}
	}
	message := strings.TrimSpace(apiErr.Message)
	if message == "" {
		// A body the SDK could not parse into a message still says more than an
		// empty string does.
		message = strings.TrimSpace(apiErr.RawJSON())
	}
	return errs.Details{
		Status:   apiErr.StatusCode,
		Code:     apiErr.Code,
		Message:  message,
		Response: apiErr.Response,
	}
}

// DataURI frames an inline image the way both protocols take one.
func DataURI(mediaType, data string) string {
	return "data:" + mediaType + ";base64," + data
}
