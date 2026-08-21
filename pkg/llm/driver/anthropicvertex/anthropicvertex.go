// Package anthropicvertex serves Claude models through Google Cloud Vertex AI.
//
// The wire protocol is Anthropic Messages, unchanged — everything downstream of
// building the HTTP client is handled by package anthropic, which this one
// wraps. What differs is authentication (Google Application Default
// Credentials rather than an API key), the endpoint (a regional Vertex host),
// and the model ID form (`claude-opus-4-5@20251101`).
//
// It is a separate package so that its Google Cloud auth dependency lands only
// in a build that asks for it. Import it for its side effect:
//
//	import _ "github.com/genai-io/sdk-go/pkg/llm/driver/anthropicvertex"
//
// Credentials come from the ambient environment the way every Google Cloud
// client resolves them — `gcloud auth application-default login`, a service
// account key at GOOGLE_APPLICATION_CREDENTIALS, or the metadata server on a
// GCP instance. There is no API key to pass; llm.Config.APIKey is ignored.
package anthropicvertex

import (
	"context"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/driver/anthropic"
)

// Name is the driver's identifier.
const Name = string(llm.APIAnthropicVertex)

// DefaultRegion is where a model is served when the deployment names no
// region. Google recommends the global endpoint for availability; a specific
// region is for data residency, and carries a price premium.
const DefaultRegion = "global"

func init() { llm.RegisterAPI(llm.APIAnthropicVertex, New) }

// New builds a driver from a Config. The GCP project and region come from
// Config.Native as an llm.VertexConfig; package llm/auth fills one in from the
// environment.
func New(cfg llm.Config) (llm.Driver, error) {
	deployment := llm.ConfigNativeOf[llm.VertexConfig](cfg)
	if deployment.Project == "" {
		return nil, &llm.Error{
			Driver:  Name,
			Kind:    llm.KindAuth,
			Message: "no Google Cloud project: set Config.Native to an llm.VertexConfig, or use auth.Provider to read it from the environment",
		}
	}
	region := deployment.Region
	if region == "" {
		region = DefaultRegion
	}

	// WithGoogleAuth resolves Application Default Credentials and installs the
	// middleware that rewrites a Messages request into Vertex's shape. It
	// reaches the network to mint a token, so a failure here is a credential
	// problem, not a request one.
	auth, err := googleAuth(context.Background(), region, deployment.Project)
	if err != nil {
		return nil, err
	}

	// The shared options carry the endpoint override, transport and headers;
	// the credential ones are skipped because Vertex has no API key.
	opts := append(anthropic.ClientOptions(cfg), auth)
	return anthropic.NewWithClient(sdk.NewClient(opts...), cfg)
}

// googleAuth is split out so the failure has a classified error rather than a
// bare one from the SDK.
func googleAuth(ctx context.Context, region, project string) (opt option.RequestOption, err error) {
	defer func() {
		// WithGoogleAuth panics rather than returning when credentials cannot
		// be found, which would take down a caller that was only enumerating
		// providers.
		if r := recover(); r != nil {
			err = &llm.Error{
				Driver:  Name,
				Kind:    llm.KindAuth,
				Message: fmt.Sprintf("Google Application Default Credentials are unavailable: %v", r),
			}
		}
	}()
	return vertex.WithGoogleAuth(ctx, region, project), nil
}
