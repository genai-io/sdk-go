// Package vertex serves Claude models through Google Cloud Vertex AI.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic/vertex"
package vertex

import (
	"context"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
)

// Name is the driver's identifier.
const Name = string(ai.APIAnthropicVertex)

// DefaultRegion is where a model is served when the deployment names no
// region. Google recommends the global endpoint for availability; a specific
// region is for data residency, and carries a price premium.
const DefaultRegion = "global"

func init() { ai.RegisterAPI(ai.APIAnthropicVertex, New) }

// New builds a driver from a Config. The GCP project and region come from
// Config.ProtocolConfig as an ai.VertexConfig; package ai/auth fills one in from the
// environment.
func New(cfg ai.Config) (ai.Driver, error) {
	deployment, err := ai.ProtocolConfigAs[ai.VertexConfig](cfg)
	if err != nil {
		return nil, err
	}
	if deployment.Project == "" {
		return nil, &ai.Error{
			Driver:  Name,
			Kind:    ai.KindAuth,
			Message: "no Google Cloud project: set Config.ProtocolConfig to an ai.VertexConfig, or use auth.Config to read it from the environment",
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
	//
	// The context is the process's own: this is construction, not a call, and
	// the seam a driver factory is built through carries no context. A caller
	// who needs one can mint the credential itself and hand the client over
	// through anthropic.NewWithClient.
	auth, err := googleAuth(context.Background(), region, deployment.Project)
	if err != nil {
		return nil, err
	}

	// The auth option carries a base URL and an http.Client of its own, so it
	// goes first and the Config's endpoint and headers land over it. Its
	// http.Client is the one part that cannot be layered: the Google token is
	// injected by that client's transport, so a Config.HTTPClient would replace
	// the credential rather than wrap it, and is dropped here instead of
	// silently removing the authentication.
	cfg.HTTPClient = nil
	opts := anthropic.ClientOptions(cfg, auth)
	return anthropic.NewWithClient(anthropic.NewSDKClient(opts...), cfg, ai.APIAnthropicVertex)
}

// googleAuth is split out so the failure has a classified error rather than a
// bare one from the SDK.
func googleAuth(ctx context.Context, region, project string) (opt option.RequestOption, err error) {
	defer func() {
		// WithGoogleAuth panics rather than returning when credentials cannot
		// be found, which would take down a caller that was only enumerating
		// providers.
		if r := recover(); r != nil {
			err = &ai.Error{
				Driver:  Name,
				Kind:    ai.KindAuth,
				Message: fmt.Sprintf("Google Application Default Credentials are unavailable: %v", r),
			}
		}
	}()
	return vertex.WithGoogleAuth(ctx, region, project), nil
}
