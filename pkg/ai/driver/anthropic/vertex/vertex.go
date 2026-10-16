// Package vertex serves Claude models through Google Cloud Vertex AI.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic/vertex"
package vertex

import (
	"cmp"
	"context"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
)

// name is the driver's identifier.
const name = string(ai.APIAnthropicVertex)

func init() { ai.RegisterAPI(ai.APIAnthropicVertex, New) }

// New builds a driver from a Config. The GCP project and region come from
// Config.Vertex; package ai/auth fills it in from the environment.
func New(cfg ai.Config) (ai.Driver, error) {
	if cfg.Vertex == nil || cfg.Vertex.Project == "" {
		return nil, &ai.Error{
			Driver:  name,
			Kind:    ai.KindAuth,
			Message: "no Google Cloud project: set Config.Vertex, or use auth.Config to read it from the environment",
		}
	}

	// WithGoogleAuth resolves Application Default Credentials and installs the
	// middleware that rewrites a Messages request into Vertex's shape. It
	// reaches the network to mint a token, so a failure here is a credential
	// problem, not a request one.
	//
	// The context is the process's own: the driver-factory seam carries none. A
	// caller who needs one mints the credential and uses anthropic.NewDeployment.
	auth, err := googleAuth(context.Background(), cmp.Or(cfg.Vertex.Region, ai.VertexDefaultRegion), cfg.Vertex.Project)
	if err != nil {
		return nil, err
	}

	// The auth option goes first so the Config's endpoint and headers land over
	// it. Its http.Client is the one part that cannot be layered — the Google
	// token rides that transport — so a Config.HTTPClient is dropped instead.
	cfg.HTTPClient = nil
	return anthropic.NewDeployment(cfg, ai.APIAnthropicVertex, auth)
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
				Driver:  name,
				Kind:    ai.KindAuth,
				Message: fmt.Sprintf("Google Application Default Credentials are unavailable: %v", r),
			}
		}
	}()
	return vertex.WithGoogleAuth(ctx, region, project), nil
}
