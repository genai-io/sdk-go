// Package vertex serves Gemini models through Google Cloud Vertex AI.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/google/vertex"
//
// The body is the one driver/google sends; what changes is the address, which
// names a Google Cloud project and location, and the credential, which is
// Google Application Default Credentials rather than a key.
package vertex

import (
	"context"
	"fmt"
	"net/http"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/genai-io/sdk-go/pkg/ai"
	googledriver "github.com/genai-io/sdk-go/pkg/ai/driver/google"
)

// Name is the driver's identifier.
const Name = string(ai.APIGoogleVertex)

// DefaultRegion is where a model is served when the deployment names no
// region. Google recommends the global endpoint for availability; a specific
// region is for data residency.
const DefaultRegion = "global"

// cloudPlatformScope is the OAuth scope Vertex AI accepts a token under.
const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

func init() { ai.RegisterAPI(ai.APIGoogleVertex, New) }

// New builds a driver from a Config. The GCP project and region come from
// Config.ProtocolConfig as an ai.VertexConfig; package ai/auth fills one in from
// the environment.
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

	// Finding the credential reads files and the environment; the token is
	// minted on the first request. So a failure here is a missing credential,
	// which a caller enumerating providers should learn without a request.
	creds, err := google.FindDefaultCredentials(context.Background(), cloudPlatformScope)
	if err != nil {
		return nil, &ai.Error{
			Driver:  Name,
			Kind:    ai.KindAuth,
			Message: fmt.Sprintf("Google Application Default Credentials are unavailable: %v", err),
		}
	}
	// The token rides the transport, so a caller's client keeps its own
	// settings and gains the authorization on top.
	client := http.DefaultClient
	if cfg.HTTPClient != nil {
		client = cfg.HTTPClient
	}
	authed := *client
	authed.Transport = &oauth2.Transport{Base: client.Transport, Source: creds.TokenSource}
	cfg.APIKey = "" // the credential is the token, and a key header beside it would be refused

	base := cfg.URL()
	if base == "" {
		base = baseURL(region)
	}
	return googledriver.NewWithClient(&authed, cfg, ai.APIGoogleVertex,
		fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/google/models", base, deployment.Project, region))
}

// baseURL is the host a region is served from. The global endpoint has no
// region prefix; every other region is its own host.
func baseURL(region string) string {
	if region == DefaultRegion {
		return "https://aiplatform.googleapis.com"
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com", region)
}
