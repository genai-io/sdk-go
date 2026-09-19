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
	"strings"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/genai-io/sdk-go/pkg/ai"
	googledriver "github.com/genai-io/sdk-go/pkg/ai/driver/google"
)

// Name is the driver's identifier.
const Name = string(ai.APIGoogleVertex)

// cloudPlatformScope is the OAuth scope Vertex AI accepts a token under.
const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

func init() { ai.RegisterAPI(ai.APIGoogleVertex, New) }

// driver is the Gemini driver minus ai.ModelLister: Vertex publishes no
// listing of Gemini models, and not claiming one is how a caller learns to
// show its catalog instead. Embedding the two interfaces rather than the
// driver is what keeps Models from being promoted along.
type driver struct {
	ai.Driver
	ai.TokenCounter
}

// New builds a driver from a Config. The GCP project and region come from
// Config.ProtocolConfig as an ai.VertexConfig; package ai/auth fills one in from
// the environment.
func New(cfg ai.Config) (ai.Driver, error) {
	deployment, err := ai.VertexDeployment(cfg, Name)
	if err != nil {
		return nil, err
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
	cfg.HTTPClient = &authed
	cfg.APIKey = "" // the credential is the token, and a key header beside it would be refused

	base := strings.TrimSuffix(cfg.URL(), "/")
	if base == "" {
		base = baseURL(deployment.Region)
	}
	d, err := googledriver.NewAt(cfg, ai.APIGoogleVertex,
		fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/google/models", base, deployment.Project, deployment.Region))
	if err != nil {
		return nil, err
	}
	return driver{d, d}, nil
}

// baseURL is the host a region is served from. The global endpoint has no
// region prefix; every other region is its own host.
func baseURL(region string) string {
	if region == ai.VertexDefaultRegion {
		return "https://aiplatform.googleapis.com"
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com", region)
}
