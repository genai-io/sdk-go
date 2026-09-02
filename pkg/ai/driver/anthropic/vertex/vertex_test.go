package vertex

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
)

// otherConfig stands in for another protocol's construction settings, which
// this driver must refuse rather than ignore.
type otherConfig struct{}

func (otherConfig) ProtocolConfig() {}

// Both failures happen before any credential is resolved, so they are what a
// caller enumerating providers actually hits.
func TestNewRefusesWhatItCannotDeploy(t *testing.T) {
	model := ai.Model{ID: "claude-test", API: ai.APIAnthropicVertex}

	t.Run("no project", func(t *testing.T) {
		_, err := New(ai.Config{Model: model})
		if !ai.IsAuth(err) {
			t.Fatalf("error = %v, want a credential failure", err)
		}
		if !strings.Contains(err.Error(), "auth.Config") {
			t.Errorf("message = %q, want it to name the function that fills the project in", err)
		}
	})

	t.Run("another protocol's settings", func(t *testing.T) {
		_, err := New(ai.Config{Model: model, ProtocolConfig: otherConfig{}})
		if err == nil {
			t.Fatal("a foreign ProtocolConfig was accepted")
		}
		if !strings.Contains(err.Error(), "VertexConfig") {
			t.Errorf("message = %q, want it to name the type this driver expects", err)
		}
	})
}

// The driver names the protocol as the caller reached it. Reporting the direct
// Anthropic API would send someone debugging a Vertex deployment to the wrong
// console.
func TestTheDeploymentReportsItsOwnName(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"no"}}`)
	}))
	defer s.Close()

	cfg := ai.Config{BaseURL: s.URL, Model: ai.Model{ID: "claude-test", API: ai.APIAnthropicVertex}}
	d, err := anthropic.NewWithClient(
		anthropic.NewSDKClient(anthropic.ClientOptions(cfg)...), cfg, ai.APIAnthropicVertex)
	if err != nil {
		t.Fatalf("NewWithClient: %v", err)
	}
	if d.Name() != Name {
		t.Errorf("Name = %q, want %q", d.Name(), Name)
	}

	var streamErr error
	for _, err := range d.Stream(context.Background(), &ai.Request{Messages: []ai.Message{ai.UserMessage("hi")}}) {
		if err != nil {
			streamErr = err
		}
	}
	var e *ai.Error
	if !errors.As(streamErr, &e) {
		t.Fatalf("error is %T (%v), want *ai.Error", streamErr, streamErr)
	}
	if e.Driver != Name {
		t.Errorf("ai.Error.Driver = %q, want %q", e.Driver, Name)
	}
}

// The auth option carries a base URL of its own, and used to be applied last,
// which silently threw away the caller's endpoint.
func TestTheCallersEndpointOutranksTheDeploymentsOwn(t *testing.T) {
	reached := make(chan string, 1)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case reached <- r.URL.Path:
		default:
		}
		w.Header().Set("Content-Type", "text/event-stream")
	}))
	defer s.Close()

	// Stands in for vertex.WithGoogleAuth, which likewise applies a base URL of
	// the deployment's own.
	deployment := option.WithBaseURL("https://global-aiplatform.googleapis.com/")

	cfg := ai.Config{BaseURL: s.URL, Model: ai.Model{ID: "claude-test", API: ai.APIAnthropicVertex}}
	opts := anthropic.ClientOptions(cfg, deployment)
	d, err := anthropic.NewWithClient(anthropic.NewSDKClient(opts...), cfg, ai.APIAnthropicVertex)
	if err != nil {
		t.Fatalf("NewWithClient: %v", err)
	}
	for range d.Stream(context.Background(), &ai.Request{Messages: []ai.Message{ai.UserMessage("hi")}}) {
	}

	select {
	case path := <-reached:
		if path != "/v1/messages" {
			t.Errorf("path = %q, want the Messages endpoint", path)
		}
	default:
		t.Fatal("the request went to the deployment's host, not the Config's")
	}
}
