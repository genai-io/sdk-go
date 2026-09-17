package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// otherConfig stands in for another protocol's construction settings, which
// this driver must refuse rather than ignore.
type otherConfig struct{}

func (otherConfig) ProtocolConfig() {}

// Every failure here happens before a request, so it is what a caller
// enumerating providers actually hits.
func TestNewRefusesWhatItCannotDeploy(t *testing.T) {
	model := ai.Model{ID: "gemini-test", API: ai.APIGoogleVertex}

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

	t.Run("no credentials", func(t *testing.T) {
		// Point every ADC source at nothing.
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "missing.json"))
		_, err := New(ai.Config{Model: model, ProtocolConfig: ai.VertexConfig{Project: "p"}})
		if !ai.IsAuth(err) {
			t.Fatalf("error = %v, want a credential failure", err)
		}
	})
}

// The project and location go in the path, the token in the header, and the
// countTokens body is Vertex's flat shape rather than the Gemini API's wrapper.
func TestRequestsCarryTheDeploymentAndTheToken(t *testing.T) {
	var got struct {
		path, auth string
		body       map[string]any
	}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.auth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&got.body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"totalTokens":7}`)
	}))
	defer s.Close()
	stubCredentials(t)

	d, err := New(ai.Config{
		Model:          ai.Model{ID: "gemini-test", API: ai.APIGoogleVertex},
		BaseURL:        s.URL + "/",
		ProtocolConfig: ai.VertexConfig{Project: "my-project", Region: "us-central1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	n, err := d.(ai.TokenCounter).CountTokens(context.Background(), &ai.Request{
		System:   "be brief",
		Messages: []ai.Message{{Role: ai.RoleUser, Content: ai.TextContent("hi")}},
	})
	if err != nil || n != 7 {
		t.Fatalf("CountTokens = %d, %v", n, err)
	}
	if want := "/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-test:countTokens"; got.path != want {
		t.Errorf("path = %q, want %q", got.path, want)
	}
	if got.auth != "Bearer stub-token" {
		t.Errorf("Authorization = %q, want the ADC token", got.auth)
	}
	if _, wrapped := got.body["generateContentRequest"]; wrapped {
		t.Errorf("body = %v, want the flat shape Vertex takes", got.body)
	}
	if _, ok := got.body["systemInstruction"]; !ok {
		t.Errorf("body = %v, want the system instruction counted", got.body)
	}
}

// The driver names the protocol as the caller reached it, on itself and on
// every error it classifies. Reporting the Gemini API would send someone
// debugging a Vertex deployment to the wrong console. And it claims no model
// listing, which is what lets a caller fall back to its catalog rather than
// show an empty picker.
func TestTheDeploymentReportsItsOwnName(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"no"}}`)
	}))
	defer s.Close()
	stubCredentials(t)

	d, err := New(ai.Config{
		Model:          ai.Model{ID: "gemini-test", API: ai.APIGoogleVertex},
		BaseURL:        s.URL,
		ProtocolConfig: ai.VertexConfig{Project: "p"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if d.Name() != Name {
		t.Errorf("Name = %q, want %q", d.Name(), Name)
	}
	if _, lists := d.(ai.ModelLister); lists {
		t.Error("the Vertex driver claims a model listing, which Vertex does not publish")
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

// stubCredentials installs a user credential whose token endpoint is a server
// that always hands out "stub-token", so the driver's ADC lookup and token
// exchange both succeed without Google.
func stubCredentials(t *testing.T) {
	t.Helper()
	tokens := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"access_token":"stub-token","token_type":"Bearer","expires_in":3600}`)
	}))
	t.Cleanup(tokens.Close)
	creds := fmt.Sprintf(`{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r","token_uri":%q}`, tokens.URL)
	path := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(path, []byte(creds), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", path)
}
