package anthropicvertex_test

import (
	"errors"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/anthropicvertex"
)

func vertexModel(t *testing.T) llm.Model {
	t.Helper()
	m, err := catalog.Model("anthropic-vertex/claude-opus-5")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	return m
}

// Vertex needs a project before it can build a client at all, and saying so
// beats meeting an opaque credential error on the first request.
func TestMissingProjectIsAnAuthError(t *testing.T) {
	_, err := llm.Open(llm.Config{Model: vertexModel(t)})
	if err == nil {
		t.Fatal("expected an error without a project")
	}
	if !llm.IsAuth(err) {
		t.Errorf("err = %v, want an auth failure", err)
	}
	var typed *llm.Error
	if !errors.As(err, &typed) || typed.Driver == "" {
		t.Errorf("err = %v, want a driver-attributed llm.Error", err)
	}
}

// Credentials come from the ambient Google environment, which may or may not
// exist here. The invariant that matters either way is that a missing one
// surfaces as a classified error rather than taking the process down — the SDK
// panics on absent Application Default Credentials, and a caller merely
// enumerating providers must not die of it.
func TestAbsentCredentialsDoNotPanic(t *testing.T) {
	cfg := llm.Config{
		Model:  vertexModel(t),
		Native: llm.VertexConfig{Project: "test-project", Region: "global"},
	}

	_, err := llm.Open(cfg)
	if err == nil {
		return // this machine has Application Default Credentials; fine.
	}
	var typed *llm.Error
	if !errors.As(err, &typed) {
		t.Fatalf("err = %v, want a classified *llm.Error", err)
	}
	if typed.Kind != llm.KindAuth {
		t.Errorf("Kind = %q, want %q", typed.Kind, llm.KindAuth)
	}
}

// The protocol is registered by the blank import, so a Vertex model opens
// through the ordinary llm.Open path.
func TestProtocolIsRegistered(t *testing.T) {
	var found bool
	for _, api := range llm.RegisteredAPIs() {
		if api == llm.APIAnthropicVertex {
			found = true
		}
	}
	if !found {
		t.Errorf("APIs = %v, want %q registered", llm.RegisteredAPIs(), llm.APIAnthropicVertex)
	}
}

// Pre-4.6 generations keep the @-versioned snapshot form, and the driver must
// not mangle it — the version separator is part of the ID Vertex expects.
func TestSnapshotModelIDSurvives(t *testing.T) {
	m, err := catalog.Model("anthropic-vertex/claude-opus-4-5@20251101")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.ID != "claude-opus-4-5@20251101" {
		t.Errorf("ID = %q", m.ID)
	}
	if m.API != llm.APIAnthropicVertex {
		t.Errorf("API = %q", m.API)
	}
}
