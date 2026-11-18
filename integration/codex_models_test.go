package integration_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"github.com/genai-io/sdk-go/pkg/ai/endpoint"
)

func TestCodexCatalogVersionOverride(t *testing.T) {
	const pinned = "0.123.4"
	for _, status := range []int{http.StatusOK, http.StatusUnauthorized} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Method != http.MethodGet || r.URL.Path != "/models" || r.URL.Query().Get("client_version") != pinned {
					t.Errorf("catalog request = %s %s", r.Method, r.URL)
				}
				if r.Header.Get("Authorization") != "Bearer local-test-key" || r.Header.Get("ChatGPT-Account-ID") != "local-account" {
					t.Error("catalog missing auth/provider headers")
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				if status == http.StatusUnauthorized {
					_, _ = io.WriteString(w, `{"error":{"message":"expired credential","type":"authentication_error"}}`)
					return
				}
				_, _ = io.WriteString(w, `{"models":[{"slug":"future-unknown-model","description":"Provider catalog description","show_in_picker":true,"context_window":234567,"default_reasoning_level":"high","supported_reasoning_levels":[{"effort":"high"}]}]}`)
			}))
			defer srv.Close()
			vendor := catalog.Vendor{
				ID: "local-codex", API: ai.APIOpenAIResponses,
				Compat:  ai.OpenAIResponsesCompat{CodexModelListing: true, CodexClientVersion: pinned},
				Headers: map[string]string{"ChatGPT-Account-ID": "local-account"},
			}
			ep := vendor.Endpoint(endpoint.Config{BaseURL: srv.URL, APIKey: "local-test-key", HTTPClient: srv.Client()})
			err := ep.Refresh(context.Background())
			if got := calls.Load(); got != 1 {
				t.Fatalf("catalog requests = %d, want 1 (no retry)", got)
			}
			if status == http.StatusUnauthorized {
				var classified *ai.Error
				if !errors.As(err, &classified) || classified.Kind != ai.KindAuth {
					t.Fatalf("Refresh error = %v, want auth error", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			models := ep.Models()
			if len(models) != 1 {
				t.Fatalf("models = %+v, want one unknown visible model", models)
			}
			m := models[0]
			if m.Description != "Provider catalog description" {
				t.Fatalf("catalog description lost through endpoint: %+v", m)
			}
			if m.ID != "future-unknown-model" || m.ContextWindow != 234567 || len(m.Reasoning) != 1 || m.Reasoning[0].Effort != ai.Effort("high") || !m.Reasoning[0].Default {
				t.Fatalf("catalog model lost metadata: %+v", m)
			}
		})
	}
}
