package integration_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestInferenceAdmissionBlocksNetwork(t *testing.T) {
	for _, api := range []ai.API{ai.APIAnthropicMessages, ai.APIOpenAIChat, ai.APIOpenAIResponses, ai.APIGoogleGenAI} {
		t.Run(string(api), func(t *testing.T) {
			var reached atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				reached.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer server.Close()
			client := open(t, server.URL, ai.Model{ID: "test", API: api, MaxOutput: 1024})
			refused := errors.New("cannot durably record attempt")
			calls := 0
			var got error
			for _, err := range client.StreamWithAdmission(context.Background(), []ai.Message{ai.UserMessage("hello")}, func(_ context.Context, req *ai.Request) error {
				calls++
				if req.MaxTokens != 128 || req.System != "policy" {
					t.Fatalf("not finalized: %+v", req)
				}
				return refused
			}, ai.WithMaxTokens(128), ai.WithSystem("policy")) {
				got = err
			}
			var admission *ai.AdmissionError
			if !errors.Is(got, refused) || !errors.As(got, &admission) || calls != 1 || reached.Load() != 0 {
				t.Fatalf("err=%v admissions=%d network calls=%d", got, calls, reached.Load())
			}
		})
	}
}
