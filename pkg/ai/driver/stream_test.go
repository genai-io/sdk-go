package driver_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

func TestPrematureStreamEndIsAnErrorThroughThePublicClient(t *testing.T) {
	for _, tc := range []struct {
		api     ai.API
		payload string
	}{
		{ai.APIOpenAIChat, `data: {"choices":[{"delta":{"content":"half"}}]}`},
		{ai.APIOpenAIResponses, `data: {"type":"response.output_text.delta","delta":"half"}`},
		{ai.APIGoogleGenAI, `data: {"candidates":[{"content":{"parts":[{"text":"half"}]}}]}`},
		{ai.APIAnthropicMessages, "event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"half"}}`},
	} {
		t.Run(string(tc.api), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, tc.payload+"\n\n")
			}))
			defer server.Close()
			client, err := ai.New(ai.Config{BaseURL: server.URL, Model: ai.Model{ID: "test", API: tc.api}})
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Complete(t.Context(), []ai.Message{ai.UserMessage("hi")})
			if !ai.IsKind(err, ai.KindNetwork) || !ai.IsRetryable(err) {
				t.Fatalf("premature EOF = %v, want a retryable network error", err)
			}
			if response == nil || response.Text() != "half" || response.StopReason != ai.StopError {
				t.Fatalf("partial response = %+v, want its text retained with error status", response)
			}
		})
	}
}
