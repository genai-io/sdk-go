package integration_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func TestProviderJSONNumbers(t *testing.T) {
	for _, api := range []ai.API{ai.APIOpenAIResponses, ai.APIOpenAIChat, ai.APIAnthropicMessages} {
		t.Run(string(api), func(t *testing.T) {
			bodies := make(chan string, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				bodies <- string(raw)
				w.Header().Set("Content-Type", "text/event-stream")
			}))
			defer server.Close()
			model := ai.Model{ID: "test", API: api, MaxOutput: 1024}
			if api != ai.APIAnthropicMessages {
				model.SamplingParams = map[string]any{"numeric_extra": map[string]any{"values": []any{json.Number("9007199254740993"), json.Number("1.234567890123456789"), "001"}}}
			}
			client := open(t, server.URL, model)
			def := map[string]any{"type": "object", "properties": map[string]any{"items": map[string]any{
				"type": "array", "maxItems": 8, "const": json.Number("9007199254740993"),
				"enum": []any{json.Number("1.234567890123456789"), "8", "001", "1.25"},
			}}}
			schema := ai.Schema{Name: "numbers", Definition: def}
			for range client.StreamWithAdmission(context.Background(), []ai.Message{ai.UserMessage("hello")}, func(context.Context, *ai.Request) error { return nil }, ai.WithTools(schema), ai.WithSchema(&schema)) {
			}
			var body string
			select {
			case body = <-bodies:
			default:
				t.Fatal("request did not reach HTTP server")
			}
			if api != ai.APIAnthropicMessages && !strings.Contains(body, `"numeric_extra":{"values":[9007199254740993,1.234567890123456789,"001"]}`) {
				t.Errorf("sampling extras lost numeric types: %s", body)
			}
			// Each schema is serialized twice: tool input and structured output.
			for _, fragment := range []string{`"maxItems":8`, `"const":9007199254740993`, `"enum":[1.234567890123456789,"8","001","1.25"]`} {
				if strings.Count(body, fragment) != 2 {
					t.Errorf("want two %s in request: %s", fragment, body)
				}
			}
			if def["properties"].(map[string]any)["items"].(map[string]any)["maxItems"] != 8 {
				t.Fatal("source mutated")
			}
		})
	}
}
