package all_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/auth"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"

	_ "github.com/genai-io/sdk-go/pkg/llm/driver/all"
)

// Every protocol the catalog names must have a driver once this package is
// imported; a vendor whose protocol is unregistered is unreachable and the
// failure would only surface at runtime.
func TestEveryCatalogProtocolIsReachable(t *testing.T) {
	registered := make(map[llm.API]bool)
	for _, api := range llm.RegisteredAPIs() {
		registered[api] = true
	}
	for _, v := range catalog.All() {
		if !registered[v.API] {
			t.Errorf("vendor %q speaks %q, which has no registered driver", v.ID, v.API)
		}
	}
}

// The end-to-end path a caller actually uses: an environment variable, a model
// reference, a request. Ollama stands in because it is the one vendor whose
// endpoint is meant to be local.
func TestCatalogToCompletion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: "+`{"id":"1","choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":3,"completion_tokens":1}}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	// The bare host is what people have to hand; the vendor's /v1 suffix is
	// added for them.
	t.Setenv("OLLAMA_BASE_URL", server.URL)

	client, err := auth.Open("ollama/llama4")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if client.ContextWindow() != 131_072 {
		t.Errorf("ContextWindow = %d, want the catalog's figure", client.ContextWindow())
	}

	resp, err := client.Complete(context.Background(), &llm.Prompt{
		Messages: []llm.Message{llm.User("ping")},
	}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "pong" {
		t.Errorf("Content = %q", resp.Content)
	}
}

// The whole path a model picker actually walks: a provider renders its catalog
// baseline immediately, refreshes against the live endpoint in the background,
// and opens a model from the merged result.
func TestProviderRefreshThroughDriver(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[
				{"id":"llama4","object":"model","context_length":262144},
				{"id":"newly-pulled","object":"model"}
			]}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: "+`{"id":"1","choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":"stop"}]}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		default:
			t.Errorf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	v, ok := catalog.Find("ollama")
	if !ok {
		t.Fatal("ollama vendor missing")
	}
	p := v.Provider(llm.ProviderConfig{BaseURL: server.URL + "/v1"})

	// Before any network call the baseline is already readable.
	before := p.Models()
	if len(before) == 0 {
		t.Fatal("no baseline models")
	}
	llama, ok := p.Model("llama4")
	if !ok || llama.ContextWindow != 131_072 {
		t.Fatalf("baseline llama4 = %+v", llama)
	}

	if err := p.Refresh(context.Background()); err != nil {
		t.Fatalf("Refresh: %v", err)
	}

	// The endpoint's own figure wins where it stated one.
	llama, _ = p.Model("llama4")
	if llama.ContextWindow != 262_144 {
		t.Errorf("ContextWindow = %d, want the endpoint's 262144", llama.ContextWindow)
	}
	// A model only the endpoint knows is added, carrying the vendor's protocol.
	pulled, known := p.Model("newly-pulled")
	if !known {
		t.Fatal("a newly pulled model was not picked up")
	}
	if pulled.API != llm.APIOpenAIChat {
		t.Errorf("newly-pulled API = %q", pulled.API)
	}
	// A model the listing omitted survives rather than disappearing.
	if _, ok := p.Model("qwq"); !ok {
		t.Error("a baseline model missing from the listing was dropped")
	}

	client, err := p.Open("newly-pulled")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	resp, err := client.Complete(context.Background(), &llm.Prompt{Messages: []llm.Message{llm.User("ping")}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "pong" {
		t.Errorf("Content = %q", resp.Content)
	}
}
