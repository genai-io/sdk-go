package mcp_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/mcp"
	"github.com/genai-io/sdk-go/pkg/ai"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// The test binary doubles as an MCP server when asked, so what is under test is
// a real round trip: a process started, spoken to over its stdin and stdout,
// and stopped. A fake transport would test the translation and nothing else,
// and the translation is the easy half.
const (
	serverEnv = "SDK_GO_MCP_TEST_SERVER"
	// notifyEnv makes that server add a tool once a client is connected.
	notifyEnv = "SDK_GO_MCP_TEST_NOTIFY"
)

func TestMain(m *testing.M) {
	if os.Getenv(serverEnv) == "1" {
		serve()
		return
	}
	os.Exit(m.Run())
}

type greetArgs struct {
	Name string `json:"name" jsonschema:"who to greet"`
}

func serve() {
	s := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test-server", Version: "v1"}, nil)

	mcpsdk.AddTool(s, &mcpsdk.Tool{Name: "greet", Description: "Say hello to someone."},
		func(_ context.Context, _ *mcpsdk.CallToolRequest, a greetArgs) (*mcpsdk.CallToolResult, any, error) {
			return &mcpsdk.CallToolResult{
				Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "hello " + a.Name}},
			}, nil, nil
		})

	mcpsdk.AddTool(s, &mcpsdk.Tool{Name: "broken", Description: "Always fails."},
		func(_ context.Context, _ *mcpsdk.CallToolRequest, _ struct{}) (*mcpsdk.CallToolResult, any, error) {
			return &mcpsdk.CallToolResult{
				IsError: true,
				Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "the path does not exist"}},
			}, nil, nil
		})

	mcpsdk.AddTool(s, &mcpsdk.Tool{Name: "picture", Description: "Returns an image."},
		func(_ context.Context, _ *mcpsdk.CallToolRequest, _ struct{}) (*mcpsdk.CallToolResult, any, error) {
			return &mcpsdk.CallToolResult{
				Content: []mcpsdk.Content{&mcpsdk.ImageContent{MIMEType: "image/png", Data: []byte{0x89, 'P', 'N', 'G'}}},
			}, nil, nil
		})

	s.AddResource(&mcpsdk.Resource{URI: "file:///tmp/notes.md", Name: "notes", MIMEType: "text/markdown"},
		func(context.Context, *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
			return &mcpsdk.ReadResourceResult{}, nil
		})
	s.AddPrompt(&mcpsdk.Prompt{Name: "review", Description: "Review a diff."},
		func(context.Context, *mcpsdk.GetPromptRequest) (*mcpsdk.GetPromptResult, error) {
			return &mcpsdk.GetPromptResult{}, nil
		})

	if os.Getenv(notifyEnv) == "1" {
		go func() {
			time.Sleep(300 * time.Millisecond)
			s.AddTool(&mcpsdk.Tool{Name: "late", Description: "added after connect.",
				InputSchema: json.RawMessage(`{"type":"object"}`)},
				func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
					return &mcpsdk.CallToolResult{}, nil
				})
		}()
	}

	if err := s.Run(context.Background(), &mcpsdk.StdioTransport{}); err != nil {
		os.Exit(1)
	}
}

func connect(t *testing.T, name string) *mcp.Client {
	t.Helper()
	c, err := mcp.Connect(t.Context(), mcp.Server{
		Name:    name,
		Command: os.Args[0],
		Env:     map[string]string{serverEnv: "1"},
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func toolNamed(t *testing.T, tools []agent.Tool, name string) agent.Tool {
	t.Helper()
	for _, tool := range tools {
		if tool.Schema().Name == name {
			return tool
		}
	}
	t.Fatalf("no tool named %q in %v", name, names(tools))
	return nil
}

func names(tools []agent.Tool) []string {
	out := make([]string, 0, len(tools))
	for _, tool := range tools {
		out = append(out, tool.Schema().Name)
	}
	return out
}

// What a server advertises has to arrive as something agent.New takes, schema
// and all — that is the whole reason this package exists.
func TestAServersToolsArriveAsTheAgentsOwn(t *testing.T) {
	tools, err := connect(t, "").Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(tools) != 3 {
		t.Fatalf("tools = %v, want the three the server advertises", names(tools))
	}

	greet := toolNamed(t, tools, "greet").Schema()
	if greet.Description != "Say hello to someone." {
		t.Errorf("description = %q", greet.Description)
	}
	if greet.Definition == nil {
		t.Fatal("the tool arrived with no JSON Schema; the model would be told nothing about its arguments")
	}
	// The schema is the server's, passed through rather than re-derived.
	if !strings.Contains(schemaJSON(t, greet), `"name"`) {
		t.Errorf("the schema does not mention the argument the server declared: %s", schemaJSON(t, greet))
	}

	// And it is a working tool, not just a description of one.
	res, err := toolNamed(t, tools, "greet").Run(t.Context(), ai.ToolCall{ID: "c1", Name: "greet", Input: `{"name":"world"}`})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := res.Text(); got != "hello world" {
		t.Errorf("result = %q, want %q", got, "hello world")
	}
}

// Two servers may both advertise "search", and an agent handed both answers
// every call with whichever came first — silently. Naming the server is how a
// caller says which is which.
func TestNamingTheServerNamespacesItsTools(t *testing.T) {
	tools, err := connect(t, "fs").Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	toolNamed(t, tools, "fs__greet")

	// The server is still called by its own name on the wire, or it would not
	// recognise the call.
	res, err := toolNamed(t, tools, "fs__greet").Run(t.Context(), ai.ToolCall{ID: "c1", Input: `{"name":"world"}`})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := res.Text(); got != "hello world" {
		t.Errorf("result = %q — the qualified name went out instead of the server's own", got)
	}
}

// A tool that failed is not a turn that failed. The model is told what went
// wrong, because that is what lets it correct itself, and the call is recorded
// as an error.
func TestAFailedToolTellsTheModelWhyAndStillReportsTheFailure(t *testing.T) {
	tools, err := connect(t, "").Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	res, err := toolNamed(t, tools, "broken").Run(t.Context(), ai.ToolCall{ID: "c1", Input: `{}`})
	if err == nil {
		t.Fatal("a tool the server marked as failed was reported as a success")
	}
	if got := res.Text(); got != "the path does not exist" {
		t.Errorf("the model is told %q, want the reason the server gave", got)
	}
	if !strings.Contains(err.Error(), "the path does not exist") {
		t.Errorf("the error says %q, want the reason the server gave", err)
	}
}

// An image a tool returned costs what an image costs anywhere else and has to
// arrive as one, or the two protocols that carry images cannot be sent it.
func TestAnImageComesBackAsAnImage(t *testing.T) {
	tools, err := connect(t, "").Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	res, err := toolNamed(t, tools, "picture").Run(t.Context(), ai.ToolCall{ID: "c1", Input: `{}`})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.Content.HasImages() {
		t.Fatalf("content = %+v, want an image block", res.Content)
	}
	for _, b := range res.Content {
		if b.Type != ai.BlockImage {
			continue
		}
		if b.Image.MediaType != "image/png" {
			t.Errorf("media type = %q", b.Image.MediaType)
		}
		if want := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G'}); b.Image.Data != want {
			t.Errorf("data = %q, want %q — an image is carried base64 with no data: prefix", b.Image.Data, want)
		}
	}
}

// A server reached neither way, or both ways, is a configuration mistake worth
// a sentence rather than a connection attempt.
func TestAServerHasToSayHowItIsReached(t *testing.T) {
	for _, tc := range []struct {
		name   string
		server mcp.Server
		says   string
	}{
		{"neither", mcp.Server{Name: "empty"}, "neither a command"},
		{"both", mcp.Server{Name: "both", Command: "x", URL: "http://y"}, "both a command and a URL"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := mcp.Connect(t.Context(), tc.server)
			if err == nil {
				t.Fatal("a server that cannot be reached connected")
			}
			if !strings.Contains(err.Error(), tc.says) {
				t.Errorf("error = %q, want it to say %q", err, tc.says)
			}
		})
	}
}

// A tool taking no arguments is still called with an object. Several servers
// reject a null there, and the model routinely sends nothing at all.
func TestAToolWithNoArgumentsIsStillCallable(t *testing.T) {
	tools, err := connect(t, "").Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	for _, input := range []string{"", "{}", "null"} {
		if _, err := toolNamed(t, tools, "picture").Run(t.Context(), ai.ToolCall{ID: "c1", Input: input}); err != nil {
			t.Errorf("Run with input %q: %v", input, err)
		}
	}
}

func schemaJSON(t *testing.T, s ai.Schema) string {
	t.Helper()
	b, err := json.Marshal(s.Definition)
	if err != nil {
		t.Fatalf("the schema does not marshal: %v", err)
	}
	return string(b)
}

// A session that ended has to say so. A child process that died is otherwise
// indistinguishable from an idle one until the next call hangs.
func TestASessionSaysWhenItIsOver(t *testing.T) {
	c := connect(t, "")
	if !c.Alive() {
		t.Fatal("a session reported itself dead as soon as it connected")
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	for deadline := time.Now().Add(5 * time.Second); c.Alive(); {
		if time.Now().After(deadline) {
			t.Fatal("a closed session still reports itself alive")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A tool the loop is given must survive the client that produced it being
// asked for its tools again — the application re-reads them on every change.
func TestToolsAreReadFreshRatherThanCached(t *testing.T) {
	c := connect(t, "")
	first, err := c.Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	second, err := c.Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools again: %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("a second read saw %d tools, the first saw %d", len(second), len(first))
	}
	if _, err := toolNamed(t, second, "greet").Run(t.Context(), ai.ToolCall{ID: "c1", Input: `{"name":"x"}`}); err != nil {
		t.Errorf("a tool from the second read does not work: %v", err)
	}
}

// An application shows what a server brought, which is more than its tools: the
// /mcp listing in San counts resources and prompts beside them.
func TestAServerAlsoSaysWhatElseItOffers(t *testing.T) {
	c := connect(t, "")

	resources, err := c.Resources(t.Context())
	if err != nil {
		t.Fatalf("Resources: %v", err)
	}
	if len(resources) != 1 || resources[0].URI != "file:///tmp/notes.md" {
		t.Fatalf("resources = %+v, want the one the server offers", resources)
	}
	if resources[0].MediaType != "text/markdown" {
		t.Errorf("media type = %q", resources[0].MediaType)
	}

	prompts, err := c.Prompts(t.Context())
	if err != nil {
		t.Fatalf("Prompts: %v", err)
	}
	if len(prompts) != 1 || prompts[0].Name != "review" {
		t.Fatalf("prompts = %+v, want the one the server offers", prompts)
	}
}

// A server that adds a tool while connected says so, and the handler has to be
// able to ask what the tools now are — that is the only useful thing to do in
// one. It works because the protocol layer dispatches the notification off the
// goroutine reading the connection; if that ever stopped being true, a handler
// like this would wait forever for a reply nobody is left to read, and the
// session would be wedged rather than merely slow. So it is pinned here.
func TestAToolsChangedHandlerCanAskWhatChanged(t *testing.T) {
	saw := make(chan int, 4)
	c, err := mcp.Connect(t.Context(), mcp.Server{
		Command: os.Args[0],
		Env:     map[string]string{serverEnv: "1", notifyEnv: "1"},
	}, mcp.OnToolsChanged(func(c *mcp.Client) {
		tools, err := c.Tools(context.Background())
		if err != nil {
			saw <- -1
			return
		}
		saw <- len(tools)
	}))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })

	select {
	case n := <-saw:
		if n < 0 {
			t.Fatal("the handler could not read the tools it was told had changed")
		}
		if n != 4 {
			t.Errorf("the handler saw %d tools, want the 4 the server now offers", n)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("no tools-changed notification, or a handler that never returned from asking")
	}
}
