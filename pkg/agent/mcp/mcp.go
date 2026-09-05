// Package mcp reaches Model Context Protocol servers, so that what they
// advertise arrives as [agent.Tool].
//
//	c, err := mcp.Connect(ctx, mcp.Server{Name: "fs", Command: "mcp-server-filesystem"})
//	defer c.Close()
//	tools, err := c.Tools(ctx)
//	a, err := agent.New(client, agent.WithTools(tools...))
//
// The protocol itself is the official SDK's — github.com/modelcontextprotocol/go-sdk
// — on the same terms as every driver in pkg/ai/driver, which wrap their
// vendor's own client rather than restating a wire format. What is here is the
// part neither side can own: an MCP tool is a name, a JSON Schema and a call,
// and so is an agent's, but they are different types, and the translation
// between them has decisions in it that this package makes once.
package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Separator joins a server's name to a tool's. Two servers may both advertise
// "search", and an agent handed both would answer every call with whichever
// came first — silently, because nothing about two tools with one name is an
// error until the model picks the wrong one.
const Separator = "__"

// Server is an MCP server and how to reach it.
//
// Exactly one of Command and URL says which: a command is run as a child
// process and spoken to over its stdin and stdout, a URL is reached over HTTP.
type Server struct {
	// Name namespaces the tools this server advertises: "fs" turns "read" into
	// "fs__read". Leave it empty to take the server's names as they come,
	// which is right for a single server and a silent collision for two.
	//
	// It is also what an error from this package calls the server.
	Name string

	// Command is the program to run, with Args, in Dir. Env is added to this
	// process's environment rather than replacing it, because a server
	// launched with nothing tends to fail on PATH or HOME rather than on the
	// variable that was actually forgotten.
	Command string
	Args    []string
	Env     map[string]string
	Dir     string

	// Stderr is where the child process's stderr goes. Nil discards it.
	//
	// A server that fails to start usually says why here and nowhere else, so
	// send it somewhere you will read. Not os.Stderr from a full-screen
	// program: a server writing there paints over the interface.
	Stderr io.Writer

	// URL is the endpoint of a server reached over HTTP, with Headers on every
	// request. SSE selects the older 2024-11-05 transport; the default is
	// streamable HTTP.
	URL     string
	Headers map[string]string
	SSE     bool

	// HTTPClient is used for every request when set.
	HTTPClient *http.Client
}

// Client is one connected session with one server.
type Client struct {
	server  Server
	session *mcpsdk.ClientSession
	done    chan struct{}
}

// Option is something to ask of a session, set when it is opened.
type Option func(*options)

type options struct {
	toolsChanged func()
	keepAlive    time.Duration
}

// OnToolsChanged is called when the server says its tool list has changed —
// which servers that load tools lazily, or gate them behind a login, do while
// connected.
//
// What to do about it is the caller's: an agent is given its tools when it is
// built, so a set that changed is a set the application has to hand over again.
// It runs on the session's own goroutine, so do the work elsewhere if it is
// slow.
func OnToolsChanged(fn func()) Option {
	return func(o *options) { o.toolsChanged = fn }
}

// KeepAlive pings the server on this interval and closes the session when it
// stops answering. Without it a server that has stopped listening — a child
// process wedged rather than exited — is indistinguishable from an idle one
// until the next call hangs.
func KeepAlive(d time.Duration) Option {
	return func(o *options) { o.keepAlive = d }
}

// Implementation is how this SDK introduces itself to a server. A server may
// log it, or vary what it advertises by it.
var Implementation = &mcpsdk.Implementation{Name: "genai-io/sdk-go", Version: "v1"}

// Connect opens a session with one server. The returned client must be closed;
// for a command server, closing is what stops the child process.
func Connect(ctx context.Context, s Server, opts ...Option) (*Client, error) {
	transport, err := s.transport()
	if err != nil {
		return nil, err
	}

	var cfg options
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	clientOpts := &mcpsdk.ClientOptions{KeepAlive: cfg.keepAlive}
	if cfg.toolsChanged != nil {
		clientOpts.ToolListChangedHandler = func(context.Context, *mcpsdk.ToolListChangedRequest) {
			cfg.toolsChanged()
		}
	}

	session, err := mcpsdk.NewClient(Implementation, clientOpts).Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp: connecting to %s: %w", s.describe(), err)
	}

	c := &Client{server: s, session: session, done: make(chan struct{})}
	go func() {
		defer close(c.done)
		_ = session.Wait()
	}()
	return c, nil
}

// Done closes when the session ends, however it ended: closed here, closed by
// the server, or a child process that died. Select on it to notice.
func (c *Client) Done() <-chan struct{} { return c.done }

// Alive reports whether the session is still up — the same fact as Done, for
// the caller that is asking rather than waiting.
func (c *Client) Alive() bool {
	select {
	case <-c.done:
		return false
	default:
		return true
	}
}

// transport is how this server is reached, or why it cannot be.
func (s Server) transport() (mcpsdk.Transport, error) {
	switch {
	case s.Command != "" && s.URL != "":
		return nil, fmt.Errorf("mcp: server %s gives both a command and a URL; it is reached one way or the other", s.describe())

	case s.Command != "":
		cmd := exec.Command(s.Command, s.Args...)
		cmd.Dir = s.Dir
		cmd.Stderr = s.Stderr
		cmd.Env = os.Environ()
		for k, v := range s.Env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
		return &mcpsdk.CommandTransport{Command: cmd}, nil

	case s.URL != "":
		httpClient := s.HTTPClient
		if len(s.Headers) > 0 {
			httpClient = withHeaders(httpClient, s.Headers)
		}
		if s.SSE {
			return &mcpsdk.SSEClientTransport{Endpoint: s.URL, HTTPClient: httpClient}, nil
		}
		return &mcpsdk.StreamableClientTransport{Endpoint: s.URL, HTTPClient: httpClient}, nil
	}
	return nil, fmt.Errorf("mcp: server %s gives neither a command to run nor a URL to reach", s.describe())
}

// Tools is what this server advertises, ready to hand to agent.WithTools.
//
// Read every time rather than cached: a server may add and remove tools while
// it is connected, and a cache here would be a second answer to what the model
// can call.
func (c *Client) Tools(ctx context.Context) ([]agent.Tool, error) {
	var out []agent.Tool
	for t, err := range c.session.Tools(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("mcp: listing tools on %s: %w", c.server.describe(), err)
		}
		out = append(out, &tool{
			client: c,
			name:   t.Name,
			schema: ai.Schema{
				Name:        c.server.qualify(t.Name),
				Description: t.Description,
				Definition:  t.InputSchema,
			},
		})
	}
	return out, nil
}

// Close ends the session. For a command server this stops the child process.
func (c *Client) Close() error { return c.session.Close() }

// Server is the server this client was opened for.
func (c *Client) Server() Server { return c.server }

// qualify is the name the model is told, which is the server's own only when
// the caller did not namespace it.
func (s Server) qualify(name string) string {
	if s.Name == "" {
		return name
	}
	return s.Name + Separator + name
}

func (s Server) describe() string {
	switch {
	case s.Name != "":
		return s.Name
	case s.Command != "":
		return s.Command
	case s.URL != "":
		return s.URL
	}
	return "(unnamed)"
}

// tool is one of a server's tools, as the loop takes it. It holds the client
// rather than the session so that the name sent back is always the server's
// own, whatever the caller namespaced it to.
type tool struct {
	client *Client
	name   string
	schema ai.Schema
}

func (t *tool) Schema() ai.Schema { return t.schema }

// Run calls the tool and translates what came back.
//
// A tool that failed returns its content and an error: the loop tells the
// model the content — which is where a server puts the reason, so that a model
// can correct itself — and records the call as failed. A transport failure has
// no content, and only the error.
func (t *tool) Run(ctx context.Context, call ai.ToolCall) (agent.Result, error) {
	args, err := decodeArgs(call.Input)
	if err != nil {
		return agent.Result{}, fmt.Errorf("mcp: %s was called with arguments that are not a JSON object: %w", t.schema.Name, err)
	}

	res, err := t.client.session.CallTool(ctx, &mcpsdk.CallToolParams{Name: t.name, Arguments: args})
	if err != nil {
		return agent.Result{}, fmt.Errorf("mcp: calling %s on %s: %w", t.name, t.client.server.describe(), err)
	}

	result := agent.Result{Content: content(res.Content)}
	if res.StructuredContent != nil {
		// The interface's half of the answer, never the model's: a server that
		// publishes an output schema answers twice, once as text for the model
		// and once as the value behind it.
		result.Details = res.StructuredContent
	}
	if res.IsError {
		return result, errors.New(errorText(result.Content))
	}
	return result, nil
}

// content is what the model is told, from what the server returned. Kinds this
// SDK cannot put in a prompt — an embedded resource, an audio clip — are named
// rather than dropped, because a tool result that silently loses half its
// answer reads to the model as a tool that did half the work.
func content(blocks []mcpsdk.Content) ai.Content {
	var out ai.Content
	for _, block := range blocks {
		switch b := block.(type) {
		case *mcpsdk.TextContent:
			out = append(out, ai.TextBlock(b.Text))
		case *mcpsdk.ImageContent:
			out = append(out, ai.ImageBlock(ai.Image{
				MediaType: b.MIMEType,
				Data:      base64.StdEncoding.EncodeToString(b.Data),
			}))
		default:
			out = append(out, ai.TextBlock(fmt.Sprintf("(%T, which this client cannot pass on)", block)))
		}
	}
	return out
}

// errorText is what the failure says, for the error the loop carries. The
// content still reaches the model; this is the same thing for a log and a
// session record.
func errorText(c ai.Content) string {
	if text := strings.TrimSpace(c.Text()); text != "" {
		return text
	}
	return "the tool reported an error and said nothing about it"
}

// decodeArgs turns the model's arguments into the object a server expects. An
// empty input is an empty object, not a null: a tool that takes no arguments is
// still called with {}, and several servers reject a null there.
func decodeArgs(input string) (map[string]any, error) {
	if strings.TrimSpace(input) == "" {
		return map[string]any{}, nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		return nil, err
	}
	if args == nil {
		return map[string]any{}, nil
	}
	return args, nil
}

// Resource is something a server offers to read: a file, a database row, a
// page. It is not a tool — the model cannot call it — and what to do with one
// is the application's: put it in the system prompt, offer it in a picker,
// count it in a status line.
type Resource struct {
	URI         string
	Name        string
	Description string
	// MediaType is the IANA type, named as ai.Image names it rather than as
	// the protocol spells it.
	MediaType string
}

// Prompt is a prompt template a server offers, for a person to pick.
type Prompt struct {
	Name        string
	Description string
}

// Resources is what this server offers to read.
//
// Reading one is not here, and neither is fetching a prompt's messages:
// both answer in the server's own shapes, and turning those into ai.Message
// has decisions in it that no caller has yet had an opinion about. Listing is
// what an application needs to show what a server brought.
func (c *Client) Resources(ctx context.Context) ([]Resource, error) {
	var out []Resource
	for r, err := range c.session.Resources(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("mcp: listing resources on %s: %w", c.server.describe(), err)
		}
		out = append(out, Resource{URI: r.URI, Name: r.Name, Description: r.Description, MediaType: r.MIMEType})
	}
	return out, nil
}

// Prompts is what this server offers as templates.
func (c *Client) Prompts(ctx context.Context) ([]Prompt, error) {
	var out []Prompt
	for p, err := range c.session.Prompts(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("mcp: listing prompts on %s: %w", c.server.describe(), err)
		}
		out = append(out, Prompt{Name: p.Name, Description: p.Description})
	}
	return out, nil
}
