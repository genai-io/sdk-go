package acp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

// Initialize sends the client's protocol version and capabilities to the agent.
// It does not maintain initialization state or negotiate application policy.
func (c *Conn) Initialize(ctx context.Context, request protocol.InitializeRequest) (protocol.InitializeResponse, error) {
	return callProtocol[protocol.InitializeResponse](ctx, c, "initialize", "InitializeRequest", "InitializeResponse", request)
}

// NewSession asks the agent to create a session. Storage and admission remain
// the agent application's responsibility.
func (c *Conn) NewSession(ctx context.Context, request protocol.NewSessionRequest) (protocol.NewSessionResponse, error) {
	return callProtocol[protocol.NewSessionResponse](ctx, c, "session/new", "NewSessionRequest", "NewSessionResponse", request)
}

// Prompt waits for a prompt result and its preceding notifications. It does not
// serialize prompts for a session. Context cancellation stops the local wait,
// not remote work; cancellation during an active frame write closes the
// connection. Use Cancel to request session-level cancellation from the agent.
func (c *Conn) Prompt(ctx context.Context, request protocol.PromptRequest) (protocol.PromptResponse, error) {
	return callProtocol[protocol.PromptResponse](ctx, c, "session/prompt", "PromptRequest", "PromptResponse", request)
}

// Cancel sends session/cancel. It neither waits for the turn to unwind nor
// cancels the local Prompt wait; the agent should still answer that prompt.
func (c *Conn) Cancel(ctx context.Context, notification protocol.CancelNotification) error {
	return notifyProtocol(ctx, c, "session/cancel", "CancelNotification", notification)
}

// SessionUpdate sends an agent-to-client update. Completion means the frame was
// written, not that the client has rendered or processed it.
func (c *Conn) SessionUpdate(ctx context.Context, notification protocol.SessionNotification) error {
	return notifyProtocol(ctx, c, "session/update", "SessionNotification", notification)
}

// RequestPermission asks the client for a permission decision. The application
// must validate the selected option and apply its own authorization policy.
func (c *Conn) RequestPermission(ctx context.Context, request protocol.RequestPermissionRequest) (protocol.RequestPermissionResponse, error) {
	return callProtocol[protocol.RequestPermissionResponse](ctx, c, "session/request_permission", "RequestPermissionRequest", "RequestPermissionResponse", request)
}

func callProtocol[Response any](ctx context.Context, c *Conn, method, requestName, responseName string, request any) (Response, error) {
	var response Response
	if err := ctx.Err(); err != nil {
		return response, err
	}
	params, err := marshalProtocol(requestName, request)
	if err != nil {
		return response, err
	}
	var wire json.RawMessage
	if err := c.Call(ctx, method, params, &wire); err != nil {
		return response, err
	}
	if err := protocol.ValidateJSON(responseName, wire); err != nil {
		return response, fmt.Errorf("acp: invalid %s: %w", responseName, err)
	}
	if err := json.Unmarshal(wire, &response); err != nil {
		return response, fmt.Errorf("acp: decode %s: %w", responseName, err)
	}
	return response, nil
}

func notifyProtocol(ctx context.Context, c *Conn, method, definition string, notification any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	params, err := marshalProtocol(definition, notification)
	if err != nil {
		return err
	}
	return c.Notify(ctx, method, params)
}

func marshalProtocol(definition string, value any) (json.RawMessage, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("acp: encode %s: %w", definition, err)
	}
	if err := protocol.ValidateJSON(definition, data); err != nil {
		return nil, fmt.Errorf("acp: invalid %s: %w", definition, err)
	}
	return data, nil
}
