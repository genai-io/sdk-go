package acp

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

// ErrElicitationUnsupported means the client did not advertise the requested mode.
var ErrElicitationUnsupported = errors.New("acp: elicitation mode not supported by client")

func elicitationCapabilities(req protocol.InitializeRequest) (bool, bool) {
	c := req.ClientCapabilities.Value.Elicitation
	if !c.Present || c.Null {
		return false, false
	}
	return c.Value.Form.Present && !c.Value.Form.Null, c.Value.URL.Present && !c.Value.URL.Null
}

// CreateElicitation sends a typed elicitation. Low-level callers must check
// negotiated capabilities and own the request scope and response validation.
func (c *Conn) CreateElicitation(ctx context.Context, req protocol.CreateElicitationRequest) (protocol.CreateElicitationResponse, error) {
	return callProtocol[protocol.CreateElicitationResponse](ctx, c, "elicitation/create", req)
}

// CompleteElicitation notifies the client that a URL flow completed.
func (c *Conn) CompleteElicitation(ctx context.Context, req protocol.CompleteElicitationNotification) error {
	return notifyProtocol(ctx, c, "elicitation/complete", req)
}

// ElicitationHandler adapts a typed client callback to Handlers.Request. Unknown
// methods are rejected. Callback contexts have connection lifetime, as with
// permission handlers: local cancellation cannot close a remote UI.
func ElicitationHandler(fn func(context.Context, protocol.CreateElicitationRequest) (protocol.CreateElicitationResponse, error)) func(context.Context, *Conn, string, json.RawMessage) (any, error) {
	return func(ctx context.Context, _ *Conn, method string, params json.RawMessage) (any, error) {
		if method != "elicitation/create" || fn == nil {
			return nil, &Error{Code: -32601, Message: "Method not found"}
		}
		var req protocol.CreateElicitationRequest
		if json.Unmarshal(params, &req) != nil {
			return nil, &Error{Code: -32602, Message: "Invalid elicitation"}
		}
		return fn(ctx, req)
	}
}

// CreateElicitation binds an advertised form or URL mode to this turn's session.
// It uses the same cancellation and frame-write barrier as RequestPermission;
// late replies are discarded. Applications must validate accepted form content
// against the offered fields before using it. No remote UI cancellation is sent.
func (t *Turn) CreateElicitation(req protocol.CreateElicitationRequest) (protocol.CreateElicitationResponse, error) {
	return t.CreateElicitationContext(t.ctx, req)
}

// CreateElicitationContext additionally bounds the local wait by the caller's
// context, without weakening the turn lifetime or response-write barrier.
func (t *Turn) CreateElicitationContext(ctx context.Context, req protocol.CreateElicitationRequest) (protocol.CreateElicitationResponse, error) {
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(t.ctx, cancel)
	defer stop()
	defer cancel()
	if t.ctx.Err() != nil {
		cancel()
	}
	var res protocol.CreateElicitationResponse
	if err := t.begin(); err != nil {
		return res, err
	}
	defer t.work.Done()
	switch v := req.Value.(type) {
	case *protocol.CreateElicitationRequestFormVariant:
		if v != nil {
			req.Value = *v
		}
	case *protocol.CreateElicitationRequestURLVariant:
		if v != nil {
			req.Value = *v
		}
	}
	switch v := req.Value.(type) {
	case protocol.CreateElicitationRequestFormVariant:
		if !t.forms {
			return res, ErrElicitationUnsupported
		}
		v.SessionID = protocol.Some(t.sessionID)
		v.RequestID = protocol.Optional[protocol.RequestID]{}
		req.Value = v
	case protocol.CreateElicitationRequestURLVariant:
		if !t.urls {
			return res, ErrElicitationUnsupported
		}
		v.SessionID = protocol.Some(t.sessionID)
		v.RequestID = protocol.Optional[protocol.RequestID]{}
		req.Value = v
	default:
		return res, ErrElicitationUnsupported
	}
	params, err := marshalProtocol("elicitation/create", req)
	if err != nil {
		return res, err
	}
	var wire json.RawMessage
	if err := t.conn.call(ctx, t.conn.ctx, "elicitation/create", params, &wire); err != nil {
		return res, err
	}
	err = json.Unmarshal(wire, &res)
	return res, err
}
