package acp

import (
	"context"
	"encoding/json"

	"github.com/genai-io/sdk-go/pkg/acp/protocol"
)

// Called with a.mu held. Reservations survive until the response write barrier.
func (a *Agent) admitSessionChange(c *Conn, m message) (func() (any, error), func()) {
	if (m.method == "session/load" && a.handlers.LoadSession == nil) ||
		(m.method == "session/set_mode" && a.handlers.SetSessionMode == nil) ||
		(m.method == "session/set_config_option" && a.handlers.SetSessionConfigOption == nil) {
		return agentReject(-32601, "Method not found")
	}
	if !a.initialized {
		return agentReject(-32600, "Not initialized")
	}
	if m.method == "session/load" {
		var req protocol.LoadSessionRequest
		if json.Unmarshal(m.params, &req) != nil || req.SessionID == "" {
			return agentReject(-32602, "Invalid load parameters")
		}
		if a.sessions[req.SessionID] != nil {
			return agentReject(-32001, "Session already admitted")
		}
		if a.slots >= a.maxSessions {
			return agentReject(-32001, "Session limit reached")
		}
		ctx, cancel := context.WithCancel(c.ctx)
		turn := &Turn{conn: c, sessionID: req.SessionID, ctx: ctx, cancel: cancel, forms: a.forms, urls: a.urls}
		session := &agentSession{turn: turn}
		a.sessions[req.SessionID] = session
		a.slots++
		succeeded := false
		return func() (any, error) {
				defer turn.finish()
				res, err := a.handlers.LoadSession(ctx, turn, req)
				if err != nil {
					return nil, err
				}
				b, err := json.Marshal(res)
				succeeded = err == nil
				return json.RawMessage(b), err
			}, func() {
				a.mu.Lock()
				defer a.mu.Unlock()
				if succeeded {
					session.ready = true
					session.turn = nil
				} else {
					delete(a.sessions, req.SessionID)
					a.slots--
				}
			}
	}
	var id protocol.SessionID
	var invoke func() (any, error)
	if m.method == "session/set_mode" {
		var req protocol.SetSessionModeRequest
		if json.Unmarshal(m.params, &req) != nil {
			return agentReject(-32602, "Invalid mode parameters")
		}
		id = req.SessionID
		invoke = func() (any, error) { return a.handlers.SetSessionMode(c.ctx, req) }
	} else {
		var req protocol.SetSessionConfigOptionRequest
		if json.Unmarshal(m.params, &req) != nil {
			return agentReject(-32602, "Invalid configuration parameters")
		}
		switch v := req.Value.(type) {
		case protocol.SetSessionConfigOptionRequestBooleanVariant:
			id = v.SessionID
		case protocol.SetSessionConfigOptionRequestValueIDVariant:
			id = v.SessionID
		default:
			return agentReject(-32602, "Invalid configuration variant")
		}
		invoke = func() (any, error) { return a.handlers.SetSessionConfigOption(c.ctx, req) }
	}
	session := a.sessions[id]
	if session == nil || !session.ready {
		return agentReject(-32002, "Unknown session")
	}
	if session.turn != nil || session.configuring {
		return agentReject(-32001, "Session busy")
	}
	session.configuring = true
	return invoke, func() { a.mu.Lock(); session.configuring = false; a.mu.Unlock() }
}
