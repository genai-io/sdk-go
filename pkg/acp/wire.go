package acp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
)

// Error is a structured JSON-RPC failure; other handler errors are redacted.
type Error = jsonrpc.Error

type envelope struct {
	Version string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  json.RawMessage `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   json.RawMessage `json:"error,omitempty"`
}

type message struct {
	id     json.RawMessage
	key    string
	method string
	params json.RawMessage
	result json.RawMessage
	err    *Error
}

func readFrame(r *bufio.Reader, limit int) ([]byte, error) {
	var frame []byte
	for {
		part, err := r.ReadSlice('\n')
		if len(part) > limit+1-len(frame) {
			return nil, ErrFrameTooLarge
		}
		frame = append(frame, part...)
		if err == nil {
			return frame[:len(frame)-1], nil
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if err == io.EOF && len(frame) != 0 {
			return nil, ErrInvalidMessage
		}
		return nil, err
	}
}

func decode(frame []byte) (message, error) {
	var e envelope
	if !utf8.Valid(frame) || json.Unmarshal(frame, &e) != nil || e.Version != "2.0" {
		return message{}, ErrInvalidMessage
	}
	m := message{id: e.ID, params: e.Params, result: e.Result}
	if len(e.ID) != 0 {
		var err error
		m.key, err = idKey(e.ID)
		if err != nil {
			return message{}, err
		}
	}
	if len(e.Method) != 0 {
		if json.Unmarshal(e.Method, &m.method) != nil || m.method == "" || len(e.Result) != 0 || len(e.Error) != 0 || !structured(e.Params) {
			return message{}, ErrInvalidMessage
		}
		return m, nil
	}
	if len(e.ID) == 0 || len(e.Params) != 0 || (len(e.Result) == 0) == (len(e.Error) == 0) {
		return message{}, ErrInvalidMessage
	}
	if len(e.Error) != 0 {
		var failure struct {
			Code    *int64          `json:"code"`
			Message *string         `json:"message"`
			Data    json.RawMessage `json:"data"`
		}
		if json.Unmarshal(e.Error, &failure) != nil || failure.Code == nil || failure.Message == nil {
			return message{}, ErrInvalidMessage
		}
		m.err = &Error{Code: *failure.Code, Message: *failure.Message, Data: failure.Data}
	}
	return m, nil
}

func idKey(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if bytes.Equal(raw, []byte("null")) {
		return "null", nil
	}
	if len(raw) != 0 && raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return "s:" + s, nil
		}
	} else {
		var n int64
		if json.Unmarshal(raw, &n) == nil {
			return "n:" + strconv.FormatInt(n, 10), nil
		}
	}
	return "", ErrInvalidMessage
}

func structured(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) == 0 || raw[0] == '{' || raw[0] == '[' || bytes.Equal(raw, []byte("null"))
}

func requestFrame(id json.RawMessage, method string, params any) ([]byte, error) {
	if method == "" {
		return nil, fmt.Errorf("%w: empty method", ErrInvalidMessage)
	}
	name, _ := json.Marshal(method)
	e := envelope{Version: "2.0", ID: id, Method: name}
	if params != nil {
		var err error
		e.Params, err = json.Marshal(params)
		if err != nil {
			return nil, err
		}
		if !structured(e.Params) {
			return nil, fmt.Errorf("%w: params must be structured", ErrInvalidMessage)
		}
	}
	return json.Marshal(e)
}

func responseFrame(id json.RawMessage, result any, failure *Error) ([]byte, error) {
	e := envelope{Version: "2.0", ID: id}
	var err error
	if failure != nil {
		e.Error, err = json.Marshal(failure)
	} else {
		e.Result, err = json.Marshal(result)
	}
	if err != nil {
		return nil, err
	}
	return json.Marshal(e)
}
