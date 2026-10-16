package ai

import (
	"context"
	"errors"
	"io"
	"net"
	"testing"
)

// classifyTransport is the last word for a failure with no status, and it has
// to leave a signal it does not recognise alone rather than guessing.
func TestClassifyTransportOnlyClaimsWhatItKnows(t *testing.T) {
	timeout := &net.DNSError{IsTimeout: true}
	for name, tc := range map[string]struct {
		err  error
		want ErrorKind
		ok   bool
	}{
		"nothing":         {nil, KindUnknown, false},
		"a cancel":        {context.Canceled, KindCanceled, true},
		"a wrapped EOF":   {errors.New("read: " + io.EOF.Error()), KindUnknown, false},
		"a real EOF":      {io.EOF, KindNetwork, true},
		"a deadline":      {context.DeadlineExceeded, KindNetwork, true},
		"a net timeout":   {timeout, KindNetwork, true},
		"a plain failure": {errors.New("the model refused"), KindUnknown, false},
	} {
		t.Run(name, func(t *testing.T) {
			kind, ok := classifyTransport(tc.err)
			if kind != tc.want || ok != tc.ok {
				t.Errorf("classifyTransport = %q, %v; want %q, %v", kind, ok, tc.want, tc.ok)
			}
		})
	}
}

// The message is what a human reads first, so a part that is missing has to
// take its separator with it.
func TestErrorRendersOnlyThePartsItHas(t *testing.T) {
	for name, tc := range map[string]struct {
		err  *Error
		want string
	}{
		"everything": {
			&Error{Driver: "anthropic", Kind: KindAuth, Status: 401, Message: "bad key"},
			"anthropic: auth (http 401): bad key",
		},
		"no driver": {
			&Error{Kind: KindInvalidRequest, Message: "ai: tool 0 has no name"},
			"invalid_request: ai: tool 0 has no name",
		},
		"no kind": {
			&Error{Driver: "openai", Message: "something happened"},
			"openai: something happened",
		},
		"a wrapped cause and nothing else": {
			&Error{Err: errors.New("connection refused")},
			"connection refused",
		},
		"nothing at all": {
			&Error{},
			"ai: unspecified error",
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := tc.err.Error(); got != tc.want {
				t.Errorf("Error() = %q, want %q", got, tc.want)
			}
		})
	}
}

// An application asking whether to retry holds a bare error, not the six
// things a driver hands Classify. A dropped connection is worth another go
// whether or not somebody wrapped it first.
func TestIsRetryableAnswersForAnUnwrappedTransportFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"a mid-stream cutoff", io.ErrUnexpectedEOF, true},
		{"a request that timed out", context.DeadlineExceeded, true},
		{"a user interrupt", context.Canceled, false},
		{"something else entirely", errors.New("bad json"), false},
		{"nothing at all", nil, false},
		{"a classified failure still answers from its kind",
			&Error{Kind: KindAuth}, false},
		{"and so does a retryable one", &Error{Kind: KindOverloaded}, true},
	} {
		if got := IsRetryable(tc.err); got != tc.want {
			t.Errorf("IsRetryable(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}
