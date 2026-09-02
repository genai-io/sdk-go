package ai

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"testing"
	"time"
)

func withRetryAfter(value string) *http.Response {
	resp := &http.Response{Header: http.Header{}}
	resp.Header.Set("Retry-After", value)
	return resp
}

// Classification decides between retrying, compacting and giving up, so a
// check that fires on the wrong input makes a recoverable failure fatal.
func TestClassifyReadsEachSignalWhereItBelongs(t *testing.T) {
	for name, tc := range map[string]struct {
		status    int
		resp      *http.Response
		message   string
		err       error
		want      ErrorKind
		retryable bool
		after     time.Duration
	}{
		"a 400 saying the prompt is too long is a context overflow": {
			status: http.StatusBadRequest, message: "prompt is too long: 300000 tokens > 200000",
			want: KindContextExceeded,
		},
		"an overflow with no status at all is still one": {
			message: "maximum context length exceeded",
			want:    KindContextExceeded,
		},
		"a 401 mentioning tokens is still a bad credential": {
			// The word "token" means a credential here; reading the message
			// first would call it a context overflow and stop retrying.
			status: http.StatusUnauthorized, message: "too many tokens in the authorization header",
			want: KindAuth,
		},
		"a 503 mentioning tokens is still a busy server": {
			status: http.StatusServiceUnavailable, message: "upstream rejected: too many tokens",
			want: KindOverloaded, retryable: true,
		},
		"a 429 carries the provider's own hint": {
			status: http.StatusTooManyRequests, resp: withRetryAfter("3"),
			message: "slow down",
			want:    KindRateLimit, retryable: true, after: 3 * time.Second,
		},
		"a 529 is Anthropic's overload": {
			status: 529, message: "overloaded_error",
			want: KindOverloaded, retryable: true,
		},
		"a 404 is a request problem": {
			status: http.StatusNotFound, message: "no such model",
			want: KindInvalidRequest,
		},
		"a cancel outranks whatever the transport made of it": {
			err:  context.Canceled,
			want: KindCanceled,
		},
		"a dropped connection is worth another try": {
			err:  io.ErrUnexpectedEOF,
			want: KindNetwork, retryable: true,
		},
		"an unrecognised failure with no status stays unknown": {
			message: "something went wrong",
			want:    KindUnknown,
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := Classify("stub", tc.status, tc.resp, "", tc.message, tc.err)
			if got.Kind != tc.want {
				t.Errorf("kind = %q, want %q", got.Kind, tc.want)
			}
			if got.Retryable() != tc.retryable {
				t.Errorf("retryable = %v, want %v", got.Retryable(), tc.retryable)
			}
			if got.RetryAfter != tc.after {
				t.Errorf("retry-after = %v, want %v", got.RetryAfter, tc.after)
			}
		})
	}
}

// A driver that already classified a failure has more to go on than Classify
// does, so its answer must survive being handed back through here.
func TestClassifyPassesAnAlreadyTypedErrorThrough(t *testing.T) {
	typed := &Error{Driver: "anthropic", Kind: KindContextExceeded, Message: "prompt is too long"}
	if got := Classify("stub", http.StatusUnauthorized, nil, "", "ignored", typed); got != typed {
		t.Errorf("Classify = %+v, want the driver's own error untouched", got)
	}
}

// A stream that dies has usually lost its status, and the transport rarely says
// why. Calling that a network failure is what makes it retryable.
func TestStreamErrorFallsBackToNetwork(t *testing.T) {
	if got := StreamError("stub", 0, nil, "", "connection went away", nil); got.Kind != KindNetwork {
		t.Errorf("kind = %q, want %q", got.Kind, KindNetwork)
	}
	if got := StreamError("stub", http.StatusUnauthorized, nil, "", "bad key", nil); got.Kind != KindAuth {
		t.Errorf("kind = %q, want a real classification to survive", got.Kind)
	}
}

// The overflow signal has to be read out of prose, so the phrases that mean
// something else are as important as the ones that mean overflow.
func TestClassifyMessageIgnoresTheLookalikes(t *testing.T) {
	for _, msg := range []string{
		"prompt is too long",
		"This model's maximum context length is 128000 tokens",
		"Please reduce the length of the messages",
		"too many tokens",
	} {
		if kind, ok := ClassifyMessage(msg); !ok || kind != KindContextExceeded {
			t.Errorf("ClassifyMessage(%q) = %q, %v; want a context overflow", msg, kind, ok)
		}
	}
	for _, msg := range []string{
		"rate limit reached: too many tokens per minute",
		"quota exceeded: maximum context length of your plan",
		"too many requests",
		"please wait before sending more tokens",
		"the server had an error",
		"",
	} {
		if kind, ok := ClassifyMessage(msg); ok {
			t.Errorf("ClassifyMessage(%q) = %q, true; want it left to the status", msg, kind)
		}
	}
}

// Retry-After has two legal forms and a provider may send either.
func TestParseRetryAfterReadsBothForms(t *testing.T) {
	if got := parseRetryAfter(nil); got != 0 {
		t.Errorf("parseRetryAfter(nil) = %v, want 0", got)
	}
	for value, want := range map[string]time.Duration{
		"":                              0,
		"  ":                            0,
		"5":                             5 * time.Second,
		" 5 ":                           5 * time.Second,
		"0":                             0,
		"-3":                            0,
		"soon":                          0,
		"Mon, 02 Jan 2006 15:04:05 GMT": 0, // long past
	} {
		if got := parseRetryAfter(withRetryAfter(value)); got != want {
			t.Errorf("parseRetryAfter(%q) = %v, want %v", value, got, want)
		}
	}

	// An HTTP date is a deadline, so what comes back is the wait until it.
	future := time.Now().Add(90 * time.Second).UTC().Format(http.TimeFormat)
	got := parseRetryAfter(withRetryAfter(future))
	if got <= 60*time.Second || got > 90*time.Second {
		t.Errorf("parseRetryAfter(%q) = %v, want a wait close to 90s", future, got)
	}
}

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
