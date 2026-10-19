package errs

import (
	"context"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai"
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
		want      ai.ErrorKind
		retryable bool
		after     time.Duration
	}{
		"a 400 saying the prompt is too long is a context overflow": {
			status: http.StatusBadRequest, message: "prompt is too long: 300000 tokens > 200000",
			want: ai.KindContextExceeded,
		},
		"an overflow with no status at all is still one": {
			message: "maximum context length exceeded",
			want:    ai.KindContextExceeded,
		},
		"a 401 mentioning tokens is still a bad credential": {
			// The word "token" means a credential here; reading the message
			// first would call it a context overflow and stop retrying.
			status: http.StatusUnauthorized, message: "too many tokens in the authorization header",
			want: ai.KindAuth,
		},
		"a 503 mentioning tokens is still a busy server": {
			status: http.StatusServiceUnavailable, message: "upstream rejected: too many tokens",
			want: ai.KindOverloaded, retryable: true,
		},
		"a 429 carries the provider's own hint": {
			status: http.StatusTooManyRequests, resp: withRetryAfter("3"),
			message: "slow down",
			want:    ai.KindRateLimit, retryable: true, after: 3 * time.Second,
		},
		"a 529 is Anthropic's overload": {
			status: 529, message: "overloaded_error",
			want: ai.KindOverloaded, retryable: true,
		},
		"a 404 is a request problem": {
			status: http.StatusNotFound, message: "no such model",
			want: ai.KindInvalidRequest,
		},
		"a cancel outranks whatever the transport made of it": {
			err:  context.Canceled,
			want: ai.KindCanceled,
		},
		"a dropped connection is worth another try": {
			err:  io.ErrUnexpectedEOF,
			want: ai.KindNetwork, retryable: true,
		},
		"an unrecognised failure with no status stays unknown": {
			message: "something went wrong",
			want:    ai.KindUnknown,
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := classify("stub", Details{Status: tc.status, Response: tc.resp, Message: tc.message}, tc.err, false)
			if got.Kind != tc.want {
				t.Errorf("kind = %q, want %q", got.Kind, tc.want)
			}
			if ai.IsRetryable(got) != tc.retryable {
				t.Errorf("retryable = %v, want %v", ai.IsRetryable(got), tc.retryable)
			}
			if got.RetryAfter != tc.after {
				t.Errorf("retry-after = %v, want %v", got.RetryAfter, tc.after)
			}
		})
	}
}

// A driver that already classified a failure has more to go on than classify
// does, so its answer must survive being handed back through here.
func TestClassifyPassesAnAlreadyTypedErrorThrough(t *testing.T) {
	typed := &ai.Error{Driver: "anthropic", Kind: ai.KindContextExceeded, Message: "prompt is too long"}
	if got := classify("stub", Details{Status: http.StatusUnauthorized, Message: "ignored"}, typed, false); got != typed {
		t.Errorf("classify = %+v, want the driver's own error untouched", got)
	}
}

// A stream that dies has usually lost its status, and the transport rarely says
// why. Calling that a network failure is what makes it retryable.
func TestClassifyStreamFallsBackToNetwork(t *testing.T) {
	if got := classify("stub", Details{Message: "connection went away"}, nil, true); got.Kind != ai.KindNetwork {
		t.Errorf("kind = %q, want %q", got.Kind, ai.KindNetwork)
	}
	if got := classify("stub", Details{Status: http.StatusUnauthorized, Message: "bad key"}, nil, true); got.Kind != ai.KindAuth {
		t.Errorf("kind = %q, want a real classification to survive", got.Kind)
	}
}

// The overflow signal has to be read out of prose, so the phrases that mean
// something else are as important as the ones that mean overflow.
func TestIsContextExceededMessageIgnoresTheLookalikes(t *testing.T) {
	for _, msg := range []string{
		"prompt is too long",
		"This model's maximum context length is 128000 tokens",
		"Please reduce the length of the messages",
		"too many tokens",
	} {
		if !IsContextExceededMessage(msg) {
			t.Errorf("IsContextExceededMessage(%q) = false; want a context overflow", msg)
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
		if IsContextExceededMessage(msg) {
			t.Errorf("IsContextExceededMessage(%q) = true; want it left to the status", msg)
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
