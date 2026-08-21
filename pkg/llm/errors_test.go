package llm_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
)

func TestClassifyStatus(t *testing.T) {
	tests := []struct {
		status int
		want   llm.ErrorKind
	}{
		{http.StatusUnauthorized, llm.KindAuth},
		{http.StatusForbidden, llm.KindAuth},
		{http.StatusBadRequest, llm.KindInvalidRequest},
		{http.StatusNotFound, llm.KindInvalidRequest},
		{http.StatusTooManyRequests, llm.KindRateLimit},
		{http.StatusRequestTimeout, llm.KindOverloaded},
		{http.StatusConflict, llm.KindOverloaded},
		{http.StatusInternalServerError, llm.KindOverloaded},
		{529, llm.KindOverloaded}, // Anthropic "overloaded"
	}
	for _, tc := range tests {
		got, _ := llm.ClassifyStatus(tc.status, nil)
		if got != tc.want {
			t.Errorf("classifyStatus(%d) = %q, want %q", tc.status, got, tc.want)
		}
	}
}

func TestClassifyReadsRetryAfter(t *testing.T) {
	resp := &http.Response{Header: http.Header{"Retry-After": []string{"30"}}}
	err := llm.Classify("test", http.StatusTooManyRequests, resp, "", "slow down", nil)
	if err.Kind != llm.KindRateLimit {
		t.Fatalf("Kind = %q", err.Kind)
	}
	if got := llm.RetryAfter(err); got != 30*time.Second {
		t.Errorf("RetryAfter = %v, want 30s", got)
	}
	if !llm.IsRetryable(err) {
		t.Error("a rate limit should be retryable")
	}
}

func TestContextExceededBeatsBadRequest(t *testing.T) {
	// The condition arrives as an ordinary 400; filing it as an invalid
	// request would tell the caller to give up instead of compacting.
	err := llm.Classify("test", http.StatusBadRequest, nil, "",
		"This model's maximum context length is 200000 tokens", nil)
	if !llm.IsContextExceeded(err) {
		t.Fatalf("Kind = %q, want %q", err.Kind, llm.KindContextExceeded)
	}
	if llm.IsRetryable(err) {
		t.Error("retrying the same oversized prompt cannot help")
	}
}

func TestCancellationIsNeverRetryable(t *testing.T) {
	err := llm.Classify("test", 0, nil, "", "", fmt.Errorf("send: %w", context.Canceled))
	if err.Kind != llm.KindCanceled {
		t.Fatalf("Kind = %q, want %q", err.Kind, llm.KindCanceled)
	}
	if llm.IsRetryable(err) {
		t.Error("a user interrupt must stay fatal")
	}
}

func TestStreamErrorTreatsUnknownAsTransport(t *testing.T) {
	plain := errors.New("connection closed by peer")

	if got := llm.Classify("test", 0, nil, "", "", plain); got.Kind != llm.KindUnknown {
		t.Errorf("Classify Kind = %q, want %q", got.Kind, llm.KindUnknown)
	}
	// A streaming transport routinely loses its typed error, so an otherwise
	// unclassifiable terminal error gets the benefit of the doubt.
	streamed := llm.StreamError("test", 0, nil, "", "", plain)
	if streamed.Kind != llm.KindNetwork || !llm.IsRetryable(streamed) {
		t.Errorf("StreamError Kind = %q, retryable = %v", streamed.Kind, llm.IsRetryable(streamed))
	}
}

func TestClassifyPreservesAlreadyTypedError(t *testing.T) {
	original := &llm.Error{Driver: "d", Kind: llm.KindAuth, Message: "bad key"}
	got := llm.Classify("other", http.StatusInternalServerError, nil, "", "", fmt.Errorf("wrapped: %w", original))
	if got != original {
		t.Fatalf("a driver's own classification was overwritten: %+v", got)
	}
}

func TestErrorUnwraps(t *testing.T) {
	inner := errors.New("inner")
	err := llm.Classify("test", 500, nil, "", "", inner)
	if !errors.Is(err, inner) {
		t.Error("the underlying error should stay reachable")
	}
}

// Bedrock formats throttling as "ThrottlingException: Too many tokens, please
// wait before trying again." — which matches the "too many tokens" overflow
// signature. Classifying it as a context overflow tells the caller to compact
// a prompt that was never too long and suppresses the retry that would have
// worked, so the exclusion is checked before the signatures.
func TestThrottlingIsNotContextExceeded(t *testing.T) {
	throttles := []string{
		"ThrottlingException: Too many tokens, please wait before trying again.",
		"Rate limit reached: too many tokens in this window",
		"429 Too Many Requests",
		"quota exceeded, too many tokens",
	}
	for _, msg := range throttles {
		if kind, ok := llm.ClassifyMessage(msg); ok {
			t.Errorf("ClassifyMessage(%q) = %q, want no overflow match", msg, kind)
		}
	}

	// A genuine overflow using the same phrase still classifies.
	if _, ok := llm.ClassifyMessage("Request rejected: too many tokens for this model's window"); !ok {
		t.Error("a genuine overflow was excluded")
	}
}

func TestThrottlingStaysRetryable(t *testing.T) {
	err := llm.Classify("bedrock", 429, nil, "ThrottlingException",
		"ThrottlingException: Too many tokens, please wait before trying again.", nil)
	if llm.IsContextExceeded(err) {
		t.Fatalf("throttling classified as %q", err.Kind)
	}
	if !llm.IsRetryable(err) {
		t.Errorf("err = %v, want retryable", err)
	}
}
