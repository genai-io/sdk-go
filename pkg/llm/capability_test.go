package llm_test

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm"
	"github.com/genai-io/sdk-go/pkg/llm/llmtest"
)

func textOnlyModel() llm.Model {
	return llm.Model{ID: "text-only", Input: []llm.Modality{llm.ModalityText}}
}

// The point of validating locally is a sentence the caller can act on, and
// nothing spent on a call that was never going to work.
func TestValidateCatchesUnsupportedRequests(t *testing.T) {
	img := llm.Image{MediaType: "image/png", Data: "AAAA"}

	tests := []struct {
		name   string
		model  llm.Model
		prompt *llm.Prompt
		opts   llm.Options
		want   string
	}{
		{
			name:   "images to a text-only model",
			model:  textOnlyModel(),
			prompt: &llm.Prompt{Messages: []llm.Message{llm.User("look", img)}},
			want:   "image",
		},
		{
			name:   "tools to a model without them",
			model:  llm.Model{ID: "m", Unsupported: llm.Unsupported{Tools: true}},
			prompt: &llm.Prompt{Tools: []llm.Tool{{Name: "ls"}}},
			want:   "tools",
		},
		{
			name:   "forcing a tool where choice is unsupported",
			model:  llm.Model{ID: "m", Unsupported: llm.Unsupported{ToolChoice: true}},
			prompt: &llm.Prompt{Tools: []llm.Tool{{Name: "ls"}}},
			opts:   llm.Options{ForceTool: "ls"},
			want:   "tool is called",
		},
		{
			name:   "a system prompt to a model with no system role",
			model:  llm.Model{ID: "m", Unsupported: llm.Unsupported{System: true}},
			prompt: &llm.Prompt{System: "be brief"},
			want:   "system prompt",
		},
		{
			name:   "history to a single-turn model",
			model:  llm.Model{ID: "m", Unsupported: llm.Unsupported{Multiturn: true}},
			prompt: &llm.Prompt{Messages: []llm.Message{llm.User("a"), llm.Assistant("b"), llm.User("c")}},
			want:   "single message",
		},
		{
			name:   "a retired model",
			model:  llm.Model{ID: "old", Stage: llm.StageRetired, Replacement: "new"},
			prompt: &llm.Prompt{},
			want:   "retired",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.model.Validate(tc.prompt, tc.opts)
			if err == nil {
				t.Fatal("expected the request to be refused")
			}
			if !llm.IsUnsupported(err) {
				t.Errorf("err = %v, want KindUnsupported", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want it to mention %q", err, tc.want)
			}
			// Nothing about this is worth retrying or reauthenticating.
			if llm.IsRetryable(err) || llm.IsAuth(err) {
				t.Errorf("err = %v was misclassified", err)
			}
		})
	}
}

// The zero value is a fully capable model, so a listing that carried nothing
// but an ID is not treated as crippled.
func TestBareModelIsAssumedCapable(t *testing.T) {
	m := llm.Model{ID: "unknown-from-a-listing"}
	prompt := &llm.Prompt{
		System:   "be brief",
		Messages: []llm.Message{llm.User("a"), llm.Assistant("b"), llm.User("c")},
		Tools:    []llm.Tool{{Name: "ls"}},
	}
	if err := m.Validate(prompt, llm.Options{ForceTool: "ls"}); err != nil {
		t.Errorf("a bare model should be assumed capable: %v", err)
	}
	// It is still text-only until it says otherwise, which is the one
	// conservative default: guessing vision wrong wastes a request.
	img := llm.Image{MediaType: "image/png", Data: "AAAA"}
	if err := m.Validate(&llm.Prompt{Messages: []llm.Message{llm.User("look", img)}}, llm.Options{}); err == nil {
		t.Error("images should need to be declared")
	}
}

func TestRetiredModelIsRefusedByTheClient(t *testing.T) {
	drv := llmtest.Text("never sent")
	model := llmtest.Model
	model.Stage = llm.StageRetired
	model.Replacement = "test-model-2"

	_, err := llm.New(drv, model).Complete(context.Background(), &llm.Prompt{}, nil)
	if !llm.IsUnsupported(err) {
		t.Fatalf("err = %v, want the model refused", err)
	}
	if !strings.Contains(err.Error(), "test-model-2") {
		t.Errorf("err = %v, want it to name the replacement", err)
	}
	if drv.CallCount() != 0 {
		t.Error("the driver was called for a retired model")
	}
}

// Folding is a fallback, not an equivalent — so it is opt-in, and a caller who
// does not opt in gets a clear error rather than a quietly weaker prompt.
func TestSimulateSystemPromptFolds(t *testing.T) {
	model := llmtest.Model
	model.Unsupported.System = true

	drv := llmtest.Text("ok")
	if _, err := llm.New(drv, model).Complete(context.Background(),
		&llm.Prompt{System: "be brief", Messages: []llm.Message{llm.User("hi")}}, nil); err == nil {
		t.Fatal("a system prompt should be refused without the middleware")
	}

	drv = llmtest.Text("ok")
	client := llm.New(drv, model, llm.WithMiddleware(llm.SimulateSystemPrompt()))
	if _, err := client.Complete(context.Background(),
		&llm.Prompt{System: "be brief", Messages: []llm.Message{llm.User("hi")}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	sent := drv.Last().Prompt
	if sent.System != "" {
		t.Errorf("System = %q, want it folded away", sent.System)
	}
	if len(sent.Messages) != 3 {
		t.Fatalf("messages = %d, want the folded pair plus the original", len(sent.Messages))
	}
	if !strings.Contains(sent.Messages[0].Text(), "be brief") {
		t.Errorf("first message = %q", sent.Messages[0].Text())
	}
	if sent.Messages[1].Role != llm.RoleAssistant {
		t.Error("the acknowledgement should be attributed to the model")
	}
}

func TestAvailableFiltersRetired(t *testing.T) {
	models := []llm.Model{
		{ID: "live"},
		{ID: "dead", Stage: llm.StageRetired},
		{ID: "beta", Stage: llm.StagePreview},
	}
	got := llm.Available(models)
	if len(got) != 2 {
		t.Fatalf("got %d models, want the two that still serve", len(got))
	}
	for _, m := range got {
		if m.ID == "dead" {
			t.Error("a retired model survived the filter")
		}
	}
}

// ─── middleware ───

func TestMiddlewareRunsOutermostFirst(t *testing.T) {
	var order []string
	mark := func(name string) llm.Middleware {
		return func(next llm.Handler) llm.Handler {
			return func(ctx context.Context, p *llm.Prompt, opts llm.Options) iter.Seq2[llm.Delta, error] {
				order = append(order, name)
				return next(ctx, p, opts)
			}
		}
	}

	client := llm.New(llmtest.Text("ok"), llmtest.Model, llm.WithMiddleware(mark("first"), mark("second")))
	if _, err := client.Complete(context.Background(), &llm.Prompt{}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Errorf("order = %v, want the first-given middleware to see the request first", order)
	}
}

// A retryable failure before any output is safe to replay.
func TestRetryReplaysAFailureBeforeOutput(t *testing.T) {
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Err: llm.Classify("test", 503, nil, "", "overloaded", nil)},
		{Deltas: []llm.Delta{{Text: "second try"}, {StopReason: llm.StopEndTurn}}},
	}}
	client := llm.New(drv, llmtest.Model, llm.WithMiddleware(llm.Retry(llm.RetryPolicy{
		Attempts: 3,
		Backoff:  func(int) time.Duration { return 0 },
	})))

	resp, err := client.Complete(context.Background(), &llm.Prompt{}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "second try" {
		t.Errorf("Content = %q", resp.Content)
	}
	if drv.CallCount() != 2 {
		t.Errorf("calls = %d, want one retry", drv.CallCount())
	}
}

// Once a delta has reached the caller the answer has begun; replaying would
// duplicate what was already shown, and discarding it would lose it. Neither
// is a middleware's decision to make.
func TestRetryDoesNotReplayAfterOutput(t *testing.T) {
	boom := errors.New("died mid-answer")
	drv := &llmtest.Driver{Turns: []llmtest.Turn{
		{Deltas: []llm.Delta{{Text: "half an ans"}}, Err: llm.Classify("test", 503, nil, "", boom.Error(), boom)},
		{Deltas: []llm.Delta{{Text: "never reached"}}},
	}}
	client := llm.New(drv, llmtest.Model, llm.WithMiddleware(llm.Retry(llm.RetryPolicy{
		Attempts: 3,
		Backoff:  func(int) time.Duration { return 0 },
	})))

	resp, err := client.Complete(context.Background(), &llm.Prompt{}, nil)
	if err == nil {
		t.Fatal("expected the failure to be reported, not retried")
	}
	if drv.CallCount() != 1 {
		t.Errorf("calls = %d, want no retry after output began", drv.CallCount())
	}
	if resp == nil || resp.Content != "half an ans" {
		t.Errorf("resp = %+v, want the partial answer", resp)
	}
}

func TestRetryLeavesFatalErrorsAlone(t *testing.T) {
	drv := llmtest.Fail(llm.Classify("test", 401, nil, "", "bad key", nil))
	client := llm.New(drv, llmtest.Model, llm.WithMiddleware(llm.Retry(llm.RetryPolicy{
		Attempts: 5,
		Backoff:  func(int) time.Duration { return 0 },
	})))

	if _, err := client.Complete(context.Background(), &llm.Prompt{}, nil); !llm.IsAuth(err) {
		t.Fatalf("err = %v", err)
	}
	if drv.CallCount() != 1 {
		t.Errorf("calls = %d, want no retry on a bad credential", drv.CallCount())
	}
}

// A provider asking for a twenty-minute wait is telling you to come back
// later, not to block a request goroutine for twenty minutes.
func TestRetryRefusesAnExcessiveServerDelay(t *testing.T) {
	resp := &http.Response{Header: http.Header{"Retry-After": []string{"1200"}}}
	drv := llmtest.Fail(llm.Classify("test", 429, resp, "", "slow down", nil))
	client := llm.New(drv, llmtest.Model, llm.WithMiddleware(llm.Retry(llm.RetryPolicy{
		Attempts: 5,
		MaxDelay: time.Second,
	})))

	start := time.Now()
	if _, err := client.Complete(context.Background(), &llm.Prompt{}, nil); err == nil {
		t.Fatal("expected the failure to surface")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("blocked for %v, want an immediate failure", elapsed)
	}
	if drv.CallCount() != 1 {
		t.Errorf("calls = %d", drv.CallCount())
	}
}
