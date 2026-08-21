package llm

import (
	"context"
	"iter"
	"time"
)

// Handler runs one model call: the seam every middleware wraps.
//
// It is the same shape as Driver.Generate, so a driver is already a Handler
// and a middleware chain collapses back to one.
type Handler func(ctx context.Context, p *Prompt, opts Options) iter.Seq2[Delta, error]

// Middleware wraps a Handler.
//
// This is where retry, caching, request logging, cost metering and prompt
// rewriting belong. Putting the seam here rather than implementing those in
// the SDK keeps policy with the caller, who is the only one who knows the
// budget for a turn, what may be cached, and what must not be logged.
//
// Middleware runs outermost-first: the first one given sees the request first
// and the deltas last.
type Middleware func(Handler) Handler

// chain wraps h with mw, outermost first.
func chain(h Handler, mw []Middleware) Handler {
	for i := len(mw) - 1; i >= 0; i-- {
		h = mw[i](h)
	}
	return h
}

// WithMiddleware adds middleware to every call this client makes.
func WithMiddleware(mw ...Middleware) Option {
	return func(c *Client) { c.middleware = append(c.middleware, mw...) }
}

// RetryPolicy configures Retry.
type RetryPolicy struct {
	// Attempts is the total number of tries, including the first. Values below
	// two disable retrying.
	Attempts int
	// Backoff returns how long to wait before attempt n (1 for the first
	// retry). Nil means an exponential 1s, 2s, 4s… capped at MaxDelay.
	Backoff func(attempt int) time.Duration
	// MaxDelay caps any wait, including one the provider asked for in a
	// Retry-After header. A provider that asks for twenty minutes is telling
	// you to come back later, not to block a request goroutine for twenty
	// minutes; exceeding this fails immediately so the caller can decide.
	// Zero means one minute.
	MaxDelay time.Duration
	// Retryable decides whether an error is worth another try. Nil means
	// IsRetryable — rate limits, overloads and transport failures.
	Retryable func(error) bool
}

// Retry retries a call that failed *before producing any output*.
//
// That restriction is the whole design. Once a delta has reached the caller,
// the answer has begun; resending would either duplicate the text already
// shown or silently discard it, and neither is something a middleware may
// decide on the caller's behalf. A stream that dies mid-answer is therefore
// reported, not retried — recovering from that needs the conversation, which
// only the layer above has.
//
// The provider's own Retry-After hint is honoured when it fits inside
// MaxDelay.
func Retry(policy RetryPolicy) Middleware {
	attempts := policy.Attempts
	if attempts < 2 {
		attempts = 1
	}
	maxDelay := policy.MaxDelay
	if maxDelay <= 0 {
		maxDelay = time.Minute
	}
	retryable := policy.Retryable
	if retryable == nil {
		retryable = IsRetryable
	}
	backoff := policy.Backoff
	if backoff == nil {
		backoff = func(attempt int) time.Duration {
			d := time.Second << (attempt - 1)
			return min(d, maxDelay)
		}
	}

	return func(next Handler) Handler {
		return func(ctx context.Context, p *Prompt, opts Options) iter.Seq2[Delta, error] {
			return func(yield func(Delta, error) bool) {
				for attempt := 1; ; attempt++ {
					produced := false
					var failure error

					for delta, err := range next(ctx, p, opts) {
						if err != nil {
							failure = err
							break
						}
						produced = true
						if !yield(delta, nil) {
							return
						}
					}

					switch {
					case failure == nil:
						return
					// Output already reached the caller: the answer has begun,
					// and no middleware may decide to replay or discard it.
					case produced,
						attempt >= attempts,
						!retryable(failure),
						ctx.Err() != nil:
						yield(Delta{}, failure)
						return
					}

					delay := backoff(attempt)
					if hinted := RetryAfter(failure); hinted > 0 {
						delay = hinted
					}
					if delay > maxDelay {
						yield(Delta{}, failure)
						return
					}
					select {
					case <-time.After(delay):
					case <-ctx.Done():
						yield(Delta{}, failure)
						return
					}
				}
			}
		}
	}
}

// SimulateSystemPrompt folds a system prompt into an opening exchange for a
// model that has no system role, instead of failing validation.
//
// It is opt-in because the substitution is lossy: a folded prompt is ordinary
// conversation the model may argue with or forget, where a real system prompt
// is weighted and cached differently. A caller who wants that trade makes it
// explicitly; one who does not gets a clear error instead of a quietly weaker
// prompt.
func SimulateSystemPrompt() Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, p *Prompt, opts Options) iter.Seq2[Delta, error] {
			return next(ctx, foldSystemPrompt(p), opts)
		}
	}
}
