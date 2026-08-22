package ai

import (
	"context"
	"iter"
	"time"
)

// Retry is the one execution policy this package ships, because it is the one
// with a rule that is dangerous to get wrong.
//
// Every driver disables its vendor SDK's own retry, so without middleware a
// caller gets none at all. Writing one is easy to get subtly wrong in a way
// that shows up as duplicated or vanished output rather than as an error, so
// the correct version lives here. Everything else — caching, logging, cost
// metering — stays the caller's, because only the application knows what may
// be cached and what must not be logged.

// Retry replays a failed call, at most attempts times in total, waiting
// backoff before the second try and doubling it after each further failure.
//
//	client := ai.New(ai.Wrap(driver, ai.Retry(3, time.Second)), model)
//
// Three conditions all have to hold before it replays:
//
//   - the failure is classified retryable, so a bad key or an impossible
//     request fails once rather than three times;
//   - nothing has been yielded yet. Once a delta has reached the caller the
//     answer has begun, and resending would either duplicate the text already
//     shown or discard it. This is the rule callers should not have to
//     discover, and it is why a mid-stream failure is returned as-is;
//   - the context is still live. A canceled context ends the call with its own
//     error rather than sleeping first.
//
// A provider that says how long to wait wins over the backoff: Retry-After on
// a 429 is the server telling you when it will answer, and guessing shorter
// wastes the attempt.
//
// attempts counts tries, not retries: Retry(1, …) never replays, and
// Retry(0, …) is treated the same way rather than looping forever.
func Retry(attempts int, backoff time.Duration) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) iter.Seq2[Delta, error] {
			return func(yield func(Delta, error) bool) {
				wait := backoff
				for attempt := 1; ; attempt++ {
					started, failure := false, error(nil)
					for delta, err := range next(ctx, req) {
						if err != nil {
							failure = err
							break
						}
						started = true
						if !yield(delta, nil) {
							return
						}
					}
					switch {
					case failure == nil:
						return
					case started, attempt >= attempts, !IsRetryable(failure):
						yield(Delta{}, failure)
						return
					}

					pause := wait
					if after := RetryAfter(failure); after > 0 {
						pause = after
					}
					timer := time.NewTimer(pause)
					select {
					case <-ctx.Done():
						timer.Stop()
						yield(Delta{}, ctx.Err())
						return
					case <-timer.C:
					}
					wait *= 2
				}
			}
		}
	}
}
