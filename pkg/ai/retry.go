package ai

import (
	"context"
	"iter"
	"time"
)

// Retry replays a failed call, at most attempts times in total, waiting
// backoff before the second try and doubling it after each further failure. A
// provider's own Retry-After hint replaces that pause where it sent one, and
// attempts of one or less is a single try.
//
// A call is replayed only while the caller has seen no content block: an
// Anthropic stream opens with a message_start delta of model, ID and input
// tokens, and treating that as output would leave every 529 unretried.
//
// It is the one execution policy this package ships, because it is the one
// whose rule is dangerous to get wrong.
//
//	client := ai.NewClientWithDriver(ai.Wrap(driver, ai.Retry(3, time.Second)), model)
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
						// Only a block is output the caller could have acted
						// on; usage, an ID or a stop reason is bookkeeping.
						if delta.Block.Type != "" {
							started = true
						}
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
						// Classified, so a cancel caught in the backoff is the
						// same kind of error as one a driver reports.
						yield(Delta{}, &Error{Kind: KindCanceled, Err: ctx.Err()})
						return
					case <-timer.C:
					}
					wait *= 2
				}
			}
		}
	}
}
