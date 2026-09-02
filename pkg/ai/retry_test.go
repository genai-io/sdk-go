package ai

import (
	"context"
	"iter"
	"testing"
	"time"
)

// retried runs one client through Retry and reports how many attempts the
// driver saw, and when each started.
func retried(t *testing.T, attempts int, backoff time.Duration, scripts ...script) (*scripted, []time.Time, error) {
	t.Helper()
	d := &scripted{scripts: scripts}
	var starts []time.Time
	clock := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) iter.Seq2[Delta, error] {
			starts = append(starts, time.Now())
			return next(ctx, req)
		}
	}
	c := NewClientWithDriver(Wrap(d, Retry(attempts, backoff), clock), stubModel())
	_, err := c.Complete(context.Background(), []Message{UserMessage("hi")})
	return d, starts, err
}

func overloaded() error { return &Error{Kind: KindOverloaded, Status: 529, Message: "overloaded"} }

// Retry's whole job is knowing what may be replayed: getting it wrong either
// duplicates an answer or abandons a call that would have succeeded.
func TestRetryReplaysOnlyWhatItMay(t *testing.T) {
	ok := script{deltas: []Delta{{Block: TextBlock("ok")}}}

	for name, tc := range map[string]struct {
		attempts int
		scripts  []script
		want     int
		wantErr  bool
	}{
		"a transient failure is replayed until it succeeds": {
			attempts: 3,
			scripts:  []script{{err: overloaded()}, {err: overloaded()}, ok},
			want:     3,
		},
		"the attempt budget is a total, not a number of replays": {
			attempts: 2,
			scripts:  []script{{err: overloaded()}},
			want:     2, wantErr: true,
		},
		"a fatal failure is not replayed": {
			attempts: 3,
			scripts:  []script{{err: &Error{Kind: KindAuth, Status: 401, Message: "bad key"}}},
			want:     1, wantErr: true,
		},
		"a failure after content is not replayed": {
			attempts: 3,
			scripts:  []script{{deltas: []Delta{{Block: TextBlock("partial")}}, err: overloaded()}},
			want:     1, wantErr: true,
		},
		"a failure after a metadata-only delta is replayed": {
			attempts: 3,
			scripts: []script{{
				// What Anthropic's message_start carries, and nothing else: no
				// block, so the caller has seen nothing to duplicate.
				deltas: []Delta{{Model: "claude", ID: "msg_1", Usage: &Usage{Input: 12}}},
				err:    overloaded(),
			}, ok},
			want: 2,
		},
		"no attempts at all still tries once": {
			attempts: 0,
			scripts:  []script{{err: overloaded()}},
			want:     1, wantErr: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			d, _, err := retried(t, tc.attempts, time.Millisecond, tc.scripts...)
			if d.calls != tc.want {
				t.Errorf("the driver saw %d attempts, want %d", d.calls, tc.want)
			}
			if (err != nil) != tc.wantErr {
				t.Errorf("err = %v, want an error: %v", err, tc.wantErr)
			}
		})
	}
}

// Backing off is the point of waiting at all: a server that is overloaded now
// is more likely to still be overloaded in the same interval than in twice it.
func TestRetryDoublesTheWaitBetweenAttempts(t *testing.T) {
	const backoff = 20 * time.Millisecond
	_, starts, err := retried(t, 3, backoff, script{err: overloaded()})
	if err == nil {
		t.Fatal("the call succeeded; it was scripted to fail every time")
	}
	if len(starts) != 3 {
		t.Fatalf("attempts = %d, want 3", len(starts))
	}
	first, second := starts[1].Sub(starts[0]), starts[2].Sub(starts[1])
	if first < backoff {
		t.Errorf("the first pause was %v, want at least %v", first, backoff)
	}
	if second < 2*backoff {
		t.Errorf("the second pause was %v, want at least %v — the backoff did not double", second, 2*backoff)
	}
}

// A provider that says when to come back knows better than any local schedule,
// in both directions: the hint is used even when it is far shorter.
func TestRetryPrefersTheProvidersOwnHint(t *testing.T) {
	limited := &Error{Kind: KindRateLimit, Status: 429, RetryAfter: 10 * time.Millisecond}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, err := retried(t, 2, time.Hour,
			script{err: limited}, script{deltas: []Delta{{Block: TextBlock("ok")}}})
		if err != nil {
			t.Errorf("Complete: %v", err)
		}
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Retry slept out its own hour instead of the provider's hint")
	}
}

// A cancel noticed while waiting out the backoff is the same failure as one a
// driver reports; callers switch on the kind, not on where it came from.
func TestRetryClassifiesACancelDuringTheBackoff(t *testing.T) {
	d := &scripted{scripts: []script{{err: overloaded()}}}
	c := NewClientWithDriver(Wrap(d, Retry(5, time.Hour)), stubModel())

	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(20 * time.Millisecond); cancel() }()

	errc := make(chan error, 1)
	go func() {
		_, err := c.Complete(ctx, []Message{UserMessage("hi")})
		errc <- err
	}()
	select {
	case err := <-errc:
		if !IsKind(err, KindCanceled) {
			t.Errorf("err = %v, want it classified as %q", err, KindCanceled)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Complete slept through the backoff instead of honouring the cancel")
	}
	if d.calls != 1 {
		t.Errorf("the driver saw %d attempts, want the one before the cancel", d.calls)
	}
}
