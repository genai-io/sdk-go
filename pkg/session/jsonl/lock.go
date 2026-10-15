package jsonl

import (
	"context"
	"os"
	"time"
)

// lockFile takes an exclusive lock on path. With wait it retries until the
// lock is free or ctx ends; without, a held lock is an error.
func lockFile(ctx context.Context, path string, wait bool) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	for {
		busy, err := tryLock(f)
		if err == nil {
			return f, nil
		}
		if !wait || !busy {
			_ = f.Close()
			return nil, err
		}
		select {
		case <-ctx.Done():
			_ = f.Close()
			return nil, ctx.Err()
		case <-time.After(10 * time.Millisecond):
		}
	}
}
