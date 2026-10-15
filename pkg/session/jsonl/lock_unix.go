//go:build !windows

package jsonl

import (
	"os"

	"golang.org/x/sys/unix"
)

// tryLock takes an exclusive lock on f, failing at once if another holds it.
func tryLock(f *os.File) error {
	return unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
}
