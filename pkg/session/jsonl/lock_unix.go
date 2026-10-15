//go:build !windows

package jsonl

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

// tryLock reports busy when another holder has the lock.
func tryLock(f *os.File) (busy bool, err error) {
	err = unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	return errors.Is(err, unix.EWOULDBLOCK), err
}
