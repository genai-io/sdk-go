//go:build !windows

package session

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

func lockFile(f *os.File) error {
	err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return ErrBusy
	}
	return err
}

// unlock removes the file while still holding it, which lock checks for.
func (l *fileLock) unlock() error {
	return errors.Join(ignore(os.Remove(l.f.Name()), os.ErrNotExist), l.f.Close())
}
