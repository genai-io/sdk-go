package session

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

func lockFile(f *os.File) error {
	err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &windows.Overlapped{})
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return ErrBusy
	}
	return err
}

// unlock removes the file once let go. Windows cannot remove a file another
// process has open, so the removal fails, harmlessly, while one is about to
// lock it.
func (l *fileLock) unlock() error {
	err := l.f.Close()
	_ = os.Remove(l.f.Name())
	return err
}
