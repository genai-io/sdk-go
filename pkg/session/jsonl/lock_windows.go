package jsonl

import (
	"os"

	"golang.org/x/sys/windows"
)

// tryLock takes an exclusive lock on f, failing at once if another holds it.
func tryLock(f *os.File) error {
	return windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &windows.Overlapped{})
}
