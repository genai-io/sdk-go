package session

import (
	"errors"
	"io/fs"
	"os"
)

// fileLock is a session's writer lease: an exclusive lock on <session>.lock,
// held by an open file, so that it is let go when the process dies too.
type fileLock struct{ f *os.File }

func lock(session string) (*fileLock, error) {
	path := session + ".lock"
	for {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
		if err != nil {
			return nil, err
		}
		if err := lockFile(f); err != nil {
			return nil, errors.Join(err, f.Close())
		}
		// A holder removes the file as it lets go, so the one opened may be
		// gone by the time it is locked, and locking it locked nothing.
		held, err := f.Stat()
		if err != nil {
			return nil, errors.Join(err, f.Close())
		}
		now, err := os.Stat(path)
		if err == nil && os.SameFile(held, now) {
			return &fileLock{f}, nil
		}
		if err := errors.Join(f.Close(), ignore(err, fs.ErrNotExist)); err != nil {
			return nil, err
		}
	}
}

func ignore(err, target error) error {
	if errors.Is(err, target) {
		return nil
	}
	return err
}
