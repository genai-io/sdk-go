package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// fileStore keeps credentials in one JSON file, readable only by its owner.
//
// What it holds cannot be recovered without signing in again, so the file is
// created 0600 and replaced by writing a temporary file and renaming it: a torn
// write would lose every vendor's credential, not just the one being saved.
type fileStore struct {
	path string
	mu   sync.Mutex
}

var fileStores sync.Map // absolute path -> *fileStore; shares writes and refreshes

// newFileStore returns a store backed by path. An empty path means
// genai-io/credentials.json under XDG_CONFIG_HOME, or ~/.config.
func newFileStore(path string) (*fileStore, error) {
	if path == "" {
		var err error
		if path, err = defaultStorePath(); err != nil {
			return nil, err
		}
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	store, _ := fileStores.LoadOrStore(path, &fileStore{path: path})
	return store.(*fileStore), nil
}

// defaultStorePath is where fileStore writes when given no path: under
// XDG_CONFIG_HOME if it is set, otherwise ~/.config.
func defaultStorePath() (string, error) {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("auth: no home directory to store credentials in: %w", err)
		}
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "genai-io", "credentials.json"), nil
}

// Load returns the credential held for a vendor.
func (s *fileStore) Load(vendor string) (Credential, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	all, err := s.read()
	if err != nil {
		return Credential{}, false, err
	}
	c, ok := all[vendor]
	return c, ok, nil
}

// Save keeps a credential, replacing any held for the same vendor.
func (s *fileStore) Save(c Credential) error {
	if c.Vendor == "" {
		return errNoVendor
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return withFileLock(context.Background(), s.path+".lock", func() error {
		all, err := s.read()
		if err != nil {
			return err
		}
		all[c.Vendor] = c
		return s.write(all)
	})
}

// Delete forgets a vendor's credential and rewrites the file.
func (s *fileStore) Delete(vendor string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return withFileLock(context.Background(), s.path+".lock", func() error {
		all, err := s.read()
		if err != nil {
			return err
		}
		if _, ok := all[vendor]; !ok {
			return nil
		}
		delete(all, vendor)
		return s.write(all)
	})
}

func (s *fileStore) WithRefreshLock(ctx context.Context, _ string, fn func() error) error {
	return withFileLock(ctx, s.path+".refresh.lock", fn)
}

func withFileLock(ctx context.Context, path string, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		locked, err := tryFileLock(f)
		if err != nil {
			return fmt.Errorf("auth: lock %s: %w", path, err)
		}
		if locked {
			return fn()
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func (s *fileStore) read() (map[string]Credential, error) {
	raw, err := os.ReadFile(s.path)
	if os.IsNotExist(err) {
		return map[string]Credential{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("auth: reading %s: %w", s.path, err)
	}
	if len(raw) == 0 {
		return map[string]Credential{}, nil
	}
	var all map[string]Credential
	if err := json.Unmarshal(raw, &all); err != nil {
		return nil, fmt.Errorf("auth: %s is not readable as credentials: %w", s.path, err)
	}
	if all == nil {
		all = map[string]Credential{}
	}
	return all, nil
}

func (s *fileStore) write(all map[string]Credential) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return fmt.Errorf("auth: creating %s: %w", filepath.Dir(s.path), err)
	}
	raw, err := json.MarshalIndent(all, "", "  ")
	if err != nil {
		return err
	}
	// Write-and-rename: the rename is atomic, so a crash mid-write leaves the
	// previous file whole rather than half of a new one.
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".credentials-*")
	if err != nil {
		return fmt.Errorf("auth: writing %s: %w", s.path, err)
	}
	// Removing the temporary file is cleanup after either outcome: on success
	// it has already been renamed away and there is nothing left to remove.
	defer func() { _ = os.Remove(tmp.Name()) }()
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(raw); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), s.path)
}
