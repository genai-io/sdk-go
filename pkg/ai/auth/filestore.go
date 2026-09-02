package auth

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FileStore keeps credentials in one JSON file, readable only by its owner.
//
// What it holds cannot be recovered without signing in again, so the file is
// created 0600 and replaced by writing a temporary file and renaming it: a torn
// write would lose every vendor's credential, not just the one being saved.
type FileStore struct {
	path string
	mu   sync.Mutex
}

// NewFileStore returns a store backed by path. An empty path uses
// DefaultStorePath.
func NewFileStore(path string) (*FileStore, error) {
	if path == "" {
		var err error
		if path, err = DefaultStorePath(); err != nil {
			return nil, err
		}
	}
	return &FileStore{path: path}, nil
}

// DefaultStorePath is where FileStore writes when given no path: under
// XDG_CONFIG_HOME if it is set, otherwise ~/.config.
func DefaultStorePath() (string, error) {
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

// Path is where this store reads and writes.
func (s *FileStore) Path() string { return s.path }

// Load returns the credential held for a vendor.
func (s *FileStore) Load(vendor string) (Credential, bool, error) {
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
func (s *FileStore) Save(c Credential) error {
	if c.Vendor == "" {
		return errNoVendor()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	all, err := s.read()
	if err != nil {
		return err
	}
	all[c.Vendor] = c
	return s.write(all)
}

// Delete forgets a vendor's credential and rewrites the file.
func (s *FileStore) Delete(vendor string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	all, err := s.read()
	if err != nil {
		return err
	}
	if _, ok := all[vendor]; !ok {
		return nil
	}
	delete(all, vendor)
	return s.write(all)
}

// List names the vendors this store holds a credential for.
func (s *FileStore) List() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	all, err := s.read()
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(all))
	for vendor := range all {
		out = append(out, vendor)
	}
	return out, nil
}

func (s *FileStore) read() (map[string]Credential, error) {
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

func (s *FileStore) write(all map[string]Credential) error {
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
