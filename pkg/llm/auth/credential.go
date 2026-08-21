package auth

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Credential is what was obtained for a vendor, and what has to survive
// between runs so a person signs in once rather than every time.
type Credential struct {
	// Vendor is the catalog vendor this belongs to.
	Vendor string `json:"vendor"`
	// Access is the token to present, or the API key for a key-based vendor.
	Access string `json:"access"`
	// Refresh renews Access without another sign-in. Empty for a token that
	// does not expire, and for an API key.
	Refresh string `json:"refresh,omitempty"`
	// ExpiresAt is when Access stops working. Zero means it does not.
	ExpiresAt time.Time `json:"expires_at,omitempty"`
	// Endpoint is the base URL the sign-in resolved to, for a vendor that
	// tells you where to talk to it only after you have authenticated.
	Endpoint string `json:"endpoint,omitempty"`
}

// Expired reports whether the credential needs renewing.
func (c Credential) Expired() bool {
	return !c.ExpiresAt.IsZero() && time.Now().After(c.ExpiresAt)
}

// Store keeps credentials between runs.
//
// It is an interface because where a credential belongs is the host
// application's decision, not this package's: a CLI wants a file, a server
// wants its own secret manager, and a test wants neither.
type Store interface {
	Load(vendor string) (Credential, bool, error)
	Save(c Credential) error
	Delete(vendor string) error
	List() ([]string, error)
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

// FileStore keeps credentials in one JSON file.
//
// This is the only thing in this SDK that writes to disk, and it writes
// secrets, so it does so carefully: the directory is created 0700, the file
// 0600, and a save is written to a temporary file and renamed, so an
// interrupted write cannot leave a truncated file where working credentials
// were.
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

// Path is where this store reads and writes.
func (s *FileStore) Path() string { return s.path }

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

func (s *FileStore) Save(c Credential) error {
	if c.Vendor == "" {
		return fmt.Errorf("auth: credential has no vendor")
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
	// Write-and-rename, so an interrupted save cannot leave a truncated file
	// where working credentials were.
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".credentials-*")
	if err != nil {
		return fmt.Errorf("auth: writing %s: %w", s.path, err)
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), s.path)
}

// MemoryStore keeps credentials for the life of the process. It is what a test
// or a server that manages its own secrets uses.
type MemoryStore struct {
	mu sync.Mutex
	m  map[string]Credential
}

func NewMemoryStore() *MemoryStore { return &MemoryStore{m: map[string]Credential{}} }

func (s *MemoryStore) Load(vendor string) (Credential, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.m[vendor]
	return c, ok, nil
}

func (s *MemoryStore) Save(c Credential) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[c.Vendor] = c
	return nil
}

func (s *MemoryStore) Delete(vendor string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, vendor)
	return nil
}

func (s *MemoryStore) List() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.m))
	for vendor := range s.m {
		out = append(out, vendor)
	}
	return out, nil
}
