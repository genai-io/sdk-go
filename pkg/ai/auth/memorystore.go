package auth

import "sync"

// MemoryStore keeps credentials for the life of the process. It is what a test
// or a server that manages its own secrets uses.
type MemoryStore struct {
	mu sync.Mutex
	m  map[string]Credential
}

func NewMemoryStore() *MemoryStore { return &MemoryStore{m: map[string]Credential{}} }

// Load returns the credential held for a vendor.
func (s *MemoryStore) Load(vendor string) (Credential, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.m[vendor]
	return c, ok, nil
}

// Save keeps a credential for the life of the process.
func (s *MemoryStore) Save(c Credential) error {
	if c.Vendor == "" {
		return errNoVendor()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[c.Vendor] = c
	return nil
}

// Delete forgets a vendor's credential.
func (s *MemoryStore) Delete(vendor string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, vendor)
	return nil
}

// List names the vendors this store holds a credential for.
func (s *MemoryStore) List() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.m))
	for vendor := range s.m {
		out = append(out, vendor)
	}
	return out, nil
}
