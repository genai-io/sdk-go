package auth

import (
	"maps"
	"slices"
	"sync"
)

// memStore keeps credentials in memory: the second Store, which these tests
// swap in for the file one. The zero value is ready to use.
type memStore struct {
	mu sync.Mutex
	m  map[string]Credential
}

func (s *memStore) Load(vendor string) (Credential, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.m[vendor]
	return c, ok, nil
}

func (s *memStore) Save(c Credential) error {
	if c.Vendor == "" {
		return errNoVendor()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = map[string]Credential{}
	}
	s.m[c.Vendor] = c
	return nil
}

func (s *memStore) Delete(vendor string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, vendor)
	return nil
}

func (s *memStore) List() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Collect(maps.Keys(s.m)), nil
}
