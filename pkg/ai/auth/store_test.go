package auth

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// TestStoresBehaveAlike runs both stores through the same script. The whole
// point of the Store interface is that a server can swap the file for memory
// and change nothing else, which is only true if they agree.
func TestStoresBehaveAlike(t *testing.T) {
	stores := map[string]func(t *testing.T) Store{
		"file":   func(t *testing.T) Store { return newTestFileStore(t) },
		"memory": func(t *testing.T) Store { return NewMemoryStore() },
	}

	for name, build := range stores {
		t.Run(name, func(t *testing.T) {
			s := build(t)

			// Nobody has signed in yet is the normal state, not an error.
			if _, found, err := s.Load("copilot"); err != nil || found {
				t.Errorf("Load on an empty store = %v, %v, want not-found and no error", found, err)
			}
			if names, err := s.List(); err != nil || len(names) != 0 {
				t.Errorf("List on an empty store = %v, %v", names, err)
			}
			// Deleting one that is not there succeeds.
			if err := s.Delete("copilot"); err != nil {
				t.Errorf("Delete of an absent credential = %v, want nil", err)
			}

			// A credential with no vendor could never be loaded back, so
			// accepting it would silently lose the sign-in it represents.
			if err := s.Save(Credential{Access: "orphan"}); err == nil {
				t.Error("Save accepted a credential with no vendor")
			}

			want := Credential{
				Vendor: "copilot", Access: "gho_1", Refresh: "r1",
				ExpiresAt: time.Now().Add(time.Hour).UTC().Truncate(time.Second),
				Endpoint:  "https://api.enterprise.githubcopilot.com",
			}
			if err := s.Save(want); err != nil {
				t.Fatalf("Save: %v", err)
			}
			got, found, err := s.Load("copilot")
			if err != nil || !found {
				t.Fatalf("Load = %v, %v", found, err)
			}
			if !got.ExpiresAt.Equal(want.ExpiresAt) {
				t.Errorf("ExpiresAt = %v, want %v", got.ExpiresAt, want.ExpiresAt)
			}
			got.ExpiresAt, want.ExpiresAt = time.Time{}, time.Time{}
			if got != want {
				t.Errorf("Load = %+v, want %+v", got, want)
			}

			// Saving the same vendor replaces rather than accumulates.
			if err := s.Save(Credential{Vendor: "copilot", Access: "gho_2"}); err != nil {
				t.Fatal(err)
			}
			if again, _, _ := s.Load("copilot"); again.Access != "gho_2" {
				t.Errorf("Access = %q, want the credential that replaced it", again.Access)
			}

			if err := s.Save(Credential{Vendor: "openai-codex", Access: "sk"}); err != nil {
				t.Fatal(err)
			}
			names, err := s.List()
			if err != nil {
				t.Fatal(err)
			}
			slices.Sort(names)
			if len(names) != 2 || names[0] != "copilot" || names[1] != "openai-codex" {
				t.Errorf("List = %v, want both vendors", names)
			}

			if err := s.Delete("copilot"); err != nil {
				t.Fatal(err)
			}
			if _, found, _ := s.Load("copilot"); found {
				t.Error("the deleted credential is still there")
			}
			if _, found, _ := s.Load("openai-codex"); !found {
				t.Error("deleting one vendor took another with it")
			}
		})
	}
}

// TestFileStoreIsReadableOnlyByItsOwner is the property that makes writing a
// credential to disk acceptable at all.
func TestFileStoreIsReadableOnlyByItsOwner(t *testing.T) {
	s := newTestFileStore(t)
	if err := s.Save(Credential{Vendor: "copilot", Access: "gho_1"}); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(s.Path())
	if err != nil {
		t.Fatalf("the store wrote nothing to %s: %v", s.Path(), err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("mode = %#o, want 0600: a credential file anyone can read is a credential anyone has", perm)
	}
	// Rewriting keeps the mode, which a plain WriteFile over an existing file
	// would not guarantee either.
	if err := s.Save(Credential{Vendor: "copilot", Access: "gho_2"}); err != nil {
		t.Fatal(err)
	}
	info, err = os.Stat(s.Path())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("mode after a rewrite = %#o, want 0600", perm)
	}
}

// TestFileStoreLeavesNoTemporaryFiles: the write goes through a temporary file
// and a rename, and a rename that left the temporary behind would scatter
// copies of the credential through the config directory.
func TestFileStoreLeavesNoTemporaryFiles(t *testing.T) {
	s := newTestFileStore(t)
	for _, access := range []string{"a", "b", "c"} {
		if err := s.Save(Credential{Vendor: "copilot", Access: access}); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(filepath.Dir(s.Path()))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != filepath.Base(s.Path()) {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("the directory holds %v, want only the credentials file", names)
	}
}

// TestFileStoreReportsAFileItCannotRead. Silently starting over would throw
// away every other vendor's sign-in the moment one byte went wrong.
func TestFileStoreReportsAFileItCannotRead(t *testing.T) {
	s := newTestFileStore(t)
	if err := os.MkdirAll(filepath.Dir(s.Path()), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(s.Path(), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, _, err := s.Load("copilot"); err == nil {
		t.Error("Load read a corrupt file as an empty one")
	}
	if err := s.Save(Credential{Vendor: "copilot", Access: "gho"}); err == nil {
		t.Error("Save overwrote a file it could not read, losing whatever was in it")
	}

	// An empty file, on the other hand, is what an interrupted first write
	// leaves and means nothing is stored yet.
	if err := os.WriteFile(s.Path(), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, found, err := s.Load("copilot"); err != nil || found {
		t.Errorf("Load of an empty file = %v, %v, want not-found and no error", found, err)
	}
}

func TestDefaultStorePathFollowsXDG(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "/somewhere/config")
	got, err := DefaultStorePath()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join("/somewhere/config", "genai-io", "credentials.json"); got != want {
		t.Errorf("DefaultStorePath = %q, want %q", got, want)
	}

	t.Setenv("XDG_CONFIG_HOME", "")
	got, err = DefaultStorePath()
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory to check the fallback against")
	}
	if want := filepath.Join(home, ".config", "genai-io", "credentials.json"); got != want {
		t.Errorf("DefaultStorePath = %q, want %q", got, want)
	}
}

func TestResolveStorePrefersWhatItIsGiven(t *testing.T) {
	given := NewMemoryStore()
	got, err := resolveStore(given)
	if err != nil {
		t.Fatal(err)
	}
	if got != Store(given) {
		t.Error("resolveStore ignored the store it was handed")
	}

	fallback := NewMemoryStore()
	defer withDefaultStore(t, fallback)()
	got, err = resolveStore(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != Store(fallback) {
		t.Error("resolveStore ignored DefaultStore, which is how a server opts out of writing to disk")
	}
}

// newTestFileStore returns a store under the test's own directory, so nothing
// here can read or write the developer's real credentials.
func newTestFileStore(t *testing.T) *FileStore {
	t.Helper()
	s, err := NewFileStore(filepath.Join(t.TempDir(), "store", "credentials.json"))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// withDefaultStore swaps the package-level default and returns the undo. The
// default is process-wide, so a test that changes it has to put it back.
func withDefaultStore(t *testing.T, s Store) func() {
	t.Helper()
	previous := DefaultStore
	DefaultStore = s
	return func() { DefaultStore = previous }
}
