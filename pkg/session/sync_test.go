package session

import (
	"context"
	"errors"
	"os"
	"testing"
)

func TestUnchangedSaveDoesNotSync(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	w, err := s.Writer(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	if err := w.Append("note", 1); err != nil {
		t.Fatal(err)
	}
	// Force fsync to fail without changing the writer's cached conversation.
	if err := w.f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := w.Save(nil); err != nil {
		t.Fatalf("unchanged Save synced: %v", err)
	}
	failure := w.Sync()
	if !errors.Is(failure, os.ErrClosed) {
		t.Fatalf("Sync = %v, want closed file error", failure)
	}
	if err := w.Sync(); err != failure {
		t.Fatalf("Sync lost sticky error: %v", err)
	}
	if err := w.Save(nil); err != failure {
		t.Fatalf("Save lost sticky error: %v", err)
	}
	if err := w.Append("note", 2); err != failure {
		t.Fatalf("Append lost sticky error: %v", err)
	}
}
