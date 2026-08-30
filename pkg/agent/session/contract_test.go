package session_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/agent/session/memory"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// The Store contract, run against every implementation. An interface with one
// implementation has never been asked whether it describes anything but that
// implementation; this is where a third one finds out what it has to honour
// before anything else in this package will work on it.
func TestStoreContract(t *testing.T) {
	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", func(*testing.T) session.Store { return memory.Open() }},
		{"jsonl", func(t *testing.T) session.Store { return jsonlStore(t) }},
	} {
		t.Run(impl.name, func(t *testing.T) {
			t.Run("Create assigns an id when none is given", func(t *testing.T) {
				meta, err := impl.open(t).Create(ctx(), session.Meta{})
				if err != nil {
					t.Fatal(err)
				}
				if meta.ID == "" {
					t.Error("Create returned a session with no id")
				}
				if meta.CreatedAt.IsZero() || meta.UpdatedAt.IsZero() {
					t.Error("Create left the timestamps unset")
				}
				if meta.Entries != 0 {
					t.Errorf("a new session reports %d entries", meta.Entries)
				}
			})

			t.Run("Create keeps an id it was given, and refuses it twice", func(t *testing.T) {
				st := impl.open(t)
				meta, err := st.Create(ctx(), session.Meta{ID: "chosen", Title: "kept"})
				if err != nil {
					t.Fatal(err)
				}
				if meta.ID != "chosen" || meta.Title != "kept" {
					t.Errorf("Create = %+v, want the id and title it was given", meta)
				}
				if _, err := st.Create(ctx(), session.Meta{ID: "chosen"}); err == nil {
					t.Error("the same id was created twice")
				}
			})

			t.Run("an absent session is ErrNotFound", func(t *testing.T) {
				st := impl.open(t)
				if _, err := st.Meta(ctx(), "nope"); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Meta = %v, want ErrNotFound", err)
				}
				if err := st.Append(ctx(), "nope", msg("x")); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Append = %v, want ErrNotFound", err)
				}
				for _, err := range st.Entries(ctx(), "nope") {
					if !errors.Is(err, session.ErrNotFound) {
						t.Errorf("Entries = %v, want ErrNotFound", err)
					}
				}
			})

			t.Run("Append numbers entries from one, in order", func(t *testing.T) {
				st := impl.open(t)
				meta, err := st.Create(ctx(), session.Meta{})
				if err != nil {
					t.Fatal(err)
				}
				if err := st.Append(ctx(), meta.ID, msg("one"), msg("two")); err != nil {
					t.Fatal(err)
				}
				if err := st.Append(ctx(), meta.ID, msg("three")); err != nil {
					t.Fatal(err)
				}

				var seqs []int64
				var texts []string
				for e, err := range st.Entries(ctx(), meta.ID) {
					if err != nil {
						t.Fatal(err)
					}
					if e.At.IsZero() {
						t.Errorf("entry %d was stored with no timestamp", e.Seq)
					}
					seqs = append(seqs, e.Seq)
					texts = append(texts, e.Message.Text())
				}
				if want := []int64{1, 2, 3}; !equal(seqs, want) {
					t.Errorf("Seq = %v, want %v", seqs, want)
				}
				if want := []string{"one", "two", "three"}; !equal(texts, want) {
					t.Errorf("order = %v, want %v", texts, want)
				}
			})

			t.Run("Meta counts what was appended", func(t *testing.T) {
				st := impl.open(t)
				meta, err := st.Create(ctx(), session.Meta{})
				if err != nil {
					t.Fatal(err)
				}
				before := time.Now().Add(-time.Second)
				if err := st.Append(ctx(), meta.ID, msg("one"), msg("two")); err != nil {
					t.Fatal(err)
				}
				got, err := st.Meta(ctx(), meta.ID)
				if err != nil {
					t.Fatal(err)
				}
				if got.Entries != 2 {
					t.Errorf("Entries = %d, want 2", got.Entries)
				}
				if !got.UpdatedAt.After(before) {
					t.Errorf("UpdatedAt = %v, want a time after the append", got.UpdatedAt)
				}
			})

			t.Run("an empty Append is not an error", func(t *testing.T) {
				st := impl.open(t)
				meta, err := st.Create(ctx(), session.Meta{})
				if err != nil {
					t.Fatal(err)
				}
				if err := st.Append(ctx(), meta.ID); err != nil {
					t.Errorf("Append with no entries = %v, want nil", err)
				}
			})

			t.Run("what was stored is not what the caller still holds", func(t *testing.T) {
				st := impl.open(t)
				meta, err := st.Create(ctx(), session.Meta{})
				if err != nil {
					t.Fatal(err)
				}
				m := ai.UserMessage("as written")
				e := session.Entry{Type: session.EntryMessage, Message: &m}
				if err := st.Append(ctx(), meta.ID, e); err != nil {
					t.Fatal(err)
				}
				m.Content = ai.TextContent("rewritten after the fact")

				for got, err := range st.Entries(ctx(), meta.ID) {
					if err != nil {
						t.Fatal(err)
					}
					if got.Message.Text() != "as written" {
						t.Errorf("stored entry reads %q; the caller edited it after appending",
							got.Message.Text())
					}
				}
			})
		})
	}
}

func equal[T comparable](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func ctx() context.Context { return context.Background() }

func msg(text string) session.Entry {
	m := ai.UserMessage(text)
	return session.Entry{Type: session.EntryMessage, Message: &m}
}
