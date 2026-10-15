package session_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

// The Store contract, run against every implementation. An interface with one
// implementation has never been asked whether it describes anything but that
// implementation; this is where a third one finds out what it has to honour.
func TestStoreContract(t *testing.T) {
	for _, impl := range stores {
		t.Run(impl.name, func(t *testing.T) {
			t.Run("Create assigns distinct ids", func(t *testing.T) {
				st := impl.open(t)
				a, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				b, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				if a == "" || a == b {
					t.Errorf("Create gave %q then %q", a, b)
				}
			})

			t.Run("an absent session is ErrNotFound", func(t *testing.T) {
				st := impl.open(t)
				if _, err := st.Claim(ctx(), "nope"); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Claim = %v, want ErrNotFound", err)
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

			t.Run("a claim excludes a second until released", func(t *testing.T) {
				st := impl.open(t)
				id, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				release, err := st.Claim(ctx(), id)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := st.Claim(ctx(), id); err == nil {
					t.Fatal("claimed twice")
				}
				if err := release(); err != nil {
					t.Fatal(err)
				}
				again, err := st.Claim(ctx(), id)
				if err != nil {
					t.Fatalf("a released claim was not given back: %v", err)
				}
				_ = again()
			})

			t.Run("Entries reads back what was appended, in order", func(t *testing.T) {
				st := impl.open(t)
				id, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				for _, text := range []string{"one", "two", "three"} {
					if err := st.Append(ctx(), id, msg(text)); err != nil {
						t.Fatal(err)
					}
				}
				var texts []string
				for e, err := range st.Entries(ctx(), id) {
					if err != nil {
						t.Fatal(err)
					}
					texts = append(texts, e.Messages[0].Text())
				}
				if want := []string{"one", "two", "three"}; !slices.Equal(texts, want) {
					t.Errorf("order = %v, want %v", texts, want)
				}
			})

			t.Run("a cancelled read ends with the cancellation", func(t *testing.T) {
				st := impl.open(t)
				id, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				if err := st.Append(ctx(), id, msg("one")); err != nil {
					t.Fatal(err)
				}
				stopped, cancel := context.WithCancel(context.Background())
				cancel()
				var failure error
				for _, err := range st.Entries(stopped, id) {
					if err != nil {
						failure = err
						break
					}
				}
				if !errors.Is(failure, context.Canceled) {
					t.Errorf("Entries = %v, want the cancellation — a read that stops "+
						"quietly reads as a shorter session", failure)
				}
			})

			t.Run("what was stored is not what the caller still holds", func(t *testing.T) {
				st := impl.open(t)
				id, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				e := msg("as written")
				if err := st.Append(ctx(), id, e); err != nil {
					t.Fatal(err)
				}
				e.Messages[0].Content = ai.TextContent("rewritten after the fact")
				for got, err := range st.Entries(ctx(), id) {
					if err != nil {
						t.Fatal(err)
					}
					if got.Messages[0].Text() != "as written" {
						t.Errorf("stored entry reads %q; the caller edited it after appending",
							got.Messages[0].Text())
					}
				}
			})
		})
	}
}

func ctx() context.Context { return context.Background() }
