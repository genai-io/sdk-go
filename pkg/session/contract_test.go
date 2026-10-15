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
			created := func(t *testing.T) (session.Store, string) {
				t.Helper()
				st := impl.open(t)
				id, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				return st, id
			}
			load := func(t *testing.T, st session.Store, id string) []string {
				t.Helper()
				msgs, err := st.Load(ctx(), id)
				if err != nil {
					t.Fatal(err)
				}
				var texts []string
				for _, m := range msgs {
					texts = append(texts, m.Text())
				}
				return texts
			}

			t.Run("Create assigns distinct ids to empty conversations", func(t *testing.T) {
				st, a := created(t)
				b, err := st.Create(ctx())
				if err != nil {
					t.Fatal(err)
				}
				if a == "" || a == b {
					t.Errorf("Create gave %q then %q", a, b)
				}
				if got := load(t, st, a); len(got) != 0 {
					t.Errorf("a new session holds %v", got)
				}
			})

			t.Run("an absent session is ErrNotFound", func(t *testing.T) {
				st := impl.open(t)
				if _, err := st.Claim(ctx(), "nope"); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Claim = %v", err)
				}
				if err := st.Append(ctx(), "nope", ai.UserMessage("x")); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Append = %v", err)
				}
				if err := st.Replace(ctx(), "nope", nil); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Replace = %v", err)
				}
				if _, err := st.Load(ctx(), "nope"); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Load = %v", err)
				}
			})

			t.Run("a claim excludes a second until released", func(t *testing.T) {
				st, id := created(t)
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

			t.Run("appends keep their order and a replacement starts over", func(t *testing.T) {
				st, id := created(t)
				for _, text := range []string{"one", "two"} {
					if err := st.Append(ctx(), id, ai.UserMessage(text)); err != nil {
						t.Fatal(err)
					}
				}
				if got, want := load(t, st, id), []string{"one", "two"}; !slices.Equal(got, want) {
					t.Errorf("after appends: %v, want %v", got, want)
				}
				if err := st.Replace(ctx(), id, []ai.Message{ai.UserMessage("summary")}); err != nil {
					t.Fatal(err)
				}
				if err := st.Append(ctx(), id, ai.UserMessage("three")); err != nil {
					t.Fatal(err)
				}
				if got, want := load(t, st, id), []string{"summary", "three"}; !slices.Equal(got, want) {
					t.Errorf("after a replacement: %v, want %v", got, want)
				}
				if err := st.Replace(ctx(), id, nil); err != nil {
					t.Fatal(err)
				}
				if got := load(t, st, id); len(got) != 0 {
					t.Errorf("a cleared conversation loads as %v", got)
				}
			})

			t.Run("a cancelled load reports the cancellation", func(t *testing.T) {
				st, id := created(t)
				stopped, cancel := context.WithCancel(context.Background())
				cancel()
				if _, err := st.Load(stopped, id); !errors.Is(err, context.Canceled) {
					t.Errorf("Load = %v, want the cancellation — a load that stops "+
						"quietly reads as a shorter conversation", err)
				}
			})

			t.Run("what was stored is not what the caller still holds", func(t *testing.T) {
				st, id := created(t)
				m := ai.UserMessage("as written")
				if err := st.Append(ctx(), id, m); err != nil {
					t.Fatal(err)
				}
				m.Content[0].Text = "rewritten after the fact"
				if got := load(t, st, id); !slices.Equal(got, []string{"as written"}) {
					t.Errorf("stored %v; the caller edited it after appending", got)
				}
			})
		})
	}
}

func ctx() context.Context { return context.Background() }
