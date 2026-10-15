package session_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
	"github.com/genai-io/sdk-go/pkg/session/jsonfile"
)

// memStore is the second implementation of session.Store, which is what keeps
// the contract honest about describing more than one.
type memStore struct {
	mu       sync.Mutex
	sessions map[string][]ai.Message
}

func (s *memStore) Save(ctx context.Context, id string, msgs []ai.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sessions == nil {
		s.sessions = map[string][]ai.Message{}
	}
	s.sessions[id] = clone(msgs)
	return nil
}

func (s *memStore) Load(ctx context.Context, id string) ([]ai.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	msgs, ok := s.sessions[id]
	if !ok {
		return nil, fmt.Errorf("memory: %s: %w", id, session.ErrNotFound)
	}
	return clone(msgs), nil
}

func clone(msgs []ai.Message) []ai.Message {
	out := slices.Clone(msgs)
	for i := range out {
		out[i].Content = out[i].Content.Clone()
	}
	return out
}

// The Store contract, run against every implementation.
func TestStoreContract(t *testing.T) {
	for _, impl := range []struct {
		name string
		open func(*testing.T) session.Store
	}{
		{"memory", func(*testing.T) session.Store { return &memStore{} }},
		{"jsonfile", func(t *testing.T) session.Store {
			s, err := jsonfile.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			return s
		}},
	} {
		t.Run(impl.name, func(t *testing.T) {
			ctx := context.Background()
			texts := func(t *testing.T, st session.Store, id string) []string {
				t.Helper()
				msgs, err := st.Load(ctx, id)
				if err != nil {
					t.Fatal(err)
				}
				var out []string
				for _, m := range msgs {
					out = append(out, m.Text())
				}
				return out
			}

			t.Run("a session never saved is ErrNotFound", func(t *testing.T) {
				if _, err := impl.open(t).Load(ctx, "new"); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("Load = %v, want ErrNotFound", err)
				}
			})

			t.Run("a save is what loads, names included", func(t *testing.T) {
				st := impl.open(t)
				first := ai.UserMessage("hello")
				first.ID = "m1"
				if err := st.Save(ctx, "s", []ai.Message{first, ai.UserMessage("again")}); err != nil {
					t.Fatal(err)
				}
				msgs, err := st.Load(ctx, "s")
				if err != nil {
					t.Fatal(err)
				}
				if len(msgs) != 2 || msgs[0].ID != "m1" || msgs[0].Text() != "hello" || msgs[1].Text() != "again" {
					t.Errorf("loaded %+v", msgs)
				}
			})

			t.Run("a save replaces the one before, down to nothing", func(t *testing.T) {
				st := impl.open(t)
				for _, msgs := range [][]ai.Message{
					{ai.UserMessage("one"), ai.UserMessage("two")},
					{ai.UserMessage("summary")},
				} {
					if err := st.Save(ctx, "s", msgs); err != nil {
						t.Fatal(err)
					}
				}
				if got := texts(t, st, "s"); !slices.Equal(got, []string{"summary"}) {
					t.Errorf("after a second save: %v", got)
				}
				if err := st.Save(ctx, "s", nil); err != nil {
					t.Fatal(err)
				}
				if got := texts(t, st, "s"); len(got) != 0 {
					t.Errorf("a cleared conversation loads as %v", got)
				}
			})

			t.Run("what was saved is not what the caller still holds", func(t *testing.T) {
				st := impl.open(t)
				msgs := []ai.Message{ai.UserMessage("as written")}
				if err := st.Save(ctx, "s", msgs); err != nil {
					t.Fatal(err)
				}
				msgs[0].Content[0].Text = "edited after saving"
				loaded, err := st.Load(ctx, "s")
				if err != nil {
					t.Fatal(err)
				}
				loaded[0].Content[0].Text = "edited after loading"
				if got := texts(t, st, "s"); !slices.Equal(got, []string{"as written"}) {
					t.Errorf("stored %v; an edit reached it", got)
				}
			})

			t.Run("a cancelled call does nothing", func(t *testing.T) {
				st := impl.open(t)
				stopped, cancel := context.WithCancel(ctx)
				cancel()
				if err := st.Save(stopped, "s", nil); !errors.Is(err, context.Canceled) {
					t.Errorf("Save = %v", err)
				}
				if _, err := st.Load(stopped, "s"); !errors.Is(err, context.Canceled) {
					t.Errorf("Load = %v", err)
				}
				if _, err := st.Load(ctx, "s"); !errors.Is(err, session.ErrNotFound) {
					t.Errorf("a cancelled save wrote something: %v", err)
				}
			})
		})
	}
}
