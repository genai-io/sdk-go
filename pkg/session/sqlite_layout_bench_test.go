package session

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Compare physical layouts without making a second production backend option.
// Every database here is a fresh benchmark fixture with identical constraints.
func sqliteLayoutStore(b *testing.B, withoutRowID bool) *Store {
	b.Helper()
	s := backend(b, true)
	if withoutRowID {
		_, err := s.sql.db.Exec(`DROP TABLE session_events;
CREATE TABLE session_events (session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, seq INTEGER NOT NULL CHECK(seq >= 0), record BLOB NOT NULL, PRIMARY KEY(session_id, seq)) WITHOUT ROWID;`)
		must(b, err)
	}
	return s
}

func sqliteLayoutFootprint(b *testing.B, s *Store) {
	b.Helper()
	// Include the live WAL before checkpoint, then measure the compact fixture
	// after checkpoint. Neither number includes external application assets.
	path := strings.TrimSuffix(s.sql.dir, ".locks")
	var live int64
	for _, suffix := range []string{"", "-wal", "-shm"} {
		info, err := os.Stat(path + suffix)
		if os.IsNotExist(err) {
			continue
		}
		must(b, err)
		live += info.Size()
	}
	b.ReportMetric(float64(live), "live-B/fixture")
	_, err := s.sql.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	must(b, err)
	info, err := os.Stat(path)
	must(b, err)
	b.ReportMetric(float64(info.Size()), "db-B/fixture")
}

func BenchmarkSQLiteLayouts(b *testing.B) {
	for _, withoutRowID := range []bool{false, true} {
		name := "rowid"
		if withoutRowID {
			name = "without-rowid"
		}
		b.Run(name, func(b *testing.B) {
			for _, large := range []bool{false, true} {
				size := 256
				label := "small"
				if large {
					size, label = 8192, "large"
				}
				b.Run("Replay1000/"+label, func(b *testing.B) {
					s := sqliteLayoutStore(b, withoutRowID)
					w, err := s.Writer(context.Background(), "history")
					must(b, err)
					msgs := make([]ai.Message, 1000)
					for i := range msgs {
						msgs[i] = ai.UserMessage(strings.Repeat("x", size))
					}
					must(b, w.Save(msgs))
					must(b, w.Close())
					b.ReportAllocs()
					b.SetBytes(int64(size * len(msgs)))
					for b.Loop() {
						got, err := s.Load(context.Background(), "history")
						must(b, err)
						if len(got) != len(msgs) {
							b.Fatal("incomplete replay")
						}
					}
					sqliteLayoutFootprint(b, s)
				})
			}
			b.Run("AppendSync", func(b *testing.B) {
				s := sqliteLayoutStore(b, withoutRowID)
				w, err := s.Writer(context.Background(), "app")
				must(b, err)
				defer w.Close()
				data := strings.Repeat("x", 1024)
				b.ReportAllocs()
				for b.Loop() {
					must(b, w.Append("audit", data))
				}
				sqliteLayoutFootprint(b, s)
			})
		})
	}
}
