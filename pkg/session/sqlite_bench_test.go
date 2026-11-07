package session

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// Both backends include the same durable boundary. These benchmarks measure
// local persistence, not provider inference or application event projection.
func BenchmarkSessionBackends(b *testing.B) {
	for _, sqlite := range []bool{false, true} {
		name := "JSONL"
		if sqlite {
			name = "SQLite"
		}
		b.Run(name, func(b *testing.B) {
			b.Run("AppendSync", func(b *testing.B) {
				s := backend(b, sqlite)
				w, err := s.Writer(context.Background(), "app")
				must(b, err)
				defer w.Close()
				payload := strings.Repeat("x", 1024)
				b.ReportAllocs()
				b.SetBytes(int64(len(payload)))
				for b.Loop() {
					must(b, w.Append("audit", payload))
					must(b, w.Sync())
				}
			})
			b.Run("SaveReplaceSuffix", func(b *testing.B) {
				s := backend(b, sqlite)
				w, err := s.Writer(context.Background(), "save")
				must(b, err)
				defer w.Close()
				msgs := make([]ai.Message, 100)
				for i := range msgs {
					msgs[i] = ai.UserMessage(strings.Repeat("history", 32))
				}
				must(b, w.Save(msgs))
				i := 0
				b.ReportAllocs()
				for b.Loop() {
					msgs[99] = ai.UserMessage(fmt.Sprint(i))
					i++
					must(b, w.Save(msgs))
				}
			})
			b.Run("Replay", func(b *testing.B) {
				s := backend(b, sqlite)
				w, err := s.Writer(context.Background(), "replay")
				must(b, err)
				msgs := make([]ai.Message, 100)
				for i := range msgs {
					msgs[i] = ai.UserMessage(strings.Repeat("history", 32))
				}
				must(b, w.Save(msgs))
				must(b, w.Close())
				b.ReportAllocs()
				for b.Loop() {
					_, err := s.Load(context.Background(), "replay")
					must(b, err)
				}
			})
			b.Run("List100", func(b *testing.B) {
				s := backend(b, sqlite)
				for i := range 100 {
					w, err := s.Writer(context.Background(), fmt.Sprint(i))
					must(b, err)
					must(b, w.Close())
				}
				b.ReportAllocs()
				for b.Loop() {
					_, err := s.List(context.Background())
					must(b, err)
				}
			})
			b.Run("IndependentAppendSync", func(b *testing.B) {
				s := backend(b, sqlite)
				var id atomic.Int64
				writers := make([]*Writer, runtime.GOMAXPROCS(0))
				for i := range writers {
					w, err := s.Writer(context.Background(), fmt.Sprint(i))
					must(b, err)
					writers[i] = w
					defer w.Close()
				}
				payload := strings.Repeat("x", 1024)
				b.ReportAllocs()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					w := writers[id.Add(1)-1]
					for pb.Next() {
						if err := w.Append("audit", payload); err != nil {
							b.Error(err)
							return
						}
						if err := w.Sync(); err != nil {
							b.Error(err)
							return
						}
					}
				})
				b.StopTimer()
			})
		})
	}
}
