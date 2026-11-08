package session_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/session"
)

func benchmarkStore(b *testing.B) (*session.Store, *session.Writer) {
	b.Helper()
	s, err := session.Open(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	w, err := s.Writer(context.Background(), "bench")
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if err := w.Close(); err != nil {
			b.Error(err)
		}
	})
	return s, w
}

// The same fixtures run on pre-batch revisions: copy this file alone to compare.
// Replay includes both application payloads and native conversation records.
func BenchmarkLegacyReplay(b *testing.B) {
	for _, kind := range []string{"App", "Native", "Mixed"} {
		b.Run(kind, func(b *testing.B) {
			s, w := benchmarkStore(b)
			ctx := context.Background()
			var messages []ai.Message
			payload := json.RawMessage(`{"text":"` + strings.Repeat("session payload ", 32) + `"}`)
			for i := 0; i < 128; i++ {
				if kind != "Native" {
					if err := w.Append("event", payload); err != nil {
						b.Fatal(err)
					}
				}
				if kind != "App" {
					messages = append(messages, ai.UserMessage(strings.Repeat("message ", 64)))
				}
			}
			if err := w.Save(messages); err != nil {
				b.Fatal(err)
			}
			if err := w.Close(); err != nil {
				b.Fatal(err)
			}
			info, err := s.Stat(ctx, "bench")
			if err != nil {
				b.Fatal(err)
			}
			for _, operation := range []string{"Records", "Load"} {
				b.Run(operation, func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(info.Size)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if operation == "Load" {
							got, err := s.Load(ctx, "bench")
							if err != nil {
								b.Fatal(err)
							}
							if len(got) != len(messages) {
								b.Fatal("wrong message count")
							}
						} else {
							count := 0
							for _, err := range s.Records(ctx, "bench") {
								if err != nil {
									b.Fatal(err)
								}
								count++
							}
							expected := 128
							if kind == "Mixed" {
								expected *= 2
							}
							if count != expected {
								b.Fatal("wrong record count")
							}
						}
					}
				})
			}
		})
	}
}

// SaveChanged and AppendSync both include file Sync. Append measures the
// existing buffered-durability API separately, not as a Commit competitor.
func BenchmarkLegacyWrite(b *testing.B) {
	for _, operation := range []string{"SaveUnchanged", "SaveChanged", "Append", "AppendSync"} {
		b.Run(operation, func(b *testing.B) {
			_, w := benchmarkStore(b)
			messages := []ai.Message{ai.UserMessage(strings.Repeat("message ", 64))}
			if operation == "SaveUnchanged" {
				if err := w.Save(messages); err != nil {
					b.Fatal(err)
				}
			}
			payload := json.RawMessage(`{"text":"` + strings.Repeat("session payload ", 32) + `"}`)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				switch operation {
				case "SaveUnchanged":
					err = w.Save(messages)
				case "SaveChanged":
					messages[0] = ai.UserMessage(fmt.Sprintf("%d %s", i, strings.Repeat("message ", 64)))
					err = w.Save(messages)
				default:
					err = w.Append("event", payload)
					if err == nil && operation == "AppendSync" {
						err = w.Sync()
					}
				}
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
