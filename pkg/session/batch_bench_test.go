package session_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/session"
)

// Each operation persists the same eight records and completes file Sync before
// returning. Individual operations additionally guarantee durability after each
// record; grouping deliberately trades those intermediate boundaries for fewer
// syncs. These are real temporary files, not a mock or a disk-free throughput test.
func BenchmarkCommitGroup(b *testing.B) {
	const count = 8
	payload := json.RawMessage(`{"text":"` + strings.Repeat("session payload ", 32) + `"}`)
	records := make([]session.Record, count)
	for i := range records {
		records[i] = session.Record{Type: session.TypeApp, Time: time.Unix(1700000000, 0).UTC(), Name: "event", Data: payload}
	}
	for _, operation := range []string{"CommitGrouped", "CommitIndividual", "AppendGroupedSync", "AppendIndividualSync"} {
		b.Run(operation, func(b *testing.B) {
			_, w := benchmarkStore(b)
			ctx := context.Background()
			var tail session.Tail
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				switch operation {
				case "CommitGrouped":
					receipt, err := w.Commit(ctx, session.Batch{ID: fmt.Sprintf("group-%d", i), Expected: tail, Records: records})
					if err != nil {
						b.Fatal(err)
					}
					tail = receipt.Tail
				case "CommitIndividual":
					for j := range records {
						receipt, err := w.Commit(ctx, session.Batch{ID: fmt.Sprintf("record-%d-%d", i, j), Expected: tail, Records: records[j : j+1]})
						if err != nil {
							b.Fatal(err)
						}
						tail = receipt.Tail
					}
				default:
					for range records {
						if err := w.Append("event", payload); err != nil {
							b.Fatal(err)
						}
						if operation == "AppendIndividualSync" {
							if err := w.Sync(); err != nil {
								b.Fatal(err)
							}
						}
					}
					if operation == "AppendGroupedSync" {
						if err := w.Sync(); err != nil {
							b.Fatal(err)
						}
					}
				}
			}
			b.ReportMetric(count, "records/op")
			syncs := 1
			if operation == "CommitIndividual" || operation == "AppendIndividualSync" {
				syncs = count
			}
			b.ReportMetric(float64(syncs), "syncs/op")
		})
	}
}
