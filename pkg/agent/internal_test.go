package agent

import (
	"context"
	"runtime"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
)

// Many turns on one agent must not grow what it holds.
func TestManyTurnsDoNotAccumulate(t *testing.T) {
	const runs = 500

	turns := make([]aitest.Turn, runs)
	for i := range turns {
		turns[i] = aitest.Says("ok")
	}
	a := newTestAgent(t, turns...)

	settle := func() uint64 {
		runtime.GC()
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.HeapAlloc
	}

	for range 50 { // warm up, so the baseline is not measuring first-run costs
		for range a.Run(context.Background(), ai.UserMessage("go")) {
		}
	}
	before := settle()
	for range runs - 50 {
		for range a.Run(context.Background(), ai.UserMessage("go")) {
		}
	}
	after := settle()

	// The conversation itself grows — that is the point of an agent — so this
	// is a ceiling on everything else, not a claim of zero growth.
	if grew := int64(after) - int64(before); grew > 4<<20 {
		t.Errorf("the heap grew %d bytes over %d turns", grew, runs-50)
	}
}

func newTestAgent(t *testing.T, turns ...aitest.Turn) *Agent {
	t.Helper()
	a, err := New(aitest.New(turns...).Client())
	if err != nil {
		t.Fatal(err)
	}
	return a
}
