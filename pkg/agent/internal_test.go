package agent

import (
	"context"
	"runtime"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/aitest"
)

// Run keeps a channel per exchange so Interrupt can say when that one is over.
// It has to hand it back when the exchange ends: an agent that kept the last
// one would hold a channel per run it ever made, and would answer the next
// Interrupt with a channel that is never going to close.
//
// This is an in-package test because the invariant is a field, and a field is
// the only honest place to check it — a finalizer test here proves whichever
// local went out of scope, not what the agent still holds.
func TestAFinishedExchangeHandsBackItsChannel(t *testing.T) {
	a := newTestAgent(t, aitest.Says("one"), aitest.Says("two"))

	a.mu.Lock()
	atStart := a.stopped
	a.mu.Unlock()
	if atStart != closed {
		t.Fatal("a fresh agent is not idle")
	}

	var during chan struct{}
	for e := range a.Run(context.Background(), ai.UserMessage("go")) {
		if _, ok := e.(TurnStart); ok {
			a.mu.Lock()
			during = a.stopped
			a.mu.Unlock()
		}
	}

	if during == nil || during == closed {
		t.Fatal("a running exchange did not take a channel of its own")
	}
	select {
	case <-during:
	default:
		t.Error("the exchange ended without closing its channel; a waiter would hang forever")
	}

	a.mu.Lock()
	after := a.stopped
	a.mu.Unlock()
	if after != closed {
		t.Error("the agent kept the finished exchange's channel instead of handing it back")
	}

	// And the next one takes a fresh one rather than reusing the closed stand-in.
	for e := range a.Run(context.Background(), ai.UserMessage("again")) {
		if _, ok := e.(TurnStart); ok {
			a.mu.Lock()
			second := a.stopped
			a.mu.Unlock()
			if second == during {
				t.Error("the second exchange reused the first one's channel")
			}
			if second == closed {
				t.Error("the second exchange did not take a channel of its own")
			}
		}
	}
}

// Many exchanges on one agent must not grow what it holds.
func TestManyExchangesDoNotAccumulate(t *testing.T) {
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
		t.Errorf("the heap grew %d bytes over %d exchanges", grew, runs-50)
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
