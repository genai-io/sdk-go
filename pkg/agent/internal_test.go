package agent

import (
	"context"
	"errors"
	"iter"
	"runtime"
	"strconv"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
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
	a := newTestAgent(t, text("one"), text("two"))

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

	scripts := make([][]ai.Delta, runs)
	for i := range scripts {
		scripts[i] = text("ok")
	}
	a := newTestAgent(t, scripts...)

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

func newTestAgent(t *testing.T, scripts ...[]ai.Delta) *Agent {
	t.Helper()
	client := ai.NewClientWithDriver(&scripted{scripts: scripts}, ai.Model{ID: "stub", API: "stub"})
	a, err := New(client)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// scripted is a model that streams a script per call. These two tests are
// in-package because the invariant they check is a field, so they cannot use
// the stub the external tests share.
type scripted struct {
	scripts [][]ai.Delta

	mu    sync.Mutex
	calls int
}

func (d *scripted) Name() string { return "scripted" }

func (d *scripted) Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
	d.mu.Lock()
	n := d.calls
	d.calls++
	d.mu.Unlock()
	return func(yield func(ai.Delta, error) bool) {
		if n >= len(d.scripts) {
			yield(ai.Delta{}, errors.New("scripted: no script for call "+strconv.Itoa(n)))
			return
		}
		for _, delta := range d.scripts[n] {
			if !yield(delta, nil) {
				return
			}
		}
	}
}

func text(s string) []ai.Delta {
	return []ai.Delta{
		{Block: ai.TextBlock(s)},
		{EndBlock: true},
		{StopReason: ai.StopEndTurn},
	}
}
