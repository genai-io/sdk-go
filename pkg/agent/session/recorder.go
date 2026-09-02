package session

import (
	"context"
	"reflect"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Recorder turns an agent's events into stored entries.
//
// It holds no state machine. Every event says which turn it belongs to and
// carries what opened it, so recording one is a translation and not a
// reconstruction — which is also why Handle must not be called concurrently
// for a session and needs no lock to say so.
type Recorder struct {
	store Store
	id    string

	// turnsBefore is how many exchanges this session already held when it was
	// opened. The agent counts from one per run; entries are numbered from
	// the session's beginning.
	turnsBefore int

	// restored is the conversation Open handed back, held until the agent
	// announces it, so that resuming does not record a copy of what it read.
	restored []ai.Message

	mu  sync.Mutex
	err error
}

// newRecorder returns a recorder writing into one session. Unexported because
// Open is the way in: without what Open read back it would number turns from
// one again and re-record the history it was resumed from.
func newRecorder(store Store, id string) *Recorder {
	return &Recorder{store: store, id: id}
}

// Handle records one event. Call it from your event loop, in order, before
// anything slow. The context is the store's: one that is not the local
// filesystem can block, and this sits on the loop delivering events.
//
// After the first failed write it does nothing, and Err says so. A fold with a
// hole in it is not a shorter conversation but a broken one — drop the message
// carrying a tool call and the results answering it are orphaned — so stopping
// leaves a prefix that still folds where carrying on would not.
func (r *Recorder) Handle(ctx context.Context, e agent.Event) {
	if r.Err() != nil {
		return
	}

	switch v := e.(type) {
	case agent.MessageAdded:
		msg := v.Message
		r.write(ctx, v.Turn, Entry{Type: EntryMessage, Message: &msg})

	case agent.MessagesReplaced:
		// The one Open handed over is not news: it was read from here.
		if r.restored != nil {
			same := reflect.DeepEqual(r.restored, v.Messages)
			r.restored = nil
			if same {
				return
			}
		}
		r.write(ctx, v.Turn, Entry{Type: EntrySnapshot, Snapshot: v.Messages})

	case agent.MessageEnd:
		rec := Inference{Attempt: v.Attempt}
		if v.Inference != nil {
			rec.System = v.Inference.System
			for _, t := range v.Inference.Tools {
				rec.Tools = append(rec.Tools, t.Schema.Name)
			}
		}
		if v.Response != nil {
			rec.Model = v.Response.Model
			rec.Usage = v.Response.Usage
			rec.StopReason = v.Response.StopReason
		}
		if v.Err != nil {
			rec.Error = v.Err.Error()
		}
		r.write(ctx, v.Turn, Entry{Type: EntryInference, Inference: &rec})

	case agent.ToolEnd:
		// The same text the model was told, from the same function, so the
		// record cannot come to disagree with the conversation.
		r.write(ctx, v.Turn, Entry{Type: EntryToolRun, ToolRun: &ToolRun{
			ID: v.ID, Name: v.Name, Args: v.Args,
			Content: agent.ResultText(v.Result, v.Err), IsError: v.Err != nil,
		}})

	case agent.TurnEnd:
		rec := Outcome{StopReason: v.StopReason, Usage: v.Usage}
		if v.Err != nil {
			rec.Err = v.Err.Error()
		}
		r.write(ctx, v.Turn, Entry{Type: EntryOutcome, Outcome: &rec})
	}
}

// Err reports the write that stopped recording, if one did.
func (r *Recorder) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}

// write stores one entry, stamping the turn it belongs to and when the agent
// said it, rather than when a slow store got round to it. The agent numbers
// turns from one on every run; that mapping happens here, once, rather than at
// each of the five call sites where forgetting it would be silent.
func (r *Recorder) write(ctx context.Context, turn int, e Entry) {
	e.Turn = r.turnsBefore + turn
	e.At = time.Now().UTC()
	if err := r.store.Append(ctx, r.id, e); err != nil {
		r.mu.Lock()
		if r.err == nil {
			r.err = err
		}
		r.mu.Unlock()
	}
}

// ID reports which session this records into.
func (r *Recorder) ID() string { return r.id }
