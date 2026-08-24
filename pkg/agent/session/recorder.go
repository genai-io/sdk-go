package session

import (
	"context"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
)

// Recorder turns an agent's events into stored entries.
type Recorder struct {
	store Store
	id    string

	mu       sync.Mutex
	open     agent.MessageStart
	openTool map[string]agent.ToolStart
	turn     int // the turn now running, from the last TurnStart
	err      error
}

// NewRecorder returns a recorder writing into one session.
func NewRecorder(store Store, id string) *Recorder {
	return &Recorder{store: store, id: id, openTool: map[string]agent.ToolStart{}}
}

// Handle records one event. Call it from your event loop before anything that
// might be slow: a session that falls behind the agent is a session that loses
// the last thing that happened.
func (r *Recorder) Handle(e agent.Event) {
	switch v := e.(type) {
	case agent.TurnStart:
		// Messages carry no turn number of their own: the events arrive in
		// order, so the turn is whichever one started last.
		r.mu.Lock()
		r.turn = v.Turn
		r.mu.Unlock()

	case agent.MessageAdded:
		msg := v.Message
		r.write(Entry{Type: EntryMessage, Message: &msg})

	case agent.MessageStart:
		r.mu.Lock()
		r.open = v
		r.mu.Unlock()

	case agent.MessageEnd:
		r.mu.Lock()
		started, turn := r.open, r.turn
		r.open = agent.MessageStart{}
		r.mu.Unlock()

		rec := Inference{Turn: turn, Attempt: started.Attempt}
		if req := started.Request; req != nil {
			rec.System = req.System
			for _, t := range req.Tools {
				rec.Tools = append(rec.Tools, t.Name)
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
		r.write(Entry{Type: EntryInference, Inference: &rec})

	case agent.ToolStart:
		r.mu.Lock()
		r.openTool[v.ID] = v
		r.mu.Unlock()

	case agent.ToolEnd:
		r.mu.Lock()
		started := r.openTool[v.ID]
		delete(r.openTool, v.ID)
		r.mu.Unlock()

		rec := Tool{ID: v.ID, Name: started.Name, Args: started.Args, Content: v.Result.Text()}
		if v.Err != nil {
			rec.IsError = true
			if rec.Content == "" {
				rec.Content = v.Err.Error()
			}
		}
		r.write(Entry{Type: EntryTool, Tool: &rec})

	case agent.TurnEnd:
		rec := Turn{Turn: v.Turn, StopReason: v.StopReason, Usage: v.Usage}
		if v.Err != nil {
			rec.Err = v.Err.Error()
		}
		r.write(Entry{Type: EntryTurn, Turn: &rec})
	}
}

// Err reports the first write that failed.
func (r *Recorder) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}

func (r *Recorder) write(e Entry) {
	if e.At.IsZero() {
		e.At = time.Now().UTC()
	}
	if err := r.store.Append(context.Background(), r.id, e); err != nil {
		r.mu.Lock()
		if r.err == nil {
			r.err = err
		}
		r.mu.Unlock()
	}
}

// ID reports which session this records into.
func (r *Recorder) ID() string { return r.id }
