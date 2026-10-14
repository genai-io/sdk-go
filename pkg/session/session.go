package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// Session is one opened durable identity. It owns its write lease until Close.
type Session struct {
	store Store
	id    string

	// turnsBefore is how many exchanges this session already held when it was
	// opened. The agent counts from one per run; entries are numbered from
	// the session's beginning.
	turnsBefore int

	// restored is the conversation Open handed back, held until the agent
	// announces it, so that resuming does not record a copy of what it read.
	restored []ai.Message

	mu      sync.Mutex
	err     error
	closed  bool
	release func() error
}

// RecordEvent adapts a native agent event to the shared Append path. Call it
// from your event loop, in order, before anything slow. The context is the
// store's: one that is not the local filesystem can block, and this sits on
// the loop delivering events.
//
// After the first failed write it does nothing, and Err says so. A fold with a
// hole in it is not a shorter conversation but a broken one — drop the message
// carrying a tool call and the results answering it are orphaned — so stopping
// leaves a prefix that still folds where carrying on would not.
func (r *Session) RecordEvent(ctx context.Context, e agent.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil || r.closed {
		return
	}

	switch v := e.(type) {
	case agent.MessageAdded:
		msg := v.Message
		_ = r.appendLocked(ctx, Entry{Turn: v.Turn, Type: EntryMessage, Message: &msg})

	case agent.MessagesReplaced:
		// The one Open handed over is not news: it was read from here.
		if r.restored != nil {
			same := reflect.DeepEqual(r.restored, v.Messages)
			r.restored = nil
			if same {
				return
			}
		}
		_ = r.appendLocked(ctx, Entry{Turn: v.Turn, Type: EntryConversationReplaced, Messages: v.Messages})

	case agent.MessageEnd:
		rec := Inference{Attempt: v.Attempt}
		if v.Inference != nil {
			if v.Inference.Client != nil {
				rec.Model = v.Inference.Client.Model().ID
			}
			rec.System = v.Inference.System
			for _, t := range v.Inference.Tools {
				rec.Tools = append(rec.Tools, t.Schema.Name)
			}
		}
		if v.Response != nil {
			// The model is on the response, which pkg/ai stamps with the
			// client that was called — so an inference a hook routed elsewhere
			// is recorded where it went, and a session may name more than one.
			rec.Model = v.Response.Model
			rec.Usage = v.Response.Usage
			rec.StopReason = v.Response.StopReason
		}
		if v.Err != nil {
			rec.Error = v.Err.Error()
		}
		_ = r.appendLocked(ctx, Entry{Turn: v.Turn, Type: EntryInference, Inference: &rec})

	case agent.ToolEnd:
		// The readable part of what the model was told, from the same pair of
		// functions, so the record cannot come to disagree with the
		// conversation about what a call said.
		run := ToolRun{
			ID: v.ID, Name: v.Name, Args: v.Args,
			Content: agent.ResultText(v.Result, v.Err), IsError: v.Err != nil,
		}
		_ = r.appendLocked(ctx, Entry{Turn: v.Turn, Type: EntryToolRun, ToolRun: &run})

	case agent.TurnEnd:
		rec := Outcome{StopReason: v.StopReason, Usage: v.Usage}
		if v.Err != nil {
			rec.Err = v.Err.Error()
		}
		_ = r.appendLocked(ctx, Entry{Turn: v.Turn, Type: EntryOutcome, Outcome: &rec})
	}
}

// Append persists one typed entry. Turn is relative to the current agent run;
// the session applies the restored exchange offset exactly once.
func (r *Session) Append(ctx context.Context, entry Entry) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.appendLocked(ctx, entry)
}

func (r *Session) appendLocked(ctx context.Context, entry Entry) error {
	if r.closed {
		return ErrClosed
	}
	if r.err != nil {
		return r.err
	}
	if !entry.payload() {
		return fmt.Errorf("session: %q entry has no payload", entry.Type)
	}
	if entry.Type == EntryCustom {
		if entry.Custom.Kind == "" {
			return fmt.Errorf("session: custom event kind is required")
		}
		if len(entry.Custom.Data) > 0 && !json.Valid(entry.Custom.Data) {
			return fmt.Errorf("session: custom event data is not valid JSON")
		}
	}
	entry.Turn += r.turnsBefore
	if entry.At.IsZero() {
		entry.At = time.Now().UTC()
	}
	if err := r.store.Append(ctx, r.id, entry); err != nil {
		r.err = err
		return err
	}
	return nil
}

var ErrClosed = errors.New("session: closed")

// Close releases the opened session after any in-flight append.
func (r *Session) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	if r.release == nil {
		return nil
	}
	return r.release()
}

// Err reports the persistence failure, or ErrClosed after resource retirement.
func (r *Session) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return r.err
	}
	if r.closed {
		return ErrClosed
	}
	return nil
}

// ID reports which session this records into.
func (r *Session) ID() string { return r.id }
