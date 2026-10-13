package session

import (
	"context"
	"fmt"
	"slices"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// fold reads a session once and answers both questions asked of it: what the
// conversation is, and how many exchanges it has held. The second is not
// derivable from the first — a snapshot resets the conversation and not the
// history — and reading the entries twice to learn it would be reading them
// twice.
func fold(ctx context.Context, store Store, id string) ([]ai.Message, int, error) {
	var msgs []ai.Message
	var turns int
	for entry, err := range store.Entries(ctx, id) {
		if err != nil {
			return nil, 0, err
		}
		if !entry.payload() {
			return nil, 0, fmt.Errorf("session: %s entry %d says %q and carries nothing",
				id, entry.Seq, entry.Type)
		}
		turns = max(turns, entry.Turn)
		switch entry.Type {
		case EntrySnapshot:
			msgs = slices.Clone(entry.Snapshot)
		case EntryMessage:
			msgs = append(msgs, *entry.Message)
		case EntryOutcome:
		}
		// EntryCustom and EntryInference and EntryToolRun are not the
		// conversation: they explain it. A fold walks past them.
	}
	return msgs, turns, nil
}

// Open starts a session or resumes one, returning a Writer to feed events to
// and the conversation folded back out of what was stored. Pass an empty id for
// a new session; the id it was given is on the Writer.
//
//	rec, history, _ := session.Open(ctx, store, "")   // or an existing id
//	a.SetMessages(history)
//	for e, err := range a.Run(ctx, ai.UserMessage(line)) {
//	    rec.Handle(ctx, e)
//	    render(e)
//	}
func Open(ctx context.Context, store Store, id string) (*Writer, []ai.Message, error) {
	if id == "" {
		meta, err := store.Create(ctx, Meta{})
		if err != nil {
			return nil, nil, err
		}
		return newWriter(store, meta.ID), nil, nil
	}

	if _, err := store.Meta(ctx, id); err != nil {
		return nil, nil, err
	}
	msgs, turns, err := fold(ctx, store, id)
	if err != nil {
		return nil, nil, err
	}

	rec := newWriter(store, id)
	// The agent numbers turns from one every time it runs, because what came
	// back from storage was someone else's counting. The session is the one
	// place that knows both numbers, so it is where they are reconciled —
	// without which a resumed session holds two exchanges both called turn 1.
	rec.turnsBefore = turns
	// And this is the history the agent is about to be handed. It will
	// announce it as replaced, correctly and unavoidably; recording it would
	// write a copy of what was just read, once per resume, forever.
	rec.restored = msgs
	return rec, msgs, nil
}
