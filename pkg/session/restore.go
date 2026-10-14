package session

import (
	"context"
	"errors"
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
		case EntryConversationReplaced:
			msgs = slices.Clone(entry.Messages)
		case EntryMessage:
			msgs = append(msgs, *entry.Message)
		case EntryOutcome:
		}
		// EntryCustom and EntryInference and EntryToolRun are not the
		// conversation: they explain it. A fold walks past them.
	}
	return msgs, turns, nil
}

// Create starts a new opened session. Close it when its writer is retired.
func Create(ctx context.Context, store Store, meta Meta) (*Session, error) {
	meta, err := store.Create(ctx, meta)
	if err != nil {
		return nil, err
	}
	release, err := store.Claim(ctx, meta.ID)
	if err != nil {
		return nil, err
	}
	return &Session{store: store, id: meta.ID, release: release}, nil
}

// Open restores an existing session while holding its exclusive writer claim.
func Open(ctx context.Context, store Store, id string) (*Session, []ai.Message, error) {
	release, err := store.Claim(ctx, id)
	if err != nil {
		return nil, nil, err
	}
	msgs, turns, err := fold(ctx, store, id)
	if err != nil {
		return nil, nil, errors.Join(err, release())
	}

	rec := &Session{store: store, id: id, release: release}
	// The agent numbers turns from one every time it runs, because what came
	// back from storage was someone else's counting. The session is the one
	// place that knows both numbers, so it is where they are reconciled —
	// without which a resumed session holds two exchanges both called turn 1.
	rec.turnsBefore = turns
	// And this is the history the agent is about to be handed. It will
	// announce it as replaced, correctly and unavoidably; recording it would
	// write a copy of what was just read, once per resume, forever.
	// Own the snapshot so edits to returned history remain visible.
	rec.restored = slices.Clone(msgs)
	for i := range rec.restored {
		rec.restored[i].Content = msgs[i].Content.Clone()
	}
	return rec, msgs, nil
}

// Read restores the native conversation without opening a writer lease.
func Read(ctx context.Context, store Store, id string) ([]ai.Message, error) {
	messages, _, err := fold(ctx, store, id)
	return messages, err
}
