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
	var messages []ai.Message
	var lastTurn int
	for entry, err := range store.Entries(ctx, id) {
		if err != nil {
			return nil, 0, err
		}
		if !entry.payload() {
			return nil, 0, fmt.Errorf("session: %s entry %d says %q and carries nothing",
				id, entry.Seq, entry.Type)
		}
		lastTurn = max(lastTurn, entry.Turn)
		switch entry.Type {
		case EntryConversationReplaced:
			messages = slices.Clone(entry.Messages)
		case EntryMessage:
			messages = append(messages, *entry.Message)
		}
		// EntryCustom, EntryInference, EntryToolRun and EntryOutcome are not
		// the conversation: they explain it. A fold walks past them.
	}
	return messages, lastTurn, nil
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
	messages, lastTurn, err := fold(ctx, store, id)
	if err != nil {
		return nil, nil, errors.Join(err, release())
	}

	s := &Session{store: store, id: id, release: release}
	// The agent numbers turns from one every time it runs, because what came
	// back from storage was someone else's counting. The session is the one
	// place that knows both numbers, so it is where they are reconciled —
	// without which a resumed session holds two exchanges both called turn 1.
	s.turnOffset = lastTurn
	// And this is the history the agent is about to be handed. It will
	// announce it as replaced, correctly and unavoidably; recording it would
	// write a copy of what was just read, once per resume, forever.
	// Own the snapshot so edits to returned history remain visible.
	s.resumeSnapshot = slices.Clone(messages)
	for i := range s.resumeSnapshot {
		s.resumeSnapshot[i].Content = messages[i].Content.Clone()
	}
	return s, messages, nil
}
