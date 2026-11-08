package session

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
)

var (
	// ErrMode means legacy records and committed batches would be mixed.
	ErrMode = errors.New("session: incompatible log mode")
	// ErrConflict means a batch ID was reused with different content.
	ErrConflict = errors.New("session: batch ID conflict")
	// ErrTail means the expected committed prefix is no longer current.
	ErrTail = errors.New("session: committed tail mismatch")
)

// Tail identifies a validated prefix. Its zero value denotes an empty log.
// Hash binds the entire prefix; Sequence counts batches and Count cumulatively
// counts all logical records, including truncates and app records (not the
// current conversation length). The hash is integrity detection, not a MAC.
type Tail struct {
	Sequence uint64 `json:"sequence"`
	Count    uint64 `json:"count"`
	Hash     string `json:"hash"`
}

// Batch is an atomic group of native session records. ID must be stable across
// retries. Callers supply record timestamps; Commit never changes the payload.
// Expected must equal the current tail for a new ID. For a retry, the original
// Expected and identically JSON-encoded Records must be supplied.
type Batch struct {
	ID       string   `json:"id"`
	Expected Tail     `json:"previous"`
	Records  []Record `json:"records"`
}

// Receipt identifies a complete validated batch. A receipt returned by Commit
// additionally acknowledges successful file Sync, including on retry.
type Receipt struct {
	ID   string
	Tail Tail
}

// CommittedBatch is a validated batch and its resulting tail.
type CommittedBatch struct {
	Batch   Batch
	Receipt Receipt
}

type envelope struct {
	Version int   `json:"batch_version"`
	Batch   Batch `json:"batch"`
	Tail    Tail  `json:"tail"`
}

type batchState struct {
	mode     string
	tail     Tail
	seen     map[string]Tail
	messages int
}

func batchHash(b Batch, t Tail) (string, error) {
	data, err := json.Marshal(struct {
		Version         int
		Batch           Batch
		Sequence, Count uint64
	}{1, b, t.Sequence, t.Count})
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:]), nil
}

func validRecord(r Record) error {
	if r.Type != TypeMessage && r.Type != TypeTruncate && r.Type != TypeApp {
		return fmt.Errorf("session: record type %q is from a newer version", r.Type)
	}
	if r.Type == "" || r.Type == TypeMessage && r.Message == nil || r.Keep < 0 || r.Type == TypeApp && r.Name == "" {
		return errors.New("session: not a session record")
	}
	return nil
}

func (s *batchState) accept(e envelope) error {
	if s.mode == "legacy" {
		return ErrMode
	}
	if e.Version != 1 {
		return fmt.Errorf("session: unsupported batch version %d", e.Version)
	}
	b := e.Batch
	if b.ID == "" || len(b.Records) == 0 {
		return errors.New("session: empty batch ID or records")
	}
	if b.Expected != s.tail || e.Tail.Sequence != s.tail.Sequence+1 || e.Tail.Count != s.tail.Count+uint64(len(b.Records)) || e.Tail.Sequence == 0 || e.Tail.Count < s.tail.Count {
		return errors.New("session: invalid batch chain")
	}
	hash, err := batchHash(b, e.Tail)
	if err != nil {
		return err
	}
	if hash != e.Tail.Hash {
		return errors.New("session: batch integrity mismatch")
	}
	if _, ok := s.seen[b.ID]; ok {
		return ErrConflict
	}
	messages := s.messages
	for _, r := range b.Records {
		if err := validRecord(r); err != nil {
			return err
		}
		switch r.Type {
		case TypeMessage:
			messages++
		case TypeTruncate:
			if r.Keep > messages {
				return errors.New("session: invalid batch truncate")
			}
			messages = r.Keep
		case TypeApp:
		default:
			return fmt.Errorf("session: unsupported record type %q", r.Type)
		}
	}
	if s.seen == nil {
		s.seen = make(map[string]Tail)
	}
	s.seen[b.ID] = e.Tail
	s.mode = "batch"
	s.tail = e.Tail
	s.messages = messages
	return nil
}

func (s *batchState) decode(line []byte) ([]Record, *CommittedBatch, error) {
	var tag map[string]json.RawMessage
	if err := json.Unmarshal(line, &tag); err != nil {
		return nil, nil, err
	}
	if _, ok := tag["batch_version"]; ok {
		var e envelope
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		if err := d.Decode(&e); err != nil {
			return nil, nil, err
		}
		if err := s.accept(e); err != nil {
			return nil, nil, err
		}
		return e.Batch.Records, &CommittedBatch{e.Batch, Receipt{e.Batch.ID, e.Tail}}, nil
	}
	if s.mode == "batch" {
		return nil, nil, ErrMode
	}
	var r Record
	if err := json.Unmarshal(line, &r); err != nil {
		return nil, nil, err
	}
	if err := validRecord(r); err != nil {
		return nil, nil, err
	}
	s.mode = "legacy"
	return []Record{r}, nil, nil
}

// Tail returns the writer's last validated committed prefix. Legacy logs have
// no committed tail and return ErrMode.
func (w *Writer) Tail() (Tail, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return Tail{}, os.ErrClosed
	}
	if w.err != nil {
		return Tail{}, w.err
	}
	if w.batches.mode == "legacy" {
		return Tail{}, ErrMode
	}
	return w.batches.tail, nil
}

// Commit appends a whole batch, then Syncs before acknowledging it. Matching
// retries are checked before Expected, and Sync again even after reopening.
// Legacy Save/Append cannot be mixed with Commit in a nonempty session.
//
// Unlocked readers can observe a complete batch before Sync returns: validated
// visibility is not a durability acknowledgment. Cancellation or I/O failure
// after writing starts is ambiguous, never a rollback. Such a writer is sticky
// failed: close, reopen, and retry the same batch to resolve the outcome.
// Inputs must not be mutated concurrently with this call.
func (w *Writer) Commit(ctx context.Context, b Batch) (Receipt, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return Receipt{}, os.ErrClosed
	}
	if w.err != nil {
		return Receipt{}, w.err
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	if w.batches.mode == "legacy" {
		return Receipt{}, ErrMode
	}
	// Round-trip to own the payload and use the same encoding on reopen.
	raw, err := json.Marshal(b)
	if err != nil {
		return Receipt{}, err
	}
	var owned Batch
	if err = json.Unmarshal(raw, &owned); err != nil {
		return Receipt{}, err
	}
	b = owned
	if old, ok := w.batches.seen[b.ID]; ok {
		fingerprint, hashErr := batchHash(b, old)
		if hashErr != nil {
			return Receipt{}, hashErr
		}
		if fingerprint != old.Hash {
			return Receipt{}, ErrConflict
		}
		if err = w.sync(); err == nil {
			err = ctx.Err()
			w.err = err
		}
		if err != nil {
			return Receipt{}, err
		}
		return Receipt{b.ID, old}, nil
	}
	if b.Expected != w.batches.tail {
		return Receipt{}, ErrTail
	}
	e := envelope{Version: 1, Batch: b, Tail: Tail{Sequence: w.batches.tail.Sequence + 1, Count: w.batches.tail.Count + uint64(len(b.Records))}}
	e.Tail.Hash, err = batchHash(b, e.Tail)
	if err != nil {
		return Receipt{}, err
	}
	// Validate without modifying the live index until the write succeeds.
	trial := batchState{mode: w.batches.mode, tail: w.batches.tail, messages: w.batches.messages}
	if err = trial.accept(e); err != nil {
		return Receipt{}, err
	}
	data, err := json.Marshal(e)
	if err != nil {
		return Receipt{}, err
	}
	if err = ctx.Err(); err != nil {
		return Receipt{}, err
	}
	data = append(data, '\n')
	n, err := w.f.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	if err != nil {
		w.err = err
		return Receipt{}, err
	}
	if err = w.sync(); err == nil {
		err = ctx.Err()
		w.err = err
	}
	if err != nil {
		return Receipt{}, err
	}
	_ = w.batches.accept(e)
	return Receipt{b.ID, e.Tail}, nil
}

// Batches returns whole validated batches, never physical framing records.
// Legacy logs return ErrMode. An incomplete final line is ignored. A receipt
// here describes integrity, not whether another process has finished Sync.
func (s *Store) Batches(ctx context.Context, id string) iter.Seq2[CommittedBatch, error] {
	return func(yield func(CommittedBatch, error) bool) {
		path, err := s.path(ctx, id)
		if err != nil {
			yield(CommittedBatch{}, err)
			return
		}
		f, err := os.Open(path)
		if err != nil {
			yield(CommittedBatch{}, notFound(id, err))
			return
		}
		defer func() { _ = f.Close() }()
		state := &batchState{}
		for group, err := range scanLog(f, id, nil, state) {
			if err == nil {
				err = ctx.Err()
			}
			if err == nil && state.mode == "legacy" {
				err = ErrMode
			}
			if err != nil {
				yield(CommittedBatch{}, err)
				return
			}
			if !yield(*group, nil) {
				return
			}
		}
	}
}
