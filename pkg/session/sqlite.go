package session

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// sqliteStore keeps writer leases separate from short SQLite transactions.
// A writer never holds a database connection between operations.
type sqliteStore struct {
	db      *sql.DB
	dir     string
	mu      sync.Mutex
	writers int
	closed  bool
}

// OpenSQLite opens an experimental local SQLite session database. It uses WAL,
// FULL synchronous commits, fullfsync (where supported), foreign keys and a
// five-second busy timeout on every connection. Do not use it on a network
// filesystem. Symlink aliases are
// canonicalized for writer leases; hard-link aliases are not supported.
func OpenSQLite(databasePath string) (*Store, error) {
	path, err := filepath.Abs(databasePath)
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	// Create securely before SQLite opens the file (and creates its WAL).
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	err = errors.Join(f.Chmod(0o600), f.Close())
	if err != nil {
		return nil, err
	}
	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return nil, err
	}
	dir := path + ".locks"
	if err = os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	if err = os.Chmod(dir, 0o700); err != nil {
		return nil, err
	}
	u := url.URL{Scheme: "file", Path: path}
	q := u.Query()
	// Go's os.File.Sync uses F_FULLFSYNC on macOS. synchronous=FULL alone
	// only selects fsync in SQLite there; match the stronger durable boundary.
	for _, pragma := range []string{"busy_timeout(5000)", "foreign_keys(ON)", "synchronous(FULL)", "fullfsync(ON)", "checkpoint_fullfsync(ON)"} {
		q.Add("_pragma", pragma)
	}
	u.RawQuery = q.Encode()
	db, err := sql.Open("sqlite", u.String())
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(8)
	fail := func(err error) (*Store, error) { return nil, errors.Join(err, db.Close()) }
	var mode string
	if err = db.QueryRow("PRAGMA journal_mode=WAL").Scan(&mode); err != nil {
		return fail(err)
	}
	if mode != "wal" {
		return fail(fmt.Errorf("session: SQLite journal mode %q, want WAL", mode))
	}
	// Serialize schema initialization across processes before checking the version.
	conn, err := db.Conn(context.Background())
	if err != nil {
		return fail(err)
	}
	defer conn.Close()
	if _, err = conn.ExecContext(context.Background(), "BEGIN IMMEDIATE"); err != nil {
		return fail(err)
	}
	defer conn.ExecContext(context.Background(), "ROLLBACK")
	var version int
	if err = conn.QueryRowContext(context.Background(), "PRAGMA user_version").Scan(&version); err != nil {
		return fail(err)
	}
	if version != 0 && version != 1 {
		return fail(fmt.Errorf("session: unsupported SQLite schema version %d", version))
	}
	if version == 0 {
		// Refuse to adopt an unrelated, unversioned database.
		var count int
		if err = conn.QueryRowContext(context.Background(), "SELECT count(*) FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'").Scan(&count); err != nil {
			return fail(err)
		}
		if count != 0 {
			return fail(errors.New("session: unrecognized SQLite schema"))
		}
		_, err = conn.ExecContext(context.Background(), `
CREATE TABLE sessions (id TEXT PRIMARY KEY, mod_time INTEGER NOT NULL, size INTEGER NOT NULL DEFAULT 0 CHECK(size >= 0), next_seq INTEGER NOT NULL DEFAULT 0 CHECK(next_seq >= 0));
CREATE TABLE session_events (session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, seq INTEGER NOT NULL CHECK(seq >= 0), record BLOB NOT NULL, PRIMARY KEY(session_id, seq));
PRAGMA user_version=1;`)
		if err != nil {
			return fail(err)
		}
	}
	// Validate expected columns even when the version claims to be supported.
	rows, err := conn.QueryContext(context.Background(), "SELECT s.id,s.mod_time,s.size,s.next_seq,e.session_id,e.seq,e.record FROM sessions s LEFT JOIN session_events e ON e.session_id=s.id LIMIT 0")
	if err != nil {
		return fail(err)
	}
	if err = rows.Close(); err != nil {
		return fail(err)
	}
	if _, err = conn.ExecContext(context.Background(), "COMMIT"); err != nil {
		return fail(err)
	}
	return &Store{sql: &sqliteStore{db: db, dir: dir}}, nil
}

// Close releases database resources. It returns ErrBusy without closing while
// this store owns writers. Closing a directory store is a no-op.
func (s *Store) Close() error {
	if s.sql == nil {
		return nil
	}
	b := s.sql
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if b.writers != 0 {
		return ErrBusy
	}
	b.closed = true
	return b.db.Close()
}

// Stat returns a session's metadata without replaying its records. Size is a
// change fingerprint, not the SQLite database's physical disk usage.
func (s *Store) Stat(ctx context.Context, id string) (Info, error) {
	if s.sql != nil {
		if err := s.sql.check(ctx, id); err != nil {
			return Info{}, err
		}
		var info Info
		var ns int64
		err := s.sql.db.QueryRowContext(ctx, "SELECT id,mod_time,size FROM sessions WHERE id=?", id).Scan(&info.ID, &ns, &info.Size)
		if errors.Is(err, sql.ErrNoRows) {
			err = ErrNotFound
		}
		info.ModTime = time.Unix(0, ns)
		return info, err
	}
	path, err := s.path(ctx, id)
	if err != nil {
		return Info{}, err
	}
	info, err := os.Stat(path)
	if err != nil {
		return Info{}, notFound(id, err)
	}
	return Info{ID: id, ModTime: info.ModTime(), Size: info.Size()}, nil
}

// Sync makes prior appends durable. SQLite appends already commit with FULL
// synchronous durability; for JSONL this calls fsync.
func (w *Writer) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.sync()
}

func (w *Writer) sync() error {
	if w.closed {
		return os.ErrClosed
	}
	if w.err != nil {
		return w.err
	}
	if w.sql != nil {
		return nil
	}
	if w.f == nil {
		return os.ErrClosed
	}
	w.err = w.f.Sync()
	return w.err
}

func (b *sqliteStore) check(ctx context.Context, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validate(id); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return os.ErrClosed
	}
	return nil
}

func (b *sqliteStore) records(ctx context.Context, id string) iter.Seq2[Record, error] {
	return func(yield func(Record, error) bool) {
		if err := b.check(ctx, id); err != nil {
			yield(Record{}, err)
			return
		}
		// One SELECT is one fixed read snapshot, including the existence check.
		rows, err := b.db.QueryContext(ctx, `SELECT e.seq,e.record,s.next_seq FROM sessions s LEFT JOIN session_events e ON e.session_id=s.id WHERE s.id=? ORDER BY e.seq`, id)
		if err != nil {
			yield(Record{}, err)
			return
		}
		defer rows.Close()
		found := false
		var seq, total int64
		for rows.Next() {
			found = true
			var n sql.NullInt64
			var data []byte
			if err = rows.Scan(&n, &data, &total); err != nil {
				yield(Record{}, err)
				return
			}
			if !n.Valid {
				break
			}
			if n.Int64 != seq || len(data) == 0 || data[len(data)-1] != '\n' || bytes.Count(data, []byte{'\n'}) != 1 {
				yield(Record{}, fmt.Errorf("session: %s: corrupt SQLite event %d", id, seq))
				return
			}
			seq++
			r, err := decodeRecord(data, id, seq)
			if err == nil {
				err = ctx.Err()
			}
			if !yield(r, err) || err != nil {
				return
			}
		}
		if err = rows.Err(); err != nil {
			yield(Record{}, err)
		} else if !found {
			yield(Record{}, ErrNotFound)
		} else if seq != total {
			yield(Record{}, fmt.Errorf("session: %s: incomplete SQLite event sequence", id))
		}
	}
}

func (b *sqliteStore) list(ctx context.Context) ([]Info, error) {
	if err := b.check(ctx, "list"); err != nil {
		return nil, err
	}
	rows, err := b.db.QueryContext(ctx, "SELECT id,mod_time,size FROM sessions ORDER BY mod_time DESC,id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Info
	for rows.Next() {
		var info Info
		var ns int64
		if err := rows.Scan(&info.ID, &ns, &info.Size); err != nil {
			return nil, err
		}
		info.ModTime = time.Unix(0, ns)
		out = append(out, info)
	}
	return out, rows.Err()
}

func (b *sqliteStore) writer(ctx context.Context, id string) (_ *Writer, err error) {
	if err = b.check(ctx, id); err != nil {
		return nil, err
	}
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, os.ErrClosed
	}
	b.writers++ // reserve before Close can race acquisition
	b.mu.Unlock()
	defer func() {
		if err != nil {
			b.mu.Lock()
			b.writers--
			b.mu.Unlock()
		}
	}()
	l, err := lock(filepath.Join(b.dir, id))
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, l.unlock())
		}
	}()
	if _, err = b.db.ExecContext(ctx, "INSERT INTO sessions(id,mod_time) VALUES(?,?) ON CONFLICT(id) DO NOTHING", id, time.Now().UnixNano()); err != nil {
		return nil, err
	}
	msgs, err := conversation(b.records(ctx, id))
	if err != nil {
		return nil, err
	}
	encs, err := encode(msgs)
	if err != nil {
		return nil, err
	}
	return &Writer{sql: b, id: id, lock: l, saved: encs}, nil
}

func (b *sqliteStore) append(id string, data []byte) error {
	tx, err := b.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	// First statement takes the write lock, avoiding deferred read/write upgrades.
	if _, err = tx.Exec("UPDATE sessions SET size=size+?,mod_time=? WHERE id=?", len(data), time.Now().UnixNano(), id); err != nil {
		return err
	}
	var seq int64
	if err = tx.QueryRow("SELECT next_seq FROM sessions WHERE id=?", id).Scan(&seq); err != nil {
		return err
	}
	for _, line := range bytes.SplitAfter(data, []byte{'\n'}) {
		if len(line) == 0 {
			continue
		}
		if _, err = tx.Exec("INSERT INTO session_events(session_id,seq,record) VALUES(?,?,?)", id, seq, line); err != nil {
			return err
		}
		seq++
	}
	if _, err = tx.Exec("UPDATE sessions SET next_seq=? WHERE id=?", seq, id); err != nil {
		return err
	}
	return tx.Commit()
}

func (b *sqliteStore) delete(ctx context.Context, id string) (err error) {
	if err = b.check(ctx, id); err != nil {
		return err
	}
	l, err := lock(filepath.Join(b.dir, id))
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, l.unlock()) }()
	result, err := b.db.ExecContext(ctx, "DELETE FROM sessions WHERE id=?", id)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err == nil && n == 0 {
		return ErrNotFound
	}
	return err
}

func (b *sqliteStore) fork(ctx context.Context, from, to string) (err error) {
	if err = b.check(ctx, from); err != nil {
		return err
	}
	if err = validate(to); err != nil {
		return err
	}
	l, err := lock(filepath.Join(b.dir, to))
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, l.unlock()) }()
	tx, err := b.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	// Acquire a database write snapshot before any reads. A source writer's
	// lease is irrelevant: its commits fall entirely before or after this copy.
	if _, err = tx.ExecContext(ctx, "UPDATE sessions SET size=size WHERE id=?", to); err != nil {
		return err
	}
	var exists int
	if err = tx.QueryRowContext(ctx, "SELECT count(*) FROM sessions WHERE id=?", to).Scan(&exists); err != nil {
		return err
	}
	if exists != 0 {
		return fmt.Errorf("session: forking to %s: %w", to, fs.ErrExist)
	}
	result, err := tx.ExecContext(ctx, "INSERT INTO sessions(id,mod_time,size,next_seq) SELECT ?,?,size,next_seq FROM sessions WHERE id=?", to, time.Now().UnixNano(), from)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return ErrNotFound
	}
	if _, err = tx.ExecContext(ctx, "INSERT INTO session_events(session_id,seq,record) SELECT ?,seq,record FROM session_events WHERE session_id=?", to, from); err != nil {
		return err
	}
	return tx.Commit()
}
