package session

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/genai-io/sdk-go/pkg/ai"
)

func backend(t testing.TB, sqlite bool) *Store {
	t.Helper()
	var s *Store
	var err error
	if sqlite {
		s, err = OpenSQLite(filepath.Join(t.TempDir(), "sessions.db"))
	} else {
		s, err = Open(t.TempDir())
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	return s
}
func must(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func collect(t testing.TB, s *Store, id string) []Record {
	t.Helper()
	var records []Record
	for r, err := range s.Records(context.Background(), id) {
		must(t, err)
		records = append(records, r)
	}
	return records
}

func TestBackendsSharedBehavior(t *testing.T) {
	for _, sqlite := range []bool{false, true} {
		t.Run(fmt.Sprintf("sqlite=%t", sqlite), func(t *testing.T) {
			s := backend(t, sqlite)
			ctx := context.Background()
			if _, err := s.Stat(ctx, "missing"); !errors.Is(err, ErrNotFound) {
				t.Fatal(err)
			}
			if _, err := s.Load(ctx, "missing"); !errors.Is(err, ErrNotFound) {
				t.Fatal(err)
			}
			w, err := s.Writer(ctx, "a")
			must(t, err)
			t.Cleanup(func() { must(t, w.Close()) })
			info, err := s.Stat(ctx, "a")
			must(t, err)
			if info.Size != 0 || info.ModTime.IsZero() {
				t.Fatal(info)
			}
			if got, err := s.Load(ctx, "a"); err != nil || len(got) != 0 {
				t.Fatalf("%v %v", got, err)
			}
			if _, err := s.Writer(ctx, "a"); !errors.Is(err, ErrBusy) {
				t.Fatal(err)
			}
			if err := s.Delete(ctx, "a"); !errors.Is(err, ErrBusy) {
				t.Fatal(err)
			}
			a, b, c := ai.UserMessage("first"), ai.UserMessage("second"), ai.UserMessage("edited")
			for _, msgs := range [][]ai.Message{{a, b}, {a, c}, nil, {b}} {
				must(t, w.Save(msgs))
			}
			must(t, w.Append("metadata", map[string]any{"nested": []any{1, "<&>", true}}))
			must(t, w.Sync())
			records := collect(t, s, "a")
			if len(records) != 7 {
				t.Fatalf("records=%d", len(records))
			}
			before, err := s.Stat(ctx, "a")
			must(t, err)
			if before.Size <= 0 || before.ModTime.Before(info.ModTime) {
				t.Fatal(before)
			}
			must(t, w.Save([]ai.Message{b}))
			after, err := s.Stat(ctx, "a")
			must(t, err)
			if !reflect.DeepEqual(before, after) {
				t.Fatalf("no-op save changes metadata: %v %v", before, after)
			}
			if err := w.Append("bad", make(chan int)); err == nil {
				t.Fatal("invalid codec accepted")
			}
			if err := w.Append("bad", json.RawMessage(`{`)); err == nil {
				t.Fatal("invalid JSON accepted")
			}
			if got := collect(t, s, "a"); !reflect.DeepEqual(got, records) {
				t.Fatal("invalid codec changed records")
			}
			must(t, s.Fork(ctx, "a", "b")) // source lease may stay live
			if !reflect.DeepEqual(records, collect(t, s, "b")) {
				t.Fatal("fork changed historical records")
			}
			if err := s.Fork(ctx, "a", "b"); !errors.Is(err, fs.ErrExist) {
				t.Fatal(err)
			}
			list, err := s.List(ctx)
			must(t, err)
			if len(list) != 2 || list[0].ID != "b" {
				t.Fatal(list)
			}
			must(t, w.Close())
			if err := w.Save([]ai.Message{b}); !errors.Is(err, os.ErrClosed) {
				t.Fatal(err)
			}
			if err := w.Sync(); !errors.Is(err, os.ErrClosed) {
				t.Fatal(err)
			}
			w, err = s.Writer(ctx, "a")
			must(t, err)
			must(t, w.Save([]ai.Message{b}))
			if !reflect.DeepEqual(records, collect(t, s, "a")) {
				t.Fatal("reopen appended duplicates")
			}
			must(t, w.Close())
			cancelCtx, cancel := context.WithCancel(ctx)
			cancel()
			if _, err := s.Writer(cancelCtx, "cancel"); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if err := s.Fork(cancelCtx, "a", "cancel"); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if err := s.Delete(cancelCtx, "a"); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if _, err := s.Stat(cancelCtx, "a"); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if _, err := s.List(cancelCtx); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			must(t, s.Delete(ctx, "b"))
		})
	}
}

func TestSQLiteRollbackAndSnapshot(t *testing.T) {
	s := backend(t, true)
	ctx := context.Background()
	w, err := s.Writer(ctx, "a")
	must(t, err)
	defer w.Close()
	old := []ai.Message{ai.UserMessage("old")}
	next := []ai.Message{ai.UserMessage("new"), ai.UserMessage("suffix")}
	must(t, w.Save(old))
	before, err := s.Stat(ctx, "a")
	must(t, err)
	// Fail after the truncate and first replacement were inserted.
	_, err = s.sql.db.Exec(`CREATE TRIGGER fail_suffix BEFORE INSERT ON session_events WHEN NEW.seq=3 BEGIN SELECT RAISE(ABORT,'injected failure'); END`)
	must(t, err)
	if err = w.Save(next); err == nil {
		t.Fatal("trigger did not fail save")
	}
	after, err := s.Stat(ctx, "a")
	must(t, err)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("failed transaction changed metadata")
	}
	got, err := s.Load(ctx, "a")
	must(t, err)
	if len(got) != 1 || got[0].Text() != "old" {
		t.Fatal(got)
	}
	_, err = s.sql.db.Exec("DROP TRIGGER fail_suffix")
	must(t, err)
	// Iterate an existing snapshot while a commit occurs on another connection.
	count := 0
	for _, err := range s.Records(ctx, "a") {
		must(t, err)
		count++
		must(t, w.Save(next))
	}
	if count != 1 {
		t.Fatalf("snapshot grew to %d", count)
	}
	got, err = s.Load(ctx, "a")
	must(t, err)
	if len(got) != 2 || got[0].Text() != "new" || got[1].Text() != "suffix" {
		t.Fatal(got)
	}
	if len(collect(t, s, "a")) != 4 {
		t.Fatal("cache advanced on failed Save")
	}
	// An uncommitted suffix is invisible to other readers.
	tx, err := s.sql.db.Begin()
	must(t, err)
	_, err = tx.Exec("DELETE FROM session_events WHERE session_id='a'")
	must(t, err)
	if len(collect(t, s, "a")) != 4 {
		t.Fatal("dirty read")
	}
	must(t, tx.Rollback())
}

func TestSQLiteCorruption(t *testing.T) {
	for _, data := range []string{`{bad}` + "\n", `{"type":"message"}` + "\n", `{"type":"truncate","keep":-1}` + "\n", `{"type":"truncate","keep":99}` + "\n", `{"type":"future"}` + "\n", `{"type":"app","name":"x"}`} {
		t.Run(data, func(t *testing.T) {
			s := backend(t, true)
			w, err := s.Writer(context.Background(), "a")
			must(t, err)
			must(t, w.Append("a", 1))
			must(t, w.Close())
			_, err = s.sql.db.Exec("UPDATE session_events SET record=?", []byte(data))
			must(t, err)
			if _, err = s.Load(context.Background(), "a"); err == nil {
				t.Fatal("corruption accepted")
			}
			if w, err = s.Writer(context.Background(), "a"); err == nil {
				w.Close()
				t.Fatal("corrupt writer accepted")
			}
		})
	}
	t.Run("sequence gap", func(t *testing.T) {
		s := backend(t, true)
		w, err := s.Writer(context.Background(), "a")
		must(t, err)
		must(t, w.Append("a", 1))
		must(t, w.Append("b", 2))
		must(t, w.Close())
		_, err = s.sql.db.Exec("DELETE FROM session_events WHERE seq=0")
		must(t, err)
		if _, err = s.Load(context.Background(), "a"); err == nil {
			t.Fatal("gap accepted")
		}
	})
}

func TestSQLiteAliasesCloseAndConfiguration(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "sessions.db")
	s, err := OpenSQLite(path)
	must(t, err)
	defer s.Close()
	alias := filepath.Join(dir, "alias.db")
	must(t, os.Symlink(path, alias))
	other, err := OpenSQLite(alias)
	must(t, err)
	defer other.Close()
	w, err := s.Writer(ctx, "a")
	must(t, err)
	if err = s.Close(); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	if _, err = other.Writer(ctx, "a"); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	if err = other.Delete(ctx, "a"); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	w2, err := other.Writer(ctx, "independent")
	must(t, err)
	must(t, w2.Append("x", 1))
	must(t, w2.Close())
	must(t, w.Close())
	must(t, s.Close())
	must(t, s.Close())
	if _, err = s.Writer(ctx, "a"); !errors.Is(err, os.ErrClosed) {
		t.Fatal(err)
	}
	if _, err = s.Load(ctx, "a"); !errors.Is(err, os.ErrClosed) {
		t.Fatal(err)
	}
	w, err = other.Writer(ctx, "a")
	must(t, err)
	must(t, w.Close())
	for _, name := range []string{path, path + "-wal", path + "-shm"} {
		info, err := os.Stat(name)
		must(t, err)
		if info.Mode().Perm() != 0o600 {
			t.Fatalf("%s: %v", name, info.Mode())
		}
	}
	// Check several simultaneously held connections: PRAGMAs are per connection.
	for range 3 {
		conn, err := other.sql.db.Conn(ctx)
		must(t, err)
		defer conn.Close()
		for pragma, want := range map[string]int{"synchronous": 2, "foreign_keys": 1, "busy_timeout": 5000, "fullfsync": 1, "checkpoint_fullfsync": 1} {
			var got int
			must(t, conn.QueryRowContext(ctx, "PRAGMA "+pragma).Scan(&got))
			if got != want {
				t.Fatalf("%s=%d", pragma, got)
			}
		}
	}
}

func TestSQLiteSchemaVersion(t *testing.T) {
	path := filepath.Join(t.TempDir(), "db")
	s, err := OpenSQLite(path)
	must(t, err)
	_, err = s.sql.db.Exec("PRAGMA user_version=99")
	must(t, err)
	must(t, s.Close())
	if s, err = OpenSQLite(path); err == nil {
		s.Close()
		t.Fatal("future version accepted")
	}
}

func TestSQLiteRejectsUnrelatedSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unrelated.db")
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	// Only sqlite_ is reserved. LIKE 'sqlite_%' would also hide this valid table.
	_, err = db.Exec("CREATE TABLE sqlitecache (value TEXT); INSERT INTO sqlitecache VALUES ('keep')")
	must(t, err)
	if s, err := OpenSQLite(path); err == nil {
		s.Close()
		t.Fatal("adopted unrelated database")
	}
	var version, sessionTables int
	must(t, db.QueryRow("PRAGMA user_version").Scan(&version))
	must(t, db.QueryRow("SELECT count(*) FROM sqlite_master WHERE name IN ('sessions','session_events')").Scan(&sessionTables))
	var value string
	must(t, db.QueryRow("SELECT value FROM sqlitecache").Scan(&value))
	if version != 0 || sessionTables != 0 || value != "keep" {
		t.Fatalf("modified unrelated schema: version=%d tables=%d value=%q", version, sessionTables, value)
	}
}

func TestSQLiteCrossProcessLease(t *testing.T) {
	if path := os.Getenv("SDK_SQLITE_LEASE_DB"); path != "" {
		s, err := OpenSQLite(path)
		if err != nil {
			t.Fatal(err)
		}
		w, err := s.Writer(context.Background(), "held")
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		defer s.Close()
		fmt.Println("LEASE_READY")
		select {} // parent kills us to exercise crash release
	}
	path := filepath.Join(t.TempDir(), "db")
	s, err := OpenSQLite(path)
	must(t, err)
	defer s.Close()
	cmd := exec.Command(os.Args[0], "-test.run=^TestSQLiteCrossProcessLease$")
	cmd.Env = append(os.Environ(), "SDK_SQLITE_LEASE_DB="+path)
	out, err := cmd.StdoutPipe()
	must(t, err)
	must(t, cmd.Start())
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()
	// Read the readiness marker without depending on sleeps.
	marker, err := bufio.NewReader(out).ReadString('\n')
	must(t, err)
	if strings.TrimSpace(marker) != "LEASE_READY" {
		t.Fatalf("child: %q", marker)
	}
	if _, err := s.Writer(context.Background(), "held"); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	if err := s.Delete(context.Background(), "held"); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	must(t, cmd.Process.Kill())
	_ = cmd.Wait()
	w, err := s.Writer(context.Background(), "held")
	must(t, err)
	must(t, w.Close())
}

func TestSQLiteConcurrentIndependentSessions(t *testing.T) {
	s := backend(t, true)
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			w, err := s.Writer(context.Background(), fmt.Sprint(i))
			if err != nil {
				t.Error(err)
				return
			}
			defer w.Close()
			for j := range 10 {
				if err := w.Append("sequence", j); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
	for i := range 8 {
		if n := len(collect(t, s, fmt.Sprint(i))); n != 10 {
			t.Fatal(n)
		}
	}
}
