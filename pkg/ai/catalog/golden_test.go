package catalog

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite the golden table in testdata")

// TestTableIsUnchanged pins the entire resolved table, byte for byte.
//
// The rows are written in shorthand — presets, shared lines, and an Infer that
// fills what a row leaves out — so tidying the table can move a figure without
// changing anything a reader would notice. A diff here is not a failure by
// itself; it is a claim to check and then record with -update.
func TestTableIsUnchanged(t *testing.T) {
	path := filepath.Join("testdata", "models.json")

	got, err := json.MarshalIndent(Models(), "", "  ")
	if err != nil {
		t.Fatalf("marshalling the table: %v", err)
	}
	got = append(got, '\n')

	if *update {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}

	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading the golden table (run: go test ./pkg/ai/catalog -update): %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("the resolved model table changed.\n"+
			"If that was the point, check the diff and record it with:\n"+
			"\tgo test ./pkg/ai/catalog -update\n\n%s", firstDiff(string(want), string(got)))
	}
}

// firstDiff reports the first line that differs, with a little context. A
// whole-table dump would bury the one line that matters.
func firstDiff(want, got string) string {
	wantLines, gotLines := splitLines(want), splitLines(got)
	for i := range max(len(wantLines), len(gotLines)) {
		w, g := at(wantLines, i), at(gotLines, i)
		if w == g {
			continue
		}
		return "first difference at line " + itoa(i+1) + ":\n  want: " + w + "\n  got:  " + g
	}
	return "the tables differ only in trailing content"
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := range len(s) {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func at(lines []string, i int) string {
	if i < len(lines) {
		return lines[i]
	}
	return "(no line)"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
