package main

import (
	"bytes"
	"os"
	"testing"
)

func TestPins(t *testing.T) {
	for file, hash := range map[string]string{"schema.json": "d6f7731bf74b3dbe3b9bbda52957a7637a12e688", "meta.json": "b9c67caa417dd2285a31d07309a215a6110bbfaf"} {
		b, err := os.ReadFile("../../pkg/acp/protocol/" + file)
		if err != nil {
			t.Fatal(err)
		}
		if err = checkBlob(b, hash); err != nil {
			t.Fatal(err)
		}
		if err = checkBlob(append(b, ' '), hash); err == nil {
			t.Fatal("modified source accepted")
		}
	}
}

func TestReproducible(t *testing.T) {
	schema, err := os.ReadFile("../../pkg/acp/protocol/schema.json")
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("../../pkg/acp/protocol/types_gen.go")
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		got, err := generate(schema)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatal("generated output differs; run go generate ./pkg/acp/protocol")
		}
	}
}
