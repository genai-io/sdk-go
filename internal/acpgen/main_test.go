package main

import (
	"bytes"
	"os"
	"testing"
)

func TestPins(t *testing.T) {
	for file, hash := range map[string]string{"schema.json": sourcePin.SchemaBlob, "meta.json": sourcePin.MetadataBlob} {
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
