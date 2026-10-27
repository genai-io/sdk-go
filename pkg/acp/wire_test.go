package acp

import (
	"bufio"
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestFrameBoundary(t *testing.T) {
	payload := []byte(`{"jsonrpc":"2.0","method":"update","params":{"text":"` + strings.Repeat("x", 8192) + `"}}`)
	frame, err := readFrame(bufio.NewReaderSize(bytes.NewReader(append(payload, '\n')), 32), len(payload))
	if err != nil || !bytes.Equal(frame, payload) {
		t.Fatalf("exact-size frame: %v", err)
	}
	if _, err := decode(frame); err != nil {
		t.Fatal(err)
	}
	_, err = readFrame(bufio.NewReaderSize(bytes.NewReader(append(payload, '\n')), 32), len(payload)-1)
	if !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("oversize frame: %v", err)
	}
}

func FuzzDecode(f *testing.F) {
	for _, seed := range []string{
		`{"jsonrpc":"2.0","id":9223372036854775807,"method":"initialize","params":{}}`,
		`{"jsonrpc":"2.0","id":null,"method":"session/new","params":{}}`,
		`{"jsonrpc":"2.0","id":"1","result":null}`,
		`{"jsonrpc":"2.0","id":"1","error":{"code":-32601,"message":"Method not found"}}`,
		`{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s"}}`,
		`{"jsonrpc":"2.0","method":"x","result":null}`,
		`[]`, "\xff", "{",
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, frame []byte) {
		if len(frame) > 64<<10 {
			return
		}
		m, err := decode(frame)
		if err != nil {
			return
		}
		if len(m.id) != 0 {
			if key, err := idKey(m.id); err != nil || key != m.key {
				t.Fatalf("unstable request ID: %v", err)
			}
		}
	})
}
