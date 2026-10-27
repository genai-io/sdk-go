package acp

import (
	"bufio"
	"bytes"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestFrameBoundary(t *testing.T) {
	for _, tc := range []struct {
		name string
		size int
	}{
		{"buffered", 16},
		{"fragmented", 8192},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := []byte(`{"jsonrpc":"2.0","method":"update","params":{"text":"` + strings.Repeat("x", tc.size) + `"}}`)
			frame, err := readFrame(bufio.NewReaderSize(bytes.NewReader(append(payload, '\n')), 128), len(payload))
			if err != nil || !bytes.Equal(frame, payload) {
				t.Fatalf("exact-size frame: %v", err)
			}
			if _, err := decode(frame); err != nil {
				t.Fatal(err)
			}
			_, err = readFrame(bufio.NewReaderSize(bytes.NewReader(append(payload, '\n')), 128), len(payload)-1)
			if !errors.Is(err, ErrFrameTooLarge) {
				t.Fatalf("oversize frame: %v", err)
			}
		})
	}
}

func TestDecodeOwnsFrame(t *testing.T) {
	for _, raw := range []string{
		`{"jsonrpc":"2.0","id":"1","method":"echo","params":{"text":"hello"}}`,
		`{"jsonrpc":"2.0","id":"1","result":{"text":"hello"}}`,
		`{"jsonrpc":"2.0","id":"1","error":{"code":-32001,"message":"denied","data":{"text":"hello"}}}`,
	} {
		want, err := decode([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		frame, err := readFrame(bufio.NewReader(strings.NewReader(raw+"\n")), len(raw))
		if err != nil {
			t.Fatal(err)
		}
		got, err := decode(frame)
		if err != nil {
			t.Fatal(err)
		}
		clear(frame)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("decoded message retained frame storage: got %+v, want %+v", got, want)
		}
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
