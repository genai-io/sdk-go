package session

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestDecodeScratchIndependence(t *testing.T) {
	lines := []string{
		`{"type":"app","name":"first","data":{"text":"first payload"}}`,
		`{"type":"app","name":"second","data":{"text":"other payload"}}`,
		`{"type":"message","message":{"role":"user","content":[{"type":"text","text":"first message"}]}}`,
		`{"type":"message","message":{"role":"user","content":[{"type":"text","text":"other message"}]}}`,
		`{"type":"truncate"}`,
	}
	var state batchState
	var records, expected []Record
	for _, line := range lines {
		var want Record
		if err := json.Unmarshal([]byte(line), &want); err != nil {
			t.Fatal(err)
		}
		expected = append(expected, want)
		record, _, err := state.decode([]byte(line))
		if err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
		if !reflect.DeepEqual(state.wire, wireRecord{}) {
			t.Fatal("scratch retains decoded fields")
		}
	}
	// A type error may leave fields decoded before the error was found. Those
	// must also be cleared, without changing previously returned payloads.
	if _, _, err := state.decode([]byte(`{"type":"app","data":{"text":"bad"},"keep":"invalid"}`)); err == nil {
		t.Fatal("accepted malformed record")
	}
	if !reflect.DeepEqual(state.wire, wireRecord{}) {
		t.Fatal("scratch retains fields after error")
	}
	if !reflect.DeepEqual(records, expected) {
		t.Fatalf("retained records changed: got %#v, want %#v", records, expected)
	}
}

func TestWireRecordFields(t *testing.T) {
	record, wire := reflect.TypeFor[Record](), reflect.TypeFor[wireRecord]()
	for i := 0; i < record.NumField(); i++ {
		field := record.Field(i)
		if !field.IsExported() {
			continue
		}
		got, ok := wire.FieldByName(field.Name)
		if !ok || got.Type != field.Type || got.Tag.Get("json") != field.Tag.Get("json") {
			t.Errorf("wireRecord field %s does not match Record", field.Name)
		}
	}
}

func TestDecodeLegacyDiscriminator(t *testing.T) {
	for _, line := range []string{
		`{"type":"app","name":"event","data":{"batch_version":1},"unknown":true}`,
		`{"type":"unknown","type":"app","name":"event"}`,
		`{"type":"app","name":"event","unknown":{"nested":true}}`,
	} {
		var state batchState
		record, batch, err := state.decode([]byte(line))
		if err != nil || batch != nil || record.Type != TypeApp {
			t.Fatalf("decode %s: record=%v batch=%v err=%v", line, record, batch, err)
		}
	}
	for _, line := range []string{
		`{"type":"app","type":"future","name":"event"}`,
		`{"type":"app","name":"event"} {}`,
		`{"type":"app","name":"event",}`,
		`{"type":"app","name":"event","batch_version":null}`,
		`{"type":"app","name":"event","batch_version":2}`,
		`{"type":"app","name":"event","batch_version":"1"}`,
		// Like the strict envelope decoder, the discriminator reserves case
		// variants too; do not treat these as unrelated legacy unknown fields.
		`{"type":"app","name":"event","BATCH_VERSION":1}`,
	} {
		var state batchState
		if _, _, err := state.decode([]byte(line)); err == nil {
			t.Fatalf("accepted %s", line)
		}
	}
}

func TestDecodeBatchDiscriminator(t *testing.T) {
	e := envelope{Version: 1, Batch: appBatch("one", Tail{}), Tail: Tail{Sequence: 1, Count: 2}}
	var err error
	e.Tail.Hash, err = batchHash(e.Batch, e.Tail)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := json.Marshal(e.Batch)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := json.Marshal(e.Tail)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"batch_version"`, `"batch_\u0076ersion"`} {
		for _, line := range []string{
			fmt.Sprintf(`{ %s : 1, "batch":%s,"tail":%s }`, key, batch, tail),
			fmt.Sprintf("{\n\"tail\":%s,\"batch\":%s, %s : 1 }", tail, batch, key),
		} {
			var state batchState
			_, group, err := state.decode([]byte(line))
			if err != nil || group == nil || len(group.Batch.Records) != 2 {
				t.Fatalf("decode %s: %v", line, err)
			}
			if _, _, err := state.decode([]byte(`{"type":"app","name":"event"}`)); !errors.Is(err, ErrMode) {
				t.Fatalf("batch then legacy: %v", err)
			}
			state = batchState{mode: "legacy"}
			if _, _, err := state.decode([]byte(line)); !errors.Is(err, ErrMode) {
				t.Fatalf("legacy then batch: %v", err)
			}
		}
	}
	// The combined legacy decoder must not weaken the strict envelope decoder.
	line := fmt.Sprintf(`{"batch_version":1,"batch":%s,"tail":%s,"unknown":true}`, batch, tail)
	var state batchState
	if _, _, err := state.decode([]byte(line)); err == nil {
		t.Fatal("accepted unknown envelope field")
	}
}
