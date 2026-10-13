package jsonl

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
)

// trimTornTail removes an incomplete tail and preserves a complete record without its newline.
func trimTornTail(path string) (err error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	// Truncating: a close that fails is a truncate that may not have landed.
	defer func() { err = errors.Join(err, f.Close()) }()

	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return err
	}
	var last [1]byte
	if _, err := f.ReadAt(last[:], size-1); err != nil {
		return err
	}
	if last[0] == '\n' {
		return nil
	}
	for window := int64(8 * 1024); ; window *= 4 {
		if window > size {
			window = size
		}
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return err
		}
		i := bytes.LastIndexByte(buf, '\n')
		if i >= 0 || window == size {
			if json.Valid(buf[i+1:]) {
				_, err := f.WriteAt([]byte{'\n'}, size)
				return err
			}
			return f.Truncate(size - window + int64(i) + 1)
		}
	}
}

// lastSeq recovers the number to carry on from by reading the end of the
// entries file. The file is the record, so this is the one count that cannot
// be wrong — meta.json can be behind, and after a crash it will be. It reads a
// block from the tail rather than the whole file, growing it only for an entry
// too long to fit.
func lastSeq(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	defer func() { _ = f.Close() }() // read-only: nothing is lost by closing it

	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return 0, err
	}
	for window := int64(8 * 1024); ; window *= 4 {
		if window > size {
			window = size
		}
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return 0, err
		}
		// Back over whole lines until one parses: the count carries on from
		// where the readable part ended.
		trimmed := bytes.TrimRight(buf, "\n")
		complete := buf[len(buf)-1] == '\n'
		for {
			i := bytes.LastIndexByte(trimmed, '\n')
			if i < 0 {
				break
			}
			if seq, ok := seqOf(trimmed[i+1:]); ok {
				return seq, nil
			}
			if complete {
				return 0, fmt.Errorf("jsonl: corrupt final record")
			}
			trimmed = trimmed[:i]
			complete = true
		}
		if window == size { // the whole file is one line, or none
			seq, ok := seqOf(trimmed)
			if !ok && complete {
				return 0, fmt.Errorf("jsonl: corrupt final record")
			}
			return seq, nil
		}
	}
}

func seqOf(line []byte) (int64, bool) {
	var e struct {
		Seq int64 `json:"seq"`
	}
	if json.Unmarshal(line, &e) != nil {
		return 0, false
	}
	return e.Seq, true
}
