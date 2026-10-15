package jsonl

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"time"
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
	return readTail(f, size, func(buf []byte) (bool, error) {
		i := bytes.LastIndexByte(buf, '\n')
		if i < 0 && int64(len(buf)) < size {
			return false, nil
		}
		if json.Valid(buf[i+1:]) {
			_, err := f.WriteAt([]byte{'\n'}, size)
			return true, err
		}
		return true, f.Truncate(size - int64(len(buf)) + int64(i) + 1)
	})
}

// lastPosition reads the committed sequence and event time from the log tail.
// It grows the tail window only when a record does not fit.
func lastPosition(path string) (logPosition, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return logPosition{}, nil
		}
		return logPosition{}, err
	}
	defer func() { _ = f.Close() }() // read-only: nothing is lost by closing it

	size, err := f.Seek(0, io.SeekEnd)
	if err != nil || size == 0 {
		return logPosition{}, err
	}
	var position logPosition
	err = readTail(f, size, func(buf []byte) (bool, error) {
		// Back over whole lines until one parses: the count carries on from
		// where the readable part ended.
		trimmed := bytes.TrimRight(buf, "\n")
		complete := buf[len(buf)-1] == '\n'
		for {
			i := bytes.LastIndexByte(trimmed, '\n')
			if i < 0 {
				break
			}
			if p, ok := positionOf(trimmed[i+1:]); ok {
				position = p
				return true, nil
			}
			if complete {
				return true, fmt.Errorf("jsonl: corrupt final record")
			}
			trimmed = trimmed[:i]
			complete = true
		}
		if int64(len(buf)) < size {
			return false, nil
		}
		// The whole file is one line, or none.
		p, ok := positionOf(trimmed)
		if !ok && complete {
			return true, fmt.Errorf("jsonl: corrupt final record")
		}
		position = p
		return true, nil
	})
	return position, err
}

// readTail hands fn the last bytes of a file of the given size, growing the
// window until fn is done or the window is the whole file.
func readTail(f *os.File, size int64, fn func(buf []byte) (done bool, err error)) error {
	for window := int64(8 * 1024); ; window *= 4 {
		window = min(window, size)
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return err
		}
		if done, err := fn(buf); done || err != nil || window == size {
			return err
		}
	}
}

// logPosition is the ordering and event time of the last committed entry.
type logPosition struct {
	Seq int64     `json:"seq"`
	At  time.Time `json:"at"`
}

func positionOf(line []byte) (logPosition, bool) {
	var p logPosition
	if json.Unmarshal(line, &p) != nil {
		return logPosition{}, false
	}
	return p, true
}
