// Package protocol provides wire values for stable ACP v1 (schema-v1.26.0).
// Schema and metadata are reproduced unmodified from agentclientprotocol/
// agent-client-protocol at SourceCommit under Apache-2.0; see LICENSE.schema.
//
// Codecs validate the official JSON Schema, preserve unknown object properties,
// and distinguish absent, null, and present optional fields. They intentionally
// do not implement the Rust-specific x-deserialize-default-on-error and
// x-deserialize-skip-invalid-items recovery hints: invalid input returns an error.
// Union Value fields accept only their generated concrete variants. Typed
// integers decode exactly within their Go bounds; typed numbers use finite
// float64 values. Arbitrary RawMessages are never numerically converted.
// Ambiguous untagged inputs are rejected, except empty config-options arrays,
// which decode as Ungrouped. No server or session policy is implemented here.
package protocol

//go:generate go run ../../../internal/acpgen

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/google/jsonschema-go/jsonschema"
)

const (
	WireVersion   = 1
	SchemaVersion = "schema-v1.26.0"
	SourceCommit  = "e2d83349bc9e28a4059ea3acefa1c91c966f74e2"
	SchemaBlob    = "d6f7731bf74b3dbe3b9bbda52957a7637a12e688"
)

//go:embed schema.json
var schemaJSON []byte

// Optional represents an absent field, explicit null, or a value. Null is
// accepted only where the schema permits it. A zero Optional is absent.
type Optional[T any] struct {
	Value   T
	Present bool
	Null    bool
}

// Some constructs a present, non-null value.
func Some[T any](v T) Optional[T] { return Optional[T]{Value: v, Present: true} }

// Null constructs a present null value.
func Null[T any]() Optional[T]      { return Optional[T]{Present: true, Null: true} }
func (v Optional[T]) present() bool { return v.Present }
func (v Optional[T]) MarshalJSON() ([]byte, error) {
	if !v.Present || v.Null {
		return []byte("null"), nil
	}
	return json.Marshal(v.Value)
}
func (v *Optional[T]) UnmarshalJSON(b []byte) error {
	var p Optional[T]
	p.Present = true
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		p.Null = true
	} else if err := unmarshalExact(b, &p.Value); err != nil {
		return err
	}
	*v = p
	return nil
}

func definitionSet(names []string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}
	return set
}

var errVariant = errors.New("protocol: invalid or missing union variant")
var validators sync.Map
var sourceOnce sync.Once
var sourceDefs map[string]json.RawMessage
var errSource error

func source() (map[string]json.RawMessage, error) {
	sourceOnce.Do(func() {
		var d struct {
			Defs map[string]json.RawMessage `json:"$defs"`
		}
		errSource = json.Unmarshal(schemaJSON, &d)
		sourceDefs = d.Defs
	})
	return sourceDefs, errSource
}
func validator(definition string) (*jsonschema.Resolved, error) {
	if v, ok := validators.Load(definition); ok {
		return v.(*jsonschema.Resolved), nil
	}
	defs, err := source()
	if err != nil {
		return nil, err
	}
	parts := strings.Split(definition, "/")
	if len(parts) > 2 || !supportedDefinitions[parts[0]] {
		return nil, fmt.Errorf("protocol: unsupported definition %q", definition)
	}
	b, ok := defs[parts[0]]
	if !ok {
		return nil, fmt.Errorf("protocol: unknown definition %q", definition)
	}
	var n map[string]json.RawMessage
	if err = json.Unmarshal(b, &n); err != nil {
		return nil, err
	}
	if len(parts) == 2 {
		i, e := strconv.Atoi(parts[1])
		if e != nil {
			return nil, e
		}
		a := n["oneOf"]
		if len(a) == 0 {
			a = n["anyOf"]
		}
		var variants []json.RawMessage
		if err = json.Unmarshal(a, &variants); err != nil {
			return nil, err
		}
		if i < 0 || i >= len(variants) {
			return nil, errVariant
		}
		common := map[string]json.RawMessage{}
		for k, v := range n {
			if k != "oneOf" && k != "anyOf" {
				common[k] = v
			}
		}
		c, err := json.Marshal(common)
		if err != nil {
			return nil, err
		}
		n = map[string]json.RawMessage{"allOf": json.RawMessage("[" + string(c) + "," + string(variants[i]) + "]")}
	}
	db, err := json.Marshal(defs)
	if err != nil {
		return nil, err
	}
	n["$defs"] = db
	n["$schema"] = json.RawMessage(`"https://json-schema.org/draft/2020-12/schema"`)
	b, err = json.Marshal(n)
	if err != nil {
		return nil, err
	}
	var s jsonschema.Schema
	if err = json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	r, err := s.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("protocol schema %s: %w", definition, err)
	}
	actual, _ := validators.LoadOrStore(definition, r)
	return actual.(*jsonschema.Resolved), nil
}

// ValidateJSON validates a named definition from the pinned official schema.
// It performs no network access or default insertion; unconstrained JSON
// numbers retain their original representation.
func ValidateJSON(definition string, data []byte) error {
	r, err := validator(definition)
	if err != nil {
		return err
	}
	var v any
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err = d.Decode(&v); err != nil {
		return err
	}
	var extra any
	if err = d.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("protocol: multiple JSON values")
		}
		return err
	}
	v, err = projectNumbers(v, []*jsonschema.Schema{r.Schema()}, r.Schema().Defs)
	if err != nil {
		return err
	}
	if err = r.Validate(v); err != nil {
		return fmt.Errorf("protocol %s: %w", definition, err)
	}
	return nil
}

func marshalObject(def string, value any) ([]byte, error) {
	v := reflect.ValueOf(value)
	t := v.Type()
	m := map[string]json.RawMessage{}
	extra := v.FieldByName("Extra").Interface().(map[string]json.RawMessage)
	for k, b := range extra {
		m[k] = b
	}
	for i := 0; i < v.NumField(); i++ {
		f := t.Field(i)
		k := f.Tag.Get("json")
		if k == "-" {
			continue
		}
		fv := v.Field(i).Interface()
		delete(m, k)
		if o, ok := fv.(interface{ present() bool }); ok && !o.present() {
			continue
		}
		b, err := json.Marshal(fv)
		if err != nil {
			return nil, err
		}
		m[k] = b
	}
	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	if err = ValidateJSON(def, b); err != nil {
		return nil, err
	}
	return b, nil
}
func unmarshalObject(def string, b []byte, dst any) error {
	if err := ValidateJSON(def, b); err != nil {
		return err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	v := reflect.ValueOf(dst).Elem()
	t := v.Type()
	for i := 0; i < v.NumField(); i++ {
		k := t.Field(i).Tag.Get("json")
		if k == "-" {
			continue
		}
		if b, ok := m[k]; ok {
			if err := unmarshalExact(b, v.Field(i).Addr().Interface()); err != nil {
				return err
			}
			delete(m, k)
		}
	}
	v.FieldByName("Extra").Set(reflect.ValueOf(m))
	return nil
}
func marshalUnion(def string, v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	if err = ValidateJSON(def, b); err != nil {
		return nil, err
	}
	return b, nil
}
func selectVariant(def string, b []byte) (int, error) {
	if err := ValidateJSON(def, b); err != nil {
		return 0, err
	}
	defs, err := source()
	if err != nil {
		return 0, err
	}
	var n map[string]json.RawMessage
	if err := json.Unmarshal(defs[def], &n); err != nil {
		return 0, err
	}
	a := n["oneOf"]
	if len(a) == 0 {
		a = n["anyOf"]
	}
	var variants []json.RawMessage
	if err := json.Unmarshal(a, &variants); err != nil {
		return 0, err
	}
	var fields map[string]json.RawMessage
	if len(bytes.TrimSpace(b)) > 0 && bytes.TrimSpace(b)[0] == '{' {
		if err := json.Unmarshal(b, &fields); err != nil {
			return 0, err
		}
	}
	// Honor recognized tags before considering permissive untagged fallbacks.
	for i, v := range variants {
		var q struct {
			Properties map[string]struct {
				Const json.RawMessage `json:"const"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(v, &q); err != nil {
			return 0, err
		}
		for k, p := range q.Properties {
			if len(p.Const) > 0 && bytes.Equal(bytes.TrimSpace(fields[k]), p.Const) {
				if err := ValidateJSON(fmt.Sprintf("%s/%d", def, i), b); err != nil {
					return 0, err
				}
				return i, nil
			}
		}
	}
	selected := -1
	for i := range variants {
		if ValidateJSON(fmt.Sprintf("%s/%d", def, i), b) != nil {
			continue
		}
		if selected >= 0 {
			if bytes.Equal(bytes.TrimSpace(b), []byte("[]")) {
				continue
			}
			return 0, fmt.Errorf("protocol %s: ambiguous untagged union", def)
		}
		selected = i
	}
	if selected < 0 {
		return 0, errVariant
	}
	return selected, nil
}
