// Package protocol provides wire values for stable ACP v1 (schema-v1.26.0).
// Schema and metadata are reproduced unmodified from agentclientprotocol/
// agent-client-protocol at SourceCommit under Apache-2.0; see LICENSE.schema.
// Go names normalize initialisms (ID, MCP, HTTP, URI); union variants follow
// <Union><Case>Variant. JSON fields and schema definition names are unchanged.
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

type schemaVariant struct {
	definition string
	schema     json.RawMessage
	tags       map[string]json.RawMessage
}

type schemaDefinition struct {
	document map[string]json.RawMessage
	variants []schemaVariant
}

type definitionEntry struct {
	once  sync.Once
	value *schemaDefinition
	err   error
}

type validatorEntry struct {
	once  sync.Once
	value *jsonschema.Resolved
	err   error
}

var errVariant = errors.New("protocol: invalid or missing union variant")
var definitions sync.Map
var validators sync.Map
var sourceOnce sync.Once
var sourceDefs map[string]json.RawMessage
var sourceDefsJSON json.RawMessage
var errSource error

func loadDefinitions() (map[string]json.RawMessage, error) {
	sourceOnce.Do(func() {
		var d struct {
			Defs map[string]json.RawMessage `json:"$defs"`
		}
		if errSource = json.Unmarshal(schemaJSON, &d); errSource != nil {
			return
		}
		sourceDefs = d.Defs
		sourceDefsJSON, errSource = json.Marshal(d.Defs)
	})
	return sourceDefs, errSource
}
func lookupDefinition(name string) (*schemaDefinition, error) {
	if !supportedDefinitions[name] {
		return nil, fmt.Errorf("protocol: unsupported definition %q", name)
	}
	value, ok := definitions.Load(name)
	if !ok {
		value, _ = definitions.LoadOrStore(name, &definitionEntry{})
	}
	entry := value.(*definitionEntry)
	entry.once.Do(func() { entry.value, entry.err = parseDefinition(name) })
	return entry.value, entry.err
}

func parseDefinition(name string) (*schemaDefinition, error) {
	defs, err := loadDefinitions()
	if err != nil {
		return nil, err
	}
	b, ok := defs[name]
	if !ok {
		return nil, fmt.Errorf("protocol: unknown definition %q", name)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(b, &document); err != nil {
		return nil, err
	}
	a := document["oneOf"]
	if len(a) == 0 {
		a = document["anyOf"]
	}
	var branches []json.RawMessage
	if len(a) != 0 {
		if err := json.Unmarshal(a, &branches); err != nil {
			return nil, err
		}
	}
	definition := &schemaDefinition{document: document}
	for i, b := range branches {
		var branch struct {
			Properties map[string]struct {
				Const json.RawMessage `json:"const"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(b, &branch); err != nil {
			return nil, err
		}
		tags := make(map[string]json.RawMessage)
		for key, property := range branch.Properties {
			if len(property.Const) != 0 {
				tags[key] = property.Const
			}
		}
		definition.variants = append(definition.variants, schemaVariant{
			definition: fmt.Sprintf("%s/%d", name, i), schema: b, tags: tags,
		})
	}
	return definition, nil
}

func lookupValidator(definition string) (*jsonschema.Resolved, error) {
	value, ok := validators.Load(definition)
	if !ok {
		parts := strings.Split(definition, "/")
		if len(parts) > 2 || !supportedDefinitions[parts[0]] {
			return nil, fmt.Errorf("protocol: unsupported definition %q", definition)
		}
		value, _ = validators.LoadOrStore(definition, &validatorEntry{})
	}
	entry := value.(*validatorEntry)
	entry.once.Do(func() { entry.value, entry.err = compileValidator(definition) })
	if entry.err != nil {
		validators.CompareAndDelete(definition, entry)
	}
	return entry.value, entry.err
}

func compileValidator(definition string) (*jsonschema.Resolved, error) {
	parts := strings.Split(definition, "/")
	metadata, err := lookupDefinition(parts[0])
	if err != nil {
		return nil, err
	}
	n := make(map[string]json.RawMessage, len(metadata.document)+2)
	for key, value := range metadata.document {
		n[key] = value
	}
	if len(parts) == 2 {
		i, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, err
		}
		if i < 0 || i >= len(metadata.variants) {
			return nil, errVariant
		}
		delete(n, "oneOf")
		delete(n, "anyOf")
		common, err := json.Marshal(n)
		if err != nil {
			return nil, err
		}
		n = map[string]json.RawMessage{"allOf": json.RawMessage("[" + string(common) + "," + string(metadata.variants[i].schema) + "]")}
	}
	n["$defs"] = sourceDefsJSON
	n["$schema"] = json.RawMessage(`"https://json-schema.org/draft/2020-12/schema"`)
	b, err := json.Marshal(n)
	if err != nil {
		return nil, err
	}
	var s jsonschema.Schema
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	r, err := s.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("protocol schema %s: %w", definition, err)
	}
	return r, nil
}

// ValidateJSON validates a named definition from the pinned official schema.
// It performs no network access or default insertion; unconstrained JSON
// numbers retain their original representation.
func ValidateJSON(definition string, data []byte) error {
	r, err := lookupValidator(definition)
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
func marshalValue(def string, v any) ([]byte, error) {
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
	metadata, err := lookupDefinition(def)
	if err != nil {
		return 0, err
	}
	var fields map[string]json.RawMessage
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) > 0 && trimmed[0] == '{' {
		if err := json.Unmarshal(b, &fields); err != nil {
			return 0, err
		}
	}
	// Honor recognized tags before considering permissive untagged fallbacks.
	for i, variant := range metadata.variants {
		for key, tag := range variant.tags {
			if bytes.Equal(bytes.TrimSpace(fields[key]), tag) {
				if err := ValidateJSON(variant.definition, b); err != nil {
					return 0, err
				}
				return i, nil
			}
		}
	}
	selected := -1
	for i, variant := range metadata.variants {
		if ValidateJSON(variant.definition, b) != nil {
			continue
		}
		if selected >= 0 {
			return 0, fmt.Errorf("protocol %s: ambiguous untagged union", def)
		}
		selected = i
	}
	if selected < 0 {
		return 0, errVariant
	}
	return selected, nil
}
