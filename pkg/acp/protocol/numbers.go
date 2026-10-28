package protocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// projectNumbers is a type adapter for jsonschema-go v0.4.3, whose type
// checks classify json.Number as string. Only schema-described paths are
// visited for numeric conversion. Unconstrained numeric leaves are carried as
// opaque RawMessages, preventing validator arithmetic even in nonmatching union
// branches. This is a validation-only projection, never the wire value.
func projectNumbers(v any, schemas []*jsonschema.Schema, defs map[string]*jsonschema.Schema) (any, error) {
	switch v.(type) {
	case json.Number, map[string]any, []any:
	default:
		return v, nil
	}
	var expanded []*jsonschema.Schema
	seen := map[*jsonschema.Schema]bool{}
	var expand func(*jsonschema.Schema)
	expand = func(s *jsonschema.Schema) {
		if s == nil || seen[s] {
			return
		}
		seen[s] = true
		expanded = append(expanded, s)
		if strings.HasPrefix(s.Ref, "#/$defs/") {
			expand(defs[strings.TrimPrefix(s.Ref, "#/$defs/")])
		}
		for _, q := range s.AllOf {
			expand(q)
		}
		for _, a := range [][]*jsonschema.Schema{s.AnyOf, s.OneOf} {
			selected := false
			if fields, ok := v.(map[string]any); ok {
				for _, q := range a {
					for k, p := range q.Properties {
						if p.Const != nil {
							if tag, ok := (*p.Const).(string); ok && fields[k] == tag {
								expand(q)
								selected = true
							}
						}
					}
				}
			}
			if !selected {
				for _, q := range a {
					expand(q)
				}
			}
		}
	}
	for _, s := range schemas {
		expand(s)
	}
	switch x := v.(type) {
	case json.Number:
		integer, number, typed := false, false, false
		for _, s := range expanded {
			typed = typed || s.Enum != nil || s.Const != nil
			types := append([]string{s.Type}, s.Types...)
			for _, t := range types {
				typed = typed || t != ""
				integer = integer || t == "integer"
				number = number || t == "number"
			}
		}
		if integer {
			c, err := normalizeInteger(string(x))
			if err != nil {
				return nil, err
			}
			if strings.HasPrefix(c, "-") {
				return strconv.ParseInt(c, 10, 64)
			}
			return strconv.ParseUint(c, 10, 64)
		}
		if number {
			f, err := strconv.ParseFloat(string(x), 64)
			if err != nil || math.IsInf(f, 0) {
				return nil, errors.New("protocol: constrained number outside float64 range")
			}
			return f, nil
		}
		if typed { // A numeric token must not masquerade as a schema string.
			return float64(0.5), nil
		}
		return json.RawMessage(x), nil
	case map[string]any:
		for k, value := range x {
			var children []*jsonschema.Schema
			for _, s := range expanded {
				if child := s.Properties[k]; child != nil {
					children = append(children, child)
				}
			}
			p, err := projectNumbers(value, children, defs)
			if err != nil {
				return nil, err
			}
			x[k] = p
		}
	case []any:
		var items []*jsonschema.Schema
		for _, s := range expanded {
			if s.Items != nil {
				items = append(items, s.Items)
			}
		}
		for i, value := range x {
			p, err := projectNumbers(value, items, defs)
			if err != nil {
				return nil, err
			}
			x[i] = p
		}
	}
	return v, nil
}

// normalizeInteger normalizes a valid JSON number to an exact, at-most-20-digit
// integer spelling. Exponents are saturated before any expansion; no big-number
// arithmetic or allocation proportional to the exponent is performed.
func normalizeInteger(s string) (string, error) {
	original := s
	if !json.Valid([]byte(s)) || len(s) == 0 || s[0] == '"' {
		return "", errors.New("protocol: expected JSON integer")
	}
	negative := strings.HasPrefix(s, "-")
	if negative {
		s = s[1:]
	}
	mantissa, exponent := s, ""
	if i := strings.IndexAny(s, "eE"); i >= 0 {
		mantissa, exponent = s[:i], s[i+1:]
	}
	fractional := 0
	if i := strings.IndexByte(mantissa, '.'); i >= 0 {
		fractional = len(mantissa) - i - 1
		mantissa = mantissa[:i] + mantissa[i+1:]
	}
	digits := strings.TrimLeft(mantissa, "0")
	if digits == "" {
		return "0", nil
	}
	// Saturating at input length + 32 is sufficient to distinguish every
	// representable integer from a fractional or overflowing nonzero value.
	limit := len(original) + 32
	exp := 0
	minus := strings.HasPrefix(exponent, "-")
	exponent = strings.TrimLeft(exponent, "+-")
	for _, c := range exponent {
		if exp > limit/10 {
			exp = limit
			break
		}
		exp = exp*10 + int(c-'0')
		if exp > limit {
			exp = limit
			break
		}
	}
	if minus {
		exp = -exp
	}
	shift := exp - fractional
	trimmed := strings.TrimRight(digits, "0")
	shift += len(digits) - len(trimmed)
	digits = trimmed
	if shift < 0 {
		return "", errors.New("protocol: fractional value for integer")
	}
	if shift > 20 || len(digits) > 20-shift {
		return "", errors.New("protocol: integer outside 64-bit range")
	}
	digits += strings.Repeat("0", shift)
	if negative {
		return "-" + digits, nil
	}
	return digits, nil
}

// unmarshalExact preserves normal custom codecs while adapting native integer
// fields and containers to JSON Schema's mathematically integral spellings.
func unmarshalExact(b []byte, dst any) error {
	if _, ok := dst.(json.Unmarshaler); ok {
		return json.Unmarshal(b, dst)
	}
	v := reflect.ValueOf(dst).Elem()
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		c, err := normalizeInteger(string(bytes.TrimSpace(b)))
		if err != nil {
			return err
		}
		n, err := strconv.ParseInt(c, 10, v.Type().Bits())
		if err != nil {
			return err
		}
		v.SetInt(n)
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		c, err := normalizeInteger(string(bytes.TrimSpace(b)))
		if err != nil {
			return err
		}
		n, err := strconv.ParseUint(c, 10, v.Type().Bits())
		if err != nil {
			return err
		}
		v.SetUint(n)
		return nil
	case reflect.Slice:
		if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
			v.SetZero()
			return nil
		}
		var items []json.RawMessage
		if err := json.Unmarshal(b, &items); err != nil {
			return err
		}
		p := reflect.MakeSlice(v.Type(), len(items), len(items))
		for i, item := range items {
			if err := unmarshalExact(item, p.Index(i).Addr().Interface()); err != nil {
				return err
			}
		}
		v.Set(p)
		return nil
	case reflect.Map:
		if v.Type().Key().Kind() != reflect.String {
			return json.Unmarshal(b, dst)
		}
		if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
			v.SetZero()
			return nil
		}
		var items map[string]json.RawMessage
		if err := json.Unmarshal(b, &items); err != nil {
			return err
		}
		p := reflect.MakeMapWithSize(v.Type(), len(items))
		for k, item := range items {
			value := reflect.New(v.Type().Elem())
			if err := unmarshalExact(item, value.Interface()); err != nil {
				return err
			}
			key := reflect.ValueOf(k).Convert(v.Type().Key())
			p.SetMapIndex(key, value.Elem())
		}
		v.Set(p)
		return nil
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("protocol decode: %w", err)
	}
	return nil
}
