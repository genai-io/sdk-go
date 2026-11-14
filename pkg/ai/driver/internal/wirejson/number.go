// Package wirejson adapts dynamic JSON values for provider serializers.
package wirejson

import "encoding/json"

// Object copies JSON maps and slices, preserving standard-library Numbers at
// provider boundaries. The providers' private JSON forks otherwise treat those
// numbers as strings. Other strings must retain their original JSON type.
func Object(src map[string]any) map[string]any {
	if src == nil {
		return nil
	}
	out := make(map[string]any, len(src))
	for k, v := range src {
		out[k] = value(v)
	}
	return out
}

func value(v any) any {
	switch v := v.(type) {
	case json.Number:
		return number(v)
	case map[string]any:
		return Object(v)
	case []any:
		if v == nil {
			return v
		}
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = value(item)
		}
		return out
	default:
		return v
	}
}

// MarshalJSON delegates validation to the standard encoder rather than trusting
// raw bytes or rounding through float64. Invalid Numbers remain encoding errors.
type number json.Number

func (n number) MarshalJSON() ([]byte, error) {
	return json.Marshal(json.Number(n))
}
