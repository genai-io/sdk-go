package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
)

// Schema constrains an answer to a JSON shape.
//
// Without it, getting structured data out of a model means asking for JSON in
// the prompt and scraping the reply — which fails in a long tail of ways that
// all look like the model misbehaving: a markdown fence, a "Sure, here you
// go!" preamble, a trailing paragraph of commentary, a truncated object. Every
// protocol here can constrain generation properly, so none of that is
// necessary.
type Schema struct {
	// Name identifies the shape. Some protocols require one, and it is what
	// appears in an error when the answer does not match.
	Name string `json:"name,omitempty"`

	// Description tells the model what the shape is for. Providers pass it to
	// the model, so it is prompt text, not documentation.
	Description string `json:"description,omitempty"`

	// Definition is the JSON Schema itself, typically a map[string]any. It is
	// the same shape a Tool takes for its parameters.
	Definition any `json:"definition,omitempty"`

	// Strict asks for exact conformance where the protocol distinguishes it
	// from best-effort. Strict mode accepts a narrower subset of JSON Schema —
	// no open-ended objects, every property required — so a schema that works
	// without it may be rejected with it.
	Strict bool `json:"strict,omitempty"`
}

// definitionMap returns the schema as the map shape every protocol's parameter
// type wants, or nil when it is not one.
func (s *Schema) definitionMap() map[string]any {
	if s == nil {
		return nil
	}
	switch def := s.Definition.(type) {
	case map[string]any:
		return def
	case nil:
		return nil
	default:
		// A caller who built the schema from a struct or a typed value gets it
		// re-encoded rather than dropped.
		raw, err := json.Marshal(def)
		if err != nil {
			return nil
		}
		var out map[string]any
		if json.Unmarshal(raw, &out) != nil {
			return nil
		}
		return out
	}
}

// schemaName is what to send for a protocol that requires a name.
func (s *Schema) schemaName() string {
	if s == nil || s.Name == "" {
		return "response"
	}
	return s.Name
}

// Unmarshal decodes the answer into v.
//
// It tolerates what a model actually returns rather than only what it should:
// a bare JSON value, one wrapped in a markdown fence, or one preceded by
// prose. A natively constrained answer needs none of that leniency, but an
// answer produced by SimulateSchema does, and a caller should not have to know
// which path produced the response they are holding.
func (r *Response) Unmarshal(v any) error {
	if r == nil {
		return fmt.Errorf("llm: no response to decode")
	}
	raw, ok := ExtractJSON(r.Content)
	if !ok {
		return fmt.Errorf("llm: response contains no JSON value: %s", truncate(r.Content, 200))
	}
	if err := json.Unmarshal([]byte(raw), v); err != nil {
		return fmt.Errorf("llm: decoding response: %w", err)
	}
	return nil
}

// Parse decodes a completion into T, threading the call's error through so a
// typed answer is one statement:
//
//	person, err := llm.Parse[Person](client.Complete(ctx, prompt, opts))
func Parse[T any](resp *Response, err error) (T, error) {
	var out T
	if err != nil {
		return out, err
	}
	return out, resp.Unmarshal(&out)
}

// ExtractJSON finds the JSON value in a model's answer.
//
// It tries the whole string first, since a constrained answer is exactly that.
// Failing which it strips a markdown fence, and failing that it scans for the
// first balanced object or array — tracking string literals and escapes, so a
// brace inside a quoted value does not end the scan early.
func ExtractJSON(s string) (string, bool) {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return "", false
	}
	if json.Valid([]byte(trimmed)) {
		return trimmed, true
	}
	if fenced, ok := stripFence(trimmed); ok && json.Valid([]byte(fenced)) {
		return fenced, true
	}
	if scanned, ok := scanBalanced(trimmed); ok {
		return scanned, true
	}
	return "", false
}

// stripFence unwraps a ```json … ``` block.
func stripFence(s string) (string, bool) {
	if !strings.HasPrefix(s, "```") {
		return "", false
	}
	rest := s[3:]
	if i := strings.IndexByte(rest, '\n'); i >= 0 {
		rest = rest[i+1:] // drop the language tag
	}
	end := strings.LastIndex(rest, "```")
	if end < 0 {
		return "", false
	}
	return strings.TrimSpace(rest[:end]), true
}

// scanBalanced returns the first balanced object or array, ignoring braces
// that sit inside string literals.
func scanBalanced(s string) (string, bool) {
	start := strings.IndexAny(s, "{[")
	if start < 0 {
		return "", false
	}
	open := s[start]
	close := byte('}')
	if open == '[' {
		close = ']'
	}

	depth, inString, escaped := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case escaped:
			escaped = false
		case c == '\\' && inString:
			escaped = true
		case c == '"':
			inString = !inString
		case inString:
		case c == open:
			depth++
		case c == close:
			depth--
			if depth == 0 {
				candidate := s[start : i+1]
				return candidate, json.Valid([]byte(candidate))
			}
		}
	}
	return "", false
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// SimulateSchema asks for a shape in the prompt, for a model that cannot
// constrain generation to one.
//
// It is opt-in, and lossy in a way native constraining is not: the model is
// merely instructed, so it can still return prose, a fence, or a near-miss.
// Response.Unmarshal is written to survive the common shapes of that, but a
// caller who needs a guarantee should use a model that offers one rather than
// this. A caller who does not opt in gets a clear error naming the model,
// which is more useful than a quietly weaker guarantee.
func SimulateSchema() Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, p *Prompt, opts Options) iter.Seq2[Delta, error] {
			if opts.Schema == nil {
				return next(ctx, p, opts)
			}
			instructed := *p
			instructed.System = strings.TrimSpace(p.System + "\n\n" + schemaInstructions(opts.Schema))
			// Clearing it is what tells the driver, and Validate, that the
			// shape is being asked for in words rather than enforced.
			relaxed := opts
			relaxed.Schema = nil
			return next(ctx, &instructed, relaxed)
		}
	}
}

// schemaInstructions renders the shape as prompt text.
func schemaInstructions(s *Schema) string {
	var b strings.Builder
	b.WriteString("Respond with a single JSON value and nothing else — no explanation, no markdown fence.")
	if s.Description != "" {
		b.WriteString("\nIt should be: ")
		b.WriteString(s.Description)
	}
	if def := s.definitionMap(); def != nil {
		if raw, err := json.MarshalIndent(def, "", "  "); err == nil {
			b.WriteString("\n\nIt must match this JSON Schema:\n")
			b.Write(raw)
		}
	}
	return b.String()
}
