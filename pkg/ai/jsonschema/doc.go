// Package jsonschema turns a Go type into a JSON Schema a model provider will
// accept, and checks a model's answer against one.
//
//	definition := jsonschema.For[Person]()
//	err := jsonschema.Check(definition, decoded)
//
// # Not JSON Schema in general
//
// The target is narrower than the specification, and deliberately so. A schema
// here has to satisfy three parties at once — the Go type it came from, the
// provider that has to accept it, and the model that has to fill it in — and
// the provider is the one a general-purpose generator ignores.
//
// OpenAI's strict structured output is the tightest of them, so it sets the
// shape:
//
//   - Every field is in required, optional ones included. Optionality is the
//     ["T","null"] type union, not an absent name in the list.
//   - Every object carries additionalProperties: false.
//   - Nothing is a boolean schema. `true` means "anything here", which strict
//     mode rejects, so a field that would need one — an interface, an `any` —
//     is refused at construction instead.
//
// A type that marshals to something other than its own fields is described by
// what it marshals to: time.Time is a date-time string, not an object of
// unexported fields. Getting that wrong produces a schema that rejects the JSON
// its own Go type writes.
//
// # The tags
//
// A tag key is the JSON Schema keyword it sets. There is no grammar to learn
// and nothing to quote — Go's own struct-tag convention does the splitting, so
// a description containing a comma is just a description containing a comma:
//
//	type Order struct {
//		Item     string `json:"item" description:"what to order, one line"`
//		Priority string `json:"priority" enum:"low|medium|high"`
//		Quantity int    `json:"quantity" description:"how many" minimum:"1" maximum:"99"`
//	}
//
// Enum members are pipe-separated, since a tag value is a string and a JSON
// array is not, and they are typed by the field they constrain — a numeric
// field gets numbers.
//
// The keywords are description, enum, format, pattern, minimum, maximum,
// multipleOf, minLength, maxLength, minItems and maxItems: the intersection
// the providers document as supported, so a tag cannot produce a schema an
// endpoint refuses.
//
// A key one edit from a keyword — enums, descrption — is refused rather than
// ignored, and so is a jsonschema tag. Go drops an unrecognised key without a
// word, which for these keys means losing the only thing the field was
// annotated with. A key that is nobody's near miss is left alone: db, validate
// and the rest belong to other tools.
//
// # Failing early
//
// Anything wrong with a type or a tag panics, the way a bad pattern panics in
// regexp.MustCompile: it is a mistake in the caller's own source, it surfaces
// when the tool or schema is constructed rather than mid-conversation, and the
// alternatives are worse — a schema the endpoint rejects, or one that quietly
// permits anything.
//
// Check is the other direction and does not panic. A model produced the value,
// so a wrong one is expected traffic, and the error is phrased for the model
// that has to correct it: "priority must be one of low, medium or high", named
// by the path the caller wrote, not by a location inside the schema.
//
// # Where things live
//
//	derive.go  a Go type and its ai tags becoming a schema
//	check.go   a decoded value measured against one
package jsonschema
