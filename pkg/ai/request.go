package ai

import (
	"maps"
	"reflect"
	"slices"
)

// Request is one model invocation: the conversation, the tools it may call,
// and the settings it runs under.
type Request struct {
	// System is separate from Messages because protocols place it outside
	// ordinary turns.
	System string

	Messages []Message
	Tools    []Tool

	// MaxTokens caps the output. Zero leaves the cap off the wire.
	MaxTokens int

	// Temperature is a pointer because omission and an explicit zero must stay
	// distinct: zero is a useful setting, not an absent one.
	Temperature *float64

	// Effort is the normalized reasoning rung. The model's ladder decides what
	// actually goes on the wire.
	Effort Effort

	// ToolChoice constrains tool use for this turn.
	ToolChoice ToolChoice

	// StopSequences ends generation when the model emits one of these.
	StopSequences []string

	// CacheRetention asks for a prompt-cache lifetime.
	CacheRetention CacheRetention

	// Schema constrains the answer to a JSON shape. Nil leaves the model free
	// to answer in prose.
	Schema *Schema

	// SamplingParams are merged into the request body verbatim, over the named
	// fields above and over the model's own SamplingParams, so a caller can
	// reach a parameter this SDK does not model. Only the OpenAI-family
	// drivers apply them.
	SamplingParams map[string]any

	// ProtocolOptions carries settings only one protocol has, as that driver's
	// own value — anthropic.Options, responses.Options. It is the escape hatch for
	// what the fields above deliberately do not model, so needing one thing a
	// protocol offers does not mean writing a whole driver.
	ProtocolOptions ProtocolOptions
}

// Option sets one field of a Request.
//
//	client := ai.NewClientWithDriver(driver, model, ai.WithEffort(ai.EffortHigh))
//	resp, err := client.Complete(ctx, messages, ai.WithEffort(ai.EffortLow))
type Option func(*Request)

// WithSystem sets the system prompt.
func WithSystem(system string) Option {
	return func(r *Request) { r.System = system }
}

// WithTools offers the model a set of tools. Calling it replaces any tools a
// lower layer set; calling it with none removes them.
func WithTools(tools ...Tool) Option {
	return func(r *Request) { r.Tools = slices.Clone(tools) }
}

// WithMaxTokens caps the output. Zero explicitly leaves the cap off the wire;
// omit the option to use the model's advertised output limit.
func WithMaxTokens(n int) Option {
	return func(r *Request) { r.MaxTokens = n }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) Option {
	return func(r *Request) { r.Temperature = &t }
}

// WithEffort sets the reasoning rung.
func WithEffort(e Effort) Option {
	return func(r *Request) { r.Effort = e }
}

// WithToolChoice constrains tool use for the turn.
func WithToolChoice(c ToolChoice) Option {
	return func(r *Request) { r.ToolChoice = c }
}

// WithForceTool requires the model to call the named tool. Shorthand for
// WithToolChoice(ToolChoiceNamed(name)).
func WithForceTool(name string) Option {
	return WithToolChoice(ToolChoiceNamed(name))
}

// WithStopSequences ends generation at any of these strings. Calling it with
// none clears a lower layer's list.
func WithStopSequences(stop ...string) Option {
	return func(r *Request) { r.StopSequences = slices.Clone(stop) }
}

// WithCacheRetention asks the provider for a prompt-cache lifetime.
func WithCacheRetention(c CacheRetention) Option {
	return func(r *Request) { r.CacheRetention = c }
}

// WithSchema constrains the answer to a JSON shape.
func WithSchema(s *Schema) Option {
	return func(r *Request) { r.Schema = cloneSchema(s) }
}

// WithSamplingParams merges raw body parameters over whatever the model and
// any lower layer set. Keys a lower layer set and this map does not name are
// left in place.
func WithSamplingParams(params map[string]any) Option {
	return func(r *Request) {
		if len(params) == 0 {
			return
		}
		if r.SamplingParams == nil {
			r.SamplingParams = make(map[string]any, len(params))
		}
		maps.Copy(r.SamplingParams, params)
	}
}

// WithProtocolOptions supplies one driver's protocol-specific settings.
func WithProtocolOptions(native ProtocolOptions) Option {
	return func(r *Request) { r.ProtocolOptions = native }
}

// newRequest builds the request for one call: the model's own defaults first,
// then the client's, then the call's. Later layers overwrite earlier ones,
// which is the whole of the resolution rule.
func newRequest(m Model, messages []Message, layers ...[]Option) *Request {
	r := &Request{
		Messages:       messages,
		MaxTokens:      m.MaxOutput,
		SamplingParams: maps.Clone(m.SamplingParams),
	}
	for _, layer := range layers {
		for _, o := range layer {
			o(r)
		}
	}
	return r
}

// ToolChoice constrains tool use for one turn.
type ToolChoice struct {
	// Unexported, so the four states stay the only four. Fields a caller could
	// set independently are exactly what this type exists to rule out.
	mode toolChoiceMode
	name string
}

type toolChoiceMode uint8

const (
	choiceAuto toolChoiceMode = iota
	choiceNone
	choiceRequired
	choiceNamed
)

// The three states that need no argument. Compare against them with ==.
var (
	// ToolChoiceAuto lets the model decide. The zero value, and the default.
	ToolChoiceAuto = ToolChoice{}
	// ToolChoiceNone forbids tool calls for this turn.
	ToolChoiceNone = ToolChoice{mode: choiceNone}
	// ToolChoiceRequired makes the model call some tool, its choice which.
	ToolChoiceRequired = ToolChoice{mode: choiceRequired}
)

// ToolChoiceNamed makes the model call the one tool named.
func ToolChoiceNamed(name string) ToolChoice {
	return ToolChoice{mode: choiceNamed, name: name}
}

// Tool returns the tool this choice requires, and whether it requires one.
func (c ToolChoice) Tool() (string, bool) { return c.name, c.mode == choiceNamed }

// String describes the choice the way an error message wants it.
func (c ToolChoice) String() string {
	switch c.mode {
	case choiceNone:
		return "none"
	case choiceRequired:
		return "required"
	case choiceNamed:
		return "tool " + c.name
	}
	return "auto"
}

// Effort is a reasoning rung, named the same way whichever vendor serves it.
type Effort string

const (
	// EffortDefault leaves the decision to the model's default rung, or to the
	// provider when the model states none.
	EffortDefault Effort = ""
	// EffortOff disables reasoning where the provider allows it.
	EffortOff     Effort = "off"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// Efforts is the canonical ordering, least to most, excluding EffortDefault.
var Efforts = []Effort{
	EffortOff, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax,
}

// Valid reports whether e is one of the portable rungs above. A model may
// still offer rungs of its own naming; Model.Offers is what answers that.
func (e Effort) Valid() bool {
	if e == EffortDefault {
		return true
	}
	_, ok := effortRank(e)
	return ok
}

func effortRank(e Effort) (int, bool) {
	for i, known := range Efforts {
		if known == e {
			return i, true
		}
	}
	return 0, false
}

// CacheRetention is how long a provider should hold a prompt cache entry.
type CacheRetention string

const (
	// CacheDefault leaves the provider's own behavior in place, which for the
	// endpoints that cache is a short-lived entry.
	CacheDefault CacheRetention = ""
	// CacheNone asks that nothing be cached. Use it for a prompt that will
	// never repeat, where a cache write is pure cost.
	CacheNone CacheRetention = "none"
	// CacheShort is the provider's short lifetime, around five minutes.
	CacheShort CacheRetention = "short"
	// CacheLong is the provider's long lifetime — an hour on Anthropic, a day
	// on OpenAI. It costs more to write and pays off after two reads rather
	// than one.
	CacheLong CacheRetention = "long"
)

// ProtocolOptions is one driver's protocol-specific request settings.
type ProtocolOptions interface {
	// ProtocolOptions marks this type as one driver's request settings. A no-op:
	//
	//	func (Options) ProtocolOptions() {}
	ProtocolOptions()
}

// ProtocolOptionsAs reads a driver's protocol-specific settings out of a request. A
// non-nil value of the wrong concrete type is an invalid request.
//
//	native, err := ai.ProtocolOptionsAs[anthropic.Options](req)
//	if native.ThinkingDisplay != "" { … }
func ProtocolOptionsAs[T ProtocolOptions](req *Request) (T, error) {
	var zero T
	if req.ProtocolOptions == nil {
		return zero, nil
	}
	native, ok := req.ProtocolOptions.(T)
	if ok {
		return native, nil
	}
	return zero, &Error{
		Kind: KindInvalidRequest,
		Message: "ai: native options have type " + reflect.TypeOf(req.ProtocolOptions).String() +
			"; driver expects " + reflect.TypeFor[T]().String(),
	}
}

// RejectProtocolOptions returns an invalid-request error when a protocol with no native
// option type receives one. Drivers call it before building their wire request.
func RejectProtocolOptions(req *Request, driver string) error {
	if req.ProtocolOptions == nil {
		return nil
	}
	return &Error{
		Driver:  driver,
		Kind:    KindInvalidRequest,
		Message: "ai: driver " + driver + " does not define native request options",
	}
}

func cloneStringMap(source map[string]any) map[string]any { return maps.Clone(source) }
