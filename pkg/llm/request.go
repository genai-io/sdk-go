package llm

// Tool is a tool definition offered to the model.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object describing the tool's arguments,
	// typically a map[string]any. Drivers translate it into whatever shape
	// their protocol wants.
	Parameters any `json:"parameters,omitempty"`
}

// ToolChoice constrains which tool, if any, the model may call.
type ToolChoice string

const (
	// ToolChoiceAuto lets the model decide. The default.
	ToolChoiceAuto ToolChoice = ""
	// ToolChoiceNone forbids tool calls for this turn.
	ToolChoiceNone ToolChoice = "none"
	// ToolChoiceRequired forces the model to call some tool.
	ToolChoiceRequired ToolChoice = "required"
)

// Effort is a normalized reasoning rung.
//
// Every vendor spells this differently — token budgets, level strings, boolean
// enable flags — and a caller should not have to know which. A model's
// ReasoningLevel list maps each rung onto what its endpoint wants, so the same
// request runs unchanged against Claude, GPT, Gemini or Qwen.
//
// The ladder matches pi-ai's so the ecosystem has one vocabulary rather than
// two; not every model offers every rung, and Model.ResolveLevel snaps a
// request onto what the model does offer.
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

// Valid reports whether e is one of the canonical rungs.
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
//
// Providers spell this differently and price it differently — Anthropic bills
// a 1-hour write at twice the input rate where a 5-minute write costs 1.25x —
// so it is a normalized request-level choice rather than something a caller
// encodes per provider. An endpoint with no prompt cache ignores it.
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

// Prompt is the conversation: what the model is being asked, and what it may
// call. It carries nothing about how to run the request — that is Options —
// so the same Prompt can be sent to two models with different settings, or to
// a driver's native entry point without translation.
//
// It is named Prompt rather than Context because Go already has one of those
// in every function signature here.
type Prompt struct {
	// System is the system prompt. Protocols place it differently — a system
	// block, an instructions field, a system_instruction content — which is
	// why it is not a Message.
	System string

	Messages []Message
	Tools    []Tool
}

// Options is how to run one inference call.
//
// A nil *Options is valid and means "all defaults". The zero value of
// MaxTokens and Temperature means "unset": Client fills MaxTokens from the
// model's advertised output limit, and an unset temperature is simply not
// sent, leaving the provider's default in place.
type Options struct {
	MaxTokens   int
	Temperature float64

	// Effort is the normalized reasoning rung. The model's ladder decides what
	// actually goes on the wire.
	Effort Effort

	// ToolChoice constrains tool use for this turn.
	ToolChoice ToolChoice

	// ForceTool names the one tool the model must call this turn. It implies
	// ToolChoiceRequired and overrides it — "call something" and "call this"
	// are the same constraint at different resolutions, and every supported
	// protocol expresses both.
	ForceTool string

	// StopSequences ends generation when the model emits one of these.
	StopSequences []string

	// CacheRetention asks for a prompt-cache lifetime.
	CacheRetention CacheRetention

	// Schema constrains the answer to a JSON shape. Nil leaves the model free
	// to answer in prose.
	Schema *Schema

	// SamplingParams are merged into the request body verbatim, over the named
	// fields and over the model's own SamplingParams, so a caller can reach a
	// parameter this SDK does not model. Only the OpenAI-family drivers apply
	// them.
	SamplingParams map[string]any

	// Native carries settings only one protocol has, as that driver's Native
	// value — anthropic.Native, openairesp.Native. It is the escape hatch for
	// what the normalized options above deliberately do not model, so needing
	// one thing a protocol offers does not mean writing a whole driver.
	//
	// A driver reads it with NativeOf, which yields the zero value when the
	// options carry none or carry another protocol's, so an unset Native and
	// an all-defaults one are the same thing. Setting a Native for the wrong
	// protocol is silently ignored rather than failing the request: the same
	// Options should stay usable when a caller swaps the model underneath it.
	Native any
}

// NativeOf reads a driver's protocol-specific settings out of a request's
// options.
//
//	native := llm.NativeOf[anthropic.Native](opts)
//	if native.ThinkingDisplay != "" { … }
func NativeOf[T any](o Options) T {
	n, _ := o.Native.(T)
	return n
}

// merged returns o with the model's defaults filled in, without mutating
// either. A nil receiver yields the model's defaults alone.
func (o *Options) merged(m Model, clientDefaults Options) Options {
	out := clientDefaults
	if o != nil {
		if o.MaxTokens != 0 {
			out.MaxTokens = o.MaxTokens
		}
		if o.Temperature != 0 {
			out.Temperature = o.Temperature
		}
		if o.Effort != EffortDefault {
			out.Effort = o.Effort
		}
		if o.ToolChoice != ToolChoiceAuto {
			out.ToolChoice = o.ToolChoice
		}
		if o.ForceTool != "" {
			out.ForceTool = o.ForceTool
		}
		if o.CacheRetention != CacheDefault {
			out.CacheRetention = o.CacheRetention
		}
		if o.Schema != nil {
			out.Schema = o.Schema
		}
		if o.Native != nil {
			out.Native = o.Native
		}
		if len(o.StopSequences) > 0 {
			out.StopSequences = o.StopSequences
		}
		if len(o.SamplingParams) > 0 {
			out.SamplingParams = o.SamplingParams
		}
	}
	if out.MaxTokens == 0 {
		out.MaxTokens = m.MaxOutput
	}
	// Per-model sampling params sit underneath the per-request ones.
	if len(m.SamplingParams) > 0 {
		merged := make(map[string]any, len(m.SamplingParams)+len(out.SamplingParams))
		for k, v := range m.SamplingParams {
			merged[k] = v
		}
		for k, v := range out.SamplingParams {
			merged[k] = v
		}
		out.SamplingParams = merged
	}
	return out
}
