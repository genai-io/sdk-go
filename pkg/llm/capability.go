package llm

import (
	"fmt"
	"strings"
)

// Unsupported records what a model cannot do.
//
// It is stated as absences rather than capabilities so that its zero value is
// a fully capable model, which is nearly all of them — an entry says only what
// is missing. That also means a hand-built Model, or one that arrived from a
// live listing with nothing but an ID, is assumed capable rather than assumed
// crippled.
type Unsupported struct {
	// Tools means the endpoint rejects tool definitions outright. Common on
	// small local models and on the older completion-style endpoints.
	Tools bool `json:"tools,omitempty"`
	// ToolChoice means tools work but constraining which one does not.
	ToolChoice bool `json:"tool_choice,omitempty"`
	// System means there is no system prompt. foldSystemPrompt turns one into
	// an opening exchange for these.
	System bool `json:"system,omitempty"`
	// Multiturn means the endpoint accepts a single message only.
	Multiturn bool `json:"multiturn,omitempty"`
	// Streaming means the endpoint has no streaming mode. The driver still
	// presents one, delivering the whole answer as a single delta.
	Streaming bool `json:"streaming,omitempty"`
	// Schema means the endpoint cannot constrain output to a JSON schema.
	// SimulateSchema asks for the shape in words instead.
	Schema bool `json:"schema,omitempty"`
	// SchemaWithTools means it can constrain output, but not in the same
	// request that offers tools.
	SchemaWithTools bool `json:"schema_with_tools,omitempty"`
}

// Stage is where a model sits in its vendor's lifecycle.
//
// A retired model is kept in the catalog rather than deleted so that a caller
// still pointing at one gets told what happened and what to move to. Deleting
// the entry turns a clear "retired on 2026-02-19, use claude-sonnet-5" into an
// opaque 404 from the provider.
type Stage string

const (
	// StageStable is the default: generally available and supported.
	StageStable Stage = ""
	// StagePreview is available but subject to change without notice.
	StagePreview Stage = "preview"
	// StageDeprecated still serves requests but has an announced end date.
	StageDeprecated Stage = "deprecated"
	// StageRetired no longer serves requests. Validate refuses to send to one.
	StageRetired Stage = "retired"
)

// Available reports whether the model still serves requests.
func (s Stage) Available() bool { return s != StageRetired }

// Validate checks a request against what the model can actually do, before it
// reaches the network.
//
// The point is to fail with a sentence a caller can act on — "deepseek-v4-pro
// does not accept images" — instead of an opaque provider rejection, or worse,
// a silently degraded answer. Sending an image to a text-only endpoint used to
// mean the image was dropped and the model answered about something it had
// never seen; that is the failure this replaces.
func (m Model) Validate(p *Prompt, opts Options) error {
	if !m.Stage.Available() {
		msg := fmt.Sprintf("model %s is retired and no longer serves requests", m)
		if m.Replacement != "" {
			msg += "; use " + m.Replacement
		}
		return &Error{Kind: KindUnsupported, Message: msg}
	}
	if err := m.checkCompat(); err != nil {
		return err
	}
	if p == nil {
		return nil
	}

	if !m.Accepts(ModalityImage) {
		for _, msg := range p.Messages {
			if msg.Content.HasImages() {
				return m.unsupported("does not accept image input")
			}
		}
	}
	if m.Unsupported.Tools && len(p.Tools) > 0 {
		return m.unsupported("does not support tools, but %d were provided", len(p.Tools))
	}
	if m.Unsupported.ToolChoice && (opts.ToolChoice != ToolChoiceAuto || opts.ForceTool != "") {
		return m.unsupported("does not support constraining which tool is called")
	}
	if m.Unsupported.System && p.System != "" {
		return m.unsupported("has no system prompt; fold it into the conversation with llm.SimulateSystemPrompt()")
	}
	if m.Unsupported.Multiturn && len(p.Messages) > 1 {
		return m.unsupported("accepts a single message, but %d were provided", len(p.Messages))
	}
	if opts.Schema != nil {
		switch {
		case m.Unsupported.Schema:
			return m.unsupported("cannot constrain output to a schema; " +
				"add llm.SimulateSchema() to ask for the shape in words instead")
		case m.Unsupported.SchemaWithTools && len(p.Tools) > 0:
			return m.unsupported("cannot constrain output and offer tools in the same request")
		}
	}
	return nil
}

func (m Model) unsupported(format string, args ...any) error {
	return &Error{
		Kind:    KindUnsupported,
		Message: fmt.Sprintf("model %s %s", m, fmt.Sprintf(format, args...)),
	}
}

// Available returns the models that still serve requests, dropping retired
// ones. A model picker wants this; a lookup by ID does not, because answering
// "that one was retired, use this instead" needs the entry to still be there.
func Available(models []Model) []Model {
	out := make([]Model, 0, len(models))
	for _, m := range models {
		if m.Stage.Available() {
			out = append(out, m)
		}
	}
	return out
}

// SystemPreface introduces a folded system prompt, and SystemAck is the
// acknowledgement attributed to the model. They are exported so a caller whose
// model responds better to different wording can replace them.
var (
	SystemPreface = "SYSTEM INSTRUCTIONS:\n"
	SystemAck     = "Understood."
)

// foldSystemPrompt turns a system prompt into an opening user turn and an
// acknowledgement, for a model that has no system role.
//
// It is a fallback, not an equivalent: a folded prompt is ordinary
// conversation the model may argue with or forget, where a real system prompt
// is weighted and cached differently. Use it when the alternative is not
// sending the instructions at all.
//
// A prompt with no system text, or a model that has a system role, comes back
// unchanged.
func foldSystemPrompt(p *Prompt) *Prompt {
	if p == nil || strings.TrimSpace(p.System) == "" {
		return p
	}
	out := *p
	out.System = ""
	out.Messages = append([]Message{
		User(SystemPreface + p.System),
		Assistant(SystemAck),
	}, p.Messages...)
	return &out
}
