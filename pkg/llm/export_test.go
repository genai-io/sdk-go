package llm

// Internal steps exposed to the external test package.
//
// These are the ordered steps of Classify and PrepareMessages. They are not
// public API — calling them out of order reintroduces the masking each
// orchestrator is written to prevent — but they are worth testing directly,
// because a step that is only reached through its orchestrator is a step whose
// edge cases are tested by accident.

var (
	ClassifyStatus       = classifyStatus
	SanitizeToolMessages = sanitizeToolMessages
	DropEmptyMessages    = dropEmptyMessages
	SanitizeText         = sanitizeText
)
