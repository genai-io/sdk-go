package endpoint

import "github.com/genai-io/sdk-go/pkg/ai"

// Snapshots taken wherever a model crosses in or out of a Endpoint, so a
// caller may keep mutating its own builders and a refresh running in another
// goroutine cannot rewrite a list someone is reading.

func cloneAll(models []ai.Model) []ai.Model {
	out := make([]ai.Model, len(models))
	for i, m := range models {
		out[i] = m.Clone()
	}
	return out
}
