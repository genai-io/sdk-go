package catalog

import "strings"

// ResolveBaseURL applies an override to a vendor's endpoint.
//
// An empty override leaves the vendor default in place. An override is
// trimmed of its trailing slash and given the vendor's required path suffix if
// it lacks one, so the bare "http://localhost:11434" people paste for a local
// Ollama still reaches its OpenAI-compatible API.
func (v Vendor) ResolveBaseURL(override string) string {
	override = strings.TrimSpace(override)
	if override == "" {
		return v.BaseURL
	}
	override = strings.TrimRight(override, "/")
	if v.BaseURLSuffix != "" && !strings.HasSuffix(override, v.BaseURLSuffix) {
		override += v.BaseURLSuffix
	}
	return override
}
