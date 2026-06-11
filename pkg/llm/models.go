package llm

// ProviderMeta carries static metadata about a registered provider.
type ProviderMeta struct {
	Name        string   // provider name (e.g. "anthropic")
	AuthMethods []string // supported auth methods (e.g. ["api_key", "vertex"])
	DisplayName string
	EnvVars     []string // required environment variables
}
