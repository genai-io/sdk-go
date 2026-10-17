package ai

import (
	"strings"
	"testing"
)

// A missing blank import is the commonest way to hold an unusable client, and
// the error is the only place it is ever explained.
func TestAnUnregisteredProtocolSaysWhatToImport(t *testing.T) {
	none := unregistered("invented", nil).Error()
	if !strings.Contains(none, driverPath) {
		t.Errorf("err = %q, want it to name where the drivers live", none)
	}

	// Registered arrives from RegisteredAPIs, which sorted it; sorting again
	// here would be the same work twice with nothing to show for it.
	some := unregistered("invented", []API{APIAnthropicMessages, APIOpenAIChat}).Error()
	if !strings.Contains(some, "anthropic-messages openai-chat-completions") {
		t.Errorf("err = %q, want it to list what is linked in, in order", some)
	}

	if _, err := newDriver(Config{Model: Model{ID: "m"}}); err == nil {
		t.Error("a model with no API resolved to a driver")
	}
	if _, err := newDriver(Config{Model: Model{API: APIOpenAIChat}}); err == nil {
		t.Error("a model with no ID resolved to a driver")
	}
}
