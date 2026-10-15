package ai

import (
	"strings"
	"testing"
)

type fakeConfig struct{ Project string }

func (fakeConfig) ProtocolConfig() {}

// ProtocolConfig carries a driver's own value through an interface, and
// falling back to the zero value would look like the option having no effect.
func TestAProtocolConfigOfTheWrongTypeIsReportedRatherThanIgnored(t *testing.T) {
	want := VertexConfig{Project: "p", Region: "r"}
	if got, err := VertexDeployment(Config{ProtocolConfig: want}, "stub"); err != nil || got != want {
		t.Errorf("deployment = %v, %v; want %v", got, err, want)
	}
	_, err := VertexDeployment(Config{ProtocolConfig: fakeConfig{}}, "stub")
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "fakeConfig") {
		t.Errorf("err = %v, want it to name the type that was actually given", err)
	}
}

// A protocol that defines no escape hatch must say so rather than drop what it
// was handed.
func TestAProtocolWithNoEscapeHatchRefusesOne(t *testing.T) {
	if err := RejectProtocolConfig(Config{}, "stub"); err != nil {
		t.Errorf("RejectProtocolConfig with none given = %v, want nil", err)
	}
	err := RejectProtocolConfig(Config{ProtocolConfig: fakeConfig{}}, "stub")
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "stub") {
		t.Errorf("err = %v, want it to name the driver", err)
	}
}

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

	if _, err := NewDriver(Config{Model: Model{ID: "m"}}); err == nil {
		t.Error("a model with no API resolved to a driver")
	}
}
