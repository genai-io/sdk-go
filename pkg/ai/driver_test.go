package ai

import (
	"strings"
	"testing"
)

type fakeConfig struct{ Project string }

func (fakeConfig) ProtocolConfig() {}

type otherConfig struct{}

func (otherConfig) ProtocolConfig() {}

// ProtocolConfig carries a driver's own value through an interface, and
// falling back to the zero value would look like the option having no effect.
func TestAProtocolValueOfTheWrongTypeIsReportedRatherThanIgnored(t *testing.T) {
	if got, err := protocolValueAs[fakeConfig](nil, "config"); err != nil || got != (fakeConfig{}) {
		t.Errorf("no config gave %v, %v; want the zero value and no error", got, err)
	}
	want := fakeConfig{Project: "p"}
	if got, err := protocolValueAs[fakeConfig](want, "config"); err != nil || got != want {
		t.Errorf("config = %v, %v; want %v", got, err, want)
	}
	_, err := protocolValueAs[fakeConfig](otherConfig{}, "config")
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "otherConfig") {
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
