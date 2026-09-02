package ai

import (
	"strings"
	"testing"
)

type fakeOptions struct{ Display string }

func (fakeOptions) ProtocolOptions() {}

type otherOptions struct{}

func (otherOptions) ProtocolOptions() {}

type fakeConfig struct{ Project string }

func (fakeConfig) ProtocolConfig() {}

// The escape hatches carry a driver's own value through an interface, and
// falling back to the zero value would look like the option having no effect.
func TestAProtocolValueOfTheWrongTypeIsReportedRatherThanIgnored(t *testing.T) {
	if got, err := ProtocolOptionsAs[fakeOptions](&Request{}); err != nil || got != (fakeOptions{}) {
		t.Errorf("no options gave %v, %v; want the zero value and no error", got, err)
	}
	want := fakeOptions{Display: "on"}
	if got, err := ProtocolOptionsAs[fakeOptions](&Request{ProtocolOptions: want}); err != nil || got != want {
		t.Errorf("options = %v, %v; want %v", got, err, want)
	}
	_, err := ProtocolOptionsAs[fakeOptions](&Request{ProtocolOptions: otherOptions{}})
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "otherOptions") {
		t.Errorf("err = %v, want it to name the type that was actually given", err)
	}

	if got, err := ProtocolConfigAs[fakeConfig](Config{}); err != nil || got != (fakeConfig{}) {
		t.Errorf("no config gave %v, %v; want the zero value and no error", got, err)
	}
	_, err = ProtocolConfigAs[fakeConfig](Config{ProtocolConfig: VertexConfig{Project: "p"}})
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "VertexConfig") {
		t.Errorf("err = %v, want it to name the type that was actually given", err)
	}
}

// A protocol that defines no escape hatch must say so rather than drop what it
// was handed.
func TestAProtocolWithNoEscapeHatchRefusesOne(t *testing.T) {
	if err := RejectProtocolOptions(&Request{}, "stub"); err != nil {
		t.Errorf("RejectProtocolOptions with none given = %v, want nil", err)
	}
	err := RejectProtocolOptions(&Request{ProtocolOptions: fakeOptions{}}, "stub")
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "stub") {
		t.Errorf("err = %v, want it to name the driver", err)
	}

	if err := RejectProtocolConfig(Config{}, "stub"); err != nil {
		t.Errorf("RejectProtocolConfig with none given = %v, want nil", err)
	}
	err = RejectProtocolConfig(Config{ProtocolConfig: fakeConfig{}}, "stub")
	if !IsKind(err, KindInvalidRequest) || !strings.Contains(err.Error(), "stub") {
		t.Errorf("err = %v, want it to name the driver", err)
	}
}

// A missing blank import is the commonest way to hold an unusable client, and
// the error is the only place it is ever explained.
func TestAnUnregisteredProtocolSaysWhatToImport(t *testing.T) {
	none := (&UnregisteredAPIError{API: "invented"}).Error()
	if !strings.Contains(none, driverPath) {
		t.Errorf("err = %q, want it to name where the drivers live", none)
	}

	// Registered arrives from RegisteredAPIs, which sorted it; sorting again
	// here would be the same work twice with nothing to show for it.
	some := (&UnregisteredAPIError{API: "invented", Registered: []API{APIAnthropicMessages, APIOpenAIChat}}).Error()
	if !strings.Contains(some, "anthropic-messages openai-chat-completions") {
		t.Errorf("err = %q, want it to list what is linked in, in order", some)
	}

	if _, err := NewDriver(Config{Model: Model{ID: "m"}}); err == nil {
		t.Error("a model with no API resolved to a driver")
	}
}

// Wrap has to keep the optional capabilities as well as Stream, or attaching
// middleware would quietly change what a client can do.
func TestWrapKeepsTheOptionalCapabilities(t *testing.T) {
	plain := Wrap(&scripted{scripts: []script{{}}})
	c := NewClientWithDriver(plain, stubModel())

	if _, err := c.Models(t.Context()); !IsUnsupported(err) {
		t.Errorf("Models = %v, want it reported as unsupported", err)
	}
	if _, err := c.CountTokens(t.Context(), []Message{UserMessage("hi")}); err != nil {
		t.Errorf("CountTokens: %v, want it to fall back to an estimate", err)
	}
}
