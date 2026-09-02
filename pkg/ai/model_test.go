package ai

import (
	"encoding/json"
	"reflect"
	"testing"
)

// fullModel sets every field non-zero, so a field the codec forgets shows up
// as a missing key. Adding a field to Model means adding it here.
func fullModel() Model {
	return Model{
		ID:            "claude-opus-5",
		API:           APIAnthropicMessages,
		Name:          "Claude Opus 5",
		Vendor:        "anthropic",
		BaseURL:       "https://example.invalid",
		ContextWindow: 200_000,
		MaxOutput:     64_000,
		Input:         []Modality{ModalityText, ModalityImage},
		Reasoning: []ReasoningLevel{
			{Effort: EffortLow, Value: "low", Budget: 1024},
			{Effort: EffortHigh, Value: "high", Default: true},
		},
		Pricing: Pricing{Currency: USD, Input: 3, Output: 15,
			Tiers: []PricingTier{{AboveInputTokens: 200_000, Input: 6}}},
		Unsupported:    Unsupported{ToolChoice: true},
		Stage:          StagePreview,
		Replacement:    "claude-opus-6",
		SamplingParams: map[string]any{"top_p": 0.9},
		Headers:        map[string]string{"x-tenant": "acme"},
		Compat:         AnthropicCompat{BearerAuth: true},
	}
}

// Counting the encoded keys against the struct's fields is what catches a
// field the codec does not write, rather than a catalog file failing later.
func TestModelSurvivesAJSONRoundTrip(t *testing.T) {
	want := fullModel()

	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		t.Fatalf("the model did not encode as an object: %v", err)
	}
	if fields := reflect.TypeFor[Model]().NumField(); len(keys) != fields {
		t.Errorf("the encoding has %d keys for %d fields (%v); a field is not being written",
			len(keys), fields, keys)
	}

	var got Model
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round trip produced\n\t%+v\nwant\n\t%+v", got, want)
	}
}

// Compat is the one field that cannot be decoded without knowing the protocol,
// which is why it travels as raw JSON and is rebuilt afterwards.
func TestModelCompatIsRebuiltAsItsProtocolsType(t *testing.T) {
	for name, tc := range map[string]struct {
		raw     string
		want    any
		wantErr bool
	}{
		"an Anthropic model": {
			raw:  `{"id":"m","api":"anthropic-messages","compat":{"bearer_auth":true}}`,
			want: AnthropicCompat{BearerAuth: true},
		},
		"Vertex uses the same compat as Messages": {
			raw:  `{"id":"m","api":"anthropic-vertex","compat":{"no_temperature":true}}`,
			want: AnthropicCompat{NoTemperature: true},
		},
		"a Chat Completions model": {
			raw:  `{"id":"m","api":"openai-chat-completions","compat":{"thinking":"effort"}}`,
			want: OpenAIChatCompat{Thinking: ThinkingEffort},
		},
		"no compat at all": {
			raw:  `{"id":"m","api":"anthropic-messages"}`,
			want: nil,
		},
		"an explicit null": {
			raw:  `{"id":"m","api":"anthropic-messages","compat":null}`,
			want: nil,
		},
		"compat for a protocol nothing registered": {
			// Loading it with the quirks silently dropped is the failure mode
			// this refuses: the model would work until the quirk mattered.
			raw:     `{"id":"m","api":"invented","compat":{"whatever":true}}`,
			wantErr: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			var got Model
			err := json.Unmarshal([]byte(tc.raw), &got)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Unmarshal: %v, want an error: %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if !reflect.DeepEqual(got.Compat, tc.want) {
				t.Errorf("compat = %#v, want %#v", got.Compat, tc.want)
			}
		})
	}
}

// A Model handed to a caller is a snapshot: the catalog's rows are a package
// table, so an aliased slice or map lets one caller edit what all others read.
func TestCloneDetachesEveryMutableField(t *testing.T) {
	original := fullModel()
	clone := original.Clone()

	clone.Input[0] = ModalityAudio
	clone.Reasoning[0].Value = "edited"
	clone.Pricing.Tiers[0].Input = 999
	clone.SamplingParams["top_p"] = 0.1
	clone.Headers["x-tenant"] = "someone-else"

	if !reflect.DeepEqual(original, fullModel()) {
		t.Errorf("editing the clone reached the original:\n\t%+v", original)
	}
}

// ResolveLevel answers a caller asking for a rung this model does not sell. It
// snaps up: asking for more thinking and getting less goes unnoticed.
func TestResolveLevelSnapsToARungTheModelHas(t *testing.T) {
	laddered := Model{Reasoning: []ReasoningLevel{
		{Effort: EffortOff, Value: ""},
		{Effort: EffortLow, Value: "low"},
		{Effort: EffortHigh, Value: "high", Default: true},
	}}

	for name, tc := range map[string]struct {
		model Model
		want  Effort
		level Effort
		ok    bool
	}{
		"an exact rung is used as it is":             {laddered, EffortLow, EffortLow, true},
		"a rung above is snapped up":                 {laddered, EffortMedium, EffortHigh, true},
		"nothing above means the highest below":      {laddered, EffortMax, EffortHigh, true},
		"the default rung answers no request":        {laddered, EffortDefault, EffortHigh, true},
		"a name off the ladder falls to the default": {laddered, Effort("turbo"), EffortHigh, true},
		"a model that does not reason resolves nothing": {
			Model{}, EffortHigh, EffortDefault, false,
		},
		"a ladder with no default answers nothing for one": {
			Model{Reasoning: []ReasoningLevel{{Effort: EffortLow, Value: "low"}}},
			EffortDefault, EffortDefault, false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := tc.model.ResolveLevel(tc.want)
			if ok != tc.ok {
				t.Fatalf("ResolveLevel(%q) resolved: %v, want %v", tc.want, ok, tc.ok)
			}
			if ok && got.Effort != tc.level {
				t.Errorf("ResolveLevel(%q) = %q, want %q", tc.want, got.Effort, tc.level)
			}
		})
	}
}

// Vertex serves the Anthropic Messages body, so every rule about that body has
// to recognise it — one naming a single protocol stops applying silently.
func TestTheAnthropicFamilyIsBothOfItsProtocols(t *testing.T) {
	for api, want := range map[API]bool{
		APIAnthropicMessages: true,
		APIAnthropicVertex:   true,
		APIOpenAIChat:        false,
		APIOpenAIResponses:   false,
		APIGoogleGenAI:       false,
		"":                   false,
	} {
		if got := api.anthropicFamily(); got != want {
			t.Errorf("%q.anthropicFamily() = %v, want %v", api, got, want)
		}
	}
}
