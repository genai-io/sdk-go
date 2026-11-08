package ai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestStreamAdmissionRejectsBeforeDriver(t *testing.T) {
	driver := &scripted{scripts: []script{{}}}
	client := NewClientWithDriver(driver, stubModel())
	want := &Error{Kind: KindNetwork, Message: "persistence unavailable"}
	calls := 0
	resp, err := collect(client.StreamAdmitted(context.Background(), []Message{UserMessage("hi")}, func(context.Context, *Request) error {
		calls++
		return want
	}))
	var admission *AdmissionError
	if !errors.As(err, &admission) || !errors.Is(err, want) || IsRetryable(err) || calls != 1 || driver.calls != 0 {
		t.Fatalf("error=%v, admission calls=%d, driver calls=%d", err, calls, driver.calls)
	}
	if resp == nil || !errors.Is(resp.Err, want) || resp.StopReason != StopError {
		t.Fatalf("response=%+v", resp)
	}
}

func TestStreamAdmissionReceivesPreparedDetachedRequest(t *testing.T) {
	driver := &scripted{scripts: []script{{}}}
	client := NewClientWithDriver(driver, stubModel(), WithMaxTokens(100))
	definition := map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}}
	params := map[string]any{"seed": int64(9007199254740993), "nested": map[string]any{"value": "original"}}
	messages := []Message{UserMessage("hi\xffthere")}
	var before []byte
	_, err := collect(client.StreamAdmitted(context.Background(), messages, func(_ context.Context, req *Request) error {
		if req.MaxTokens != 200 || req.ToolChoice != ToolChoiceNamed("search") || req.Messages[0].Content[0].Text != "hi�there" {
			t.Fatalf("not finalized: %+v", req)
		}
		before, _ = json.Marshal(req)
		req.Messages[0].Content[0].Text = "changed snapshot"
		req.Tools[0].Definition.(map[string]any)["properties"].(map[string]any)["x"] = "changed snapshot"
		req.SamplingParams["nested"].(map[string]any)["value"] = "changed snapshot"
		*req.Temperature = 1
		req.ToolChoice = ToolChoiceNone
		// Nor can application-held aliases alter the frozen dispatch after admission.
		messages[0].Content[0].Text = "changed input"
		definition["type"] = "changed input"
		params["seed"] = 0
		return nil
	}, WithMaxTokens(200), WithTemperature(0), WithTools(Schema{Name: "search", Definition: definition}), WithToolChoice(ToolChoiceNamed("search")), func(req *Request) { req.SamplingParams = params }))
	if err != nil {
		t.Fatal(err)
	}
	if driver.calls != 1 {
		t.Fatalf("driver calls=%d", driver.calls)
	}
	after, err := json.Marshal(driver.reqs[0])
	if err != nil || string(before) != string(after) || driver.reqs[0].ToolChoice != ToolChoiceNamed("search") {
		t.Fatalf("snapshot and dispatch differ: %s / %s (%v)", before, after, err)
	}
	if got := driver.reqs[0].SamplingParams["seed"]; got != json.Number("9007199254740993") {
		t.Fatalf("large integer changed: %v", got)
	}
}

func TestStreamAdmissionValidationAndCancellation(t *testing.T) {
	for _, mode := range []string{"invalid", "canceled-before", "canceled-during", "unserializable"} {
		t.Run(mode, func(t *testing.T) {
			driver := &scripted{scripts: []script{{}}}
			client := NewClientWithDriver(driver, stubModel())
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var opts []Option
			switch mode {
			case "invalid":
				opts = append(opts, WithMaxTokens(-1))
			case "canceled-before":
				cancel()
			case "unserializable":
				opts = append(opts, func(req *Request) { req.SamplingParams = map[string]any{"bad": make(chan int)} })
			}
			called := false
			_, err := collect(client.StreamAdmitted(ctx, []Message{UserMessage("hi")}, func(context.Context, *Request) error {
				called = true
				cancel()
				return nil
			}, opts...))
			if err == nil || driver.calls != 0 || called != (mode == "canceled-during") {
				t.Fatalf("error=%v called=%v driver calls=%d", err, called, driver.calls)
			}
		})
	}
}

func TestStreamNilAdmissionMatchesStream(t *testing.T) {
	run := script{deltas: []Delta{{Block: TextBlock("answer"), StopReason: StopEndTurn}}}
	client := drive(run)
	messages := []Message{UserMessage("hi")}
	plain, err := collect(client.Stream(context.Background(), messages))
	if err != nil {
		t.Fatal(err)
	}
	admitted, err := collect(client.StreamAdmitted(context.Background(), messages, nil))
	if err != nil || !reflect.DeepEqual(plain, admitted) {
		t.Fatalf("responses differ: %+v / %+v (%v)", plain, admitted, err)
	}
}
