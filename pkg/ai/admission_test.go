package ai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestCompleteAdmittedPreparedDetachedRequest(t *testing.T) {
	for i, choice := range []ToolChoice{ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceNamed("search")} {
		t.Run([]string{"auto", "none", "required", "named"}[i], func(t *testing.T) {
			driver := &scripted{scripts: []script{{deltas: []Delta{{Block: TextBlock("answer")}}}}}
			client := NewClientWithDriver(driver, stubModel(), WithMaxTokens(100), WithSystem("system"))
			messages := []Message{UserMessage("hi\xffthere")}
			opts := []Option{
				WithMaxTokens(200), WithTemperature(0), WithEffort(EffortHigh),
				WithTools(Schema{Name: "search", Definition: map[string]any{"type": "object"}}),
				WithToolChoice(choice), WithStopSequences("stop"),
				WithSchema(&Schema{Name: "answer", Definition: map[string]any{"type": "object"}}),
				func(req *Request) {
					req.CacheRetention = "short"
					req.SamplingParams = map[string]any{"seed": 42}
				},
			}
			prepared, err := client.prepare(context.Background(), messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			want, err := json.Marshal(prepared)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			resp, err := client.CompleteAdmitted(context.Background(), messages, func(_ context.Context, req *Request) error {
				calls++
				got, err := json.Marshal(req)
				if err != nil || string(got) != string(want) || req.Messages[0].Content[0].Text != "hi�there" || req.MaxTokens != 200 || req.System != "system" {
					t.Fatalf("prepared request=%s, want=%s (%v)", got, want, err)
				}
				req.Messages[0].Content[0].Text = "changed snapshot"
				req.Tools[0].Definition.(map[string]any)["type"] = "changed"
				req.Schema.Definition.(map[string]any)["type"] = "changed"
				req.StopSequences[0] = "changed"
				req.SamplingParams["seed"] = 0
				*req.Temperature = 1
				req.ToolChoice = ToolChoiceNone
				messages[0].Content[0].Text = "changed input"
				return nil
			}, opts...)
			if err != nil || resp == nil || resp.Content.Text() != "answer" || calls != 1 || driver.calls != 1 {
				t.Fatalf("response=%+v error=%v admission calls=%d driver calls=%d", resp, err, calls, driver.calls)
			}
			got, err := json.Marshal(driver.reqs[0])
			if err != nil || string(got) != string(want) {
				t.Fatalf("dispatch=%s, want=%s (%v)", got, want, err)
			}
		})
	}
}

func TestCompleteAdmittedRejectsBeforeDriver(t *testing.T) {
	for _, canceled := range []bool{false, true} {
		t.Run(map[bool]string{false: "rejected", true: "canceled"}[canceled], func(t *testing.T) {
			driver := &scripted{scripts: []script{{}}}
			client := NewClientWithDriver(driver, stubModel())
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			want := errors.New("admission failed")
			calls := 0
			resp, err := client.CompleteAdmitted(ctx, []Message{UserMessage("hi")}, func(context.Context, *Request) error {
				calls++
				if canceled {
					cancel()
					return nil
				}
				return want
			})
			if canceled {
				if !errors.Is(err, context.Canceled) || !IsKind(err, KindCanceled) || resp == nil || resp.StopReason != StopAborted {
					t.Fatalf("response=%+v error=%v", resp, err)
				}
			} else {
				var admission *AdmissionError
				if !errors.As(err, &admission) || !errors.Is(err, want) || IsRetryable(err) || resp == nil || !errors.Is(resp.Err, want) || resp.StopReason != StopError {
					t.Fatalf("response=%+v error=%v", resp, err)
				}
			}
			if calls != 1 || driver.calls != 0 {
				t.Fatalf("admission calls=%d driver calls=%d", calls, driver.calls)
			}
		})
	}
}

func TestCompleteNilAdmissionMatchesComplete(t *testing.T) {
	failure := errors.New("stream failed")
	for _, terminal := range []error{nil, failure} {
		name := "success"
		if terminal != nil {
			name = "partial-failure"
		}
		t.Run(name, func(t *testing.T) {
			client := drive(script{deltas: []Delta{{Block: TextBlock("answer"), Usage: &Usage{Input: 10, Output: 3}}}, err: terminal})
			messages := []Message{UserMessage("hi")}
			plain, plainErr := client.Complete(context.Background(), messages)
			admitted, err := client.CompleteAdmitted(context.Background(), messages, nil)
			if !errors.Is(err, terminal) || !errors.Is(plainErr, terminal) || !reflect.DeepEqual(plain, admitted) {
				t.Fatalf("responses=%+v / %+v errors=%v / %v", plain, admitted, plainErr, err)
			}
			if admitted == nil || admitted.Content.Text() != "answer" || admitted.Usage.Input != 10 || admitted.Usage.Output != 3 {
				t.Fatalf("lost content or usage: %+v", admitted)
			}
		})
	}
}

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
