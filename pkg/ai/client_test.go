package ai

import (
	"context"
	"errors"
	"iter"
	"maps"
	"strings"
	"testing"
	"unicode/utf8"
)

// script is one driver run: the deltas it produces, then the error it ends on.
type script struct {
	deltas []Delta
	err    error
}

// scripted is a Driver that replays a fixed run per call. The last script
// repeats, so a retry test states only the attempts that differ.
type scripted struct {
	scripts []script
	calls   int
	reqs    []*Request
}

func (s *scripted) Name() string { return "scripted" }

func (s *scripted) Stream(_ context.Context, req *Request) iter.Seq2[Delta, error] {
	return func(yield func(Delta, error) bool) {
		s.reqs = append(s.reqs, req)
		run := s.scripts[min(s.calls, len(s.scripts)-1)]
		s.calls++
		for _, delta := range run.deltas {
			if !yield(delta, nil) {
				return
			}
		}
		if run.err != nil {
			yield(Delta{}, run.err)
		}
	}
}

// stubModel is a model with no protocol rules of its own, so a stream test
// exercises the tracker rather than validation.
func stubModel() Model { return Model{ID: "stub", API: "stub"} }

func drive(scripts ...script) *Client {
	return NewClientWithDriver(&scripted{scripts: scripts}, stubModel())
}

// seen is one event flattened to what a consumer actually switches on.
type seen struct {
	kind  EventType
	index int
	block BlockType
	text  string
}

func collectEvents(t *testing.T, c *Client, msgs ...Message) ([]seen, *Response, error) {
	t.Helper()
	if len(msgs) == 0 {
		msgs = []Message{UserMessage("hi")}
	}
	var out []seen
	var resp *Response
	var failure error
	for event, err := range c.Stream(context.Background(), msgs) {
		if event.Type == EventDone {
			resp = event.Response
		} else {
			out = append(out, seen{event.Type, event.Index, event.Block.Type, event.Block.Text})
		}
		if err != nil {
			failure = err
		}
	}
	return out, resp, failure
}

// Every content kind shares one start/delta/end lifecycle, and the index is
// what tells two blocks apart — for every kind, not only for text.
func TestStreamGivesEveryBlockTheSameLifecycle(t *testing.T) {
	call := ToolCall{ID: "c1", Name: "search", Input: `{"q":"go"}`}

	for name, tc := range map[string]struct {
		deltas []Delta
		want   []seen
	}{
		"text arrives in fragments and closes once": {
			deltas: []Delta{
				{Block: TextBlock("Hel")},
				{Block: TextBlock("lo")},
			},
			want: []seen{
				{EventBlockStart, 0, BlockText, ""},
				{EventBlockDelta, 0, BlockText, "Hel"},
				{EventBlockDelta, 0, BlockText, "lo"},
				{EventBlockEnd, 0, BlockText, "Hello"},
			},
		},
		"switching from thinking to text closes the open block": {
			deltas: []Delta{
				{Block: ThinkingBlock("hmm", "")},
				{Block: ThinkingBlock("", "sig")},
				{Block: TextBlock("answer")},
			},
			want: []seen{
				{EventBlockStart, 0, BlockThinking, ""},
				{EventBlockDelta, 0, BlockThinking, "hmm"},
				{EventBlockDelta, 0, BlockThinking, ""},
				{EventBlockEnd, 0, BlockThinking, "hmm"},
				{EventBlockStart, 1, BlockText, ""},
				{EventBlockDelta, 1, BlockText, "answer"},
				{EventBlockEnd, 1, BlockText, "answer"},
			},
		},
		"an explicit end starts a second block of the same kind": {
			deltas: []Delta{
				{Block: TextBlock("one")},
				{EndBlock: true},
				{Block: TextBlock("two")},
			},
			want: []seen{
				{EventBlockStart, 0, BlockText, ""},
				{EventBlockDelta, 0, BlockText, "one"},
				{EventBlockEnd, 0, BlockText, "one"},
				{EventBlockStart, 1, BlockText, ""},
				{EventBlockDelta, 1, BlockText, "two"},
				{EventBlockEnd, 1, BlockText, "two"},
			},
		},
		"a whole-value block closes the text before it": {
			deltas: []Delta{
				{Block: TextBlock("calling")},
				{Block: ToolCallBlock(call)},
			},
			want: []seen{
				{EventBlockStart, 0, BlockText, ""},
				{EventBlockDelta, 0, BlockText, "calling"},
				{EventBlockEnd, 0, BlockText, "calling"},
				{EventBlockStart, 1, BlockToolCall, ""},
				{EventBlockEnd, 1, BlockToolCall, ""},
			},
		},
		"a delta carrying only metadata produces no event": {
			deltas: []Delta{
				{Usage: &Usage{Input: 10}, Model: "served-by", ID: "resp-1"},
				{Block: TextBlock("hi")},
			},
			want: []seen{
				{EventBlockStart, 0, BlockText, ""},
				{EventBlockDelta, 0, BlockText, "hi"},
				{EventBlockEnd, 0, BlockText, "hi"},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, resp, err := collectEvents(t, drive(script{deltas: tc.deltas}))
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("events = %v\nwant  = %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("event %d = %v, want %v", i, got[i], tc.want[i])
				}
			}
			if resp == nil {
				t.Fatal("no EventDone carried a response")
			}
		})
	}
}

// The aggregated response is assembled from the same deltas, and the metadata
// ones are where the model ID, the token counts and the stop reason come from.
func TestStreamAggregatesWhatTheDeltasCarried(t *testing.T) {
	call := ToolCall{ID: "c1", Name: "search", Input: `{"q":"go"}`}
	_, resp, err := collectEvents(t, drive(script{deltas: []Delta{
		{Usage: &Usage{Input: 10}, Model: "served-by", ID: "resp-1"},
		{Block: ThinkingBlock("hmm", "sig")},
		{Block: TextBlock("answer")},
		{Block: ToolCallBlock(call)},
		{Usage: &Usage{Output: 4}, StopReason: StopToolUse},
	}}))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if resp.Model != "served-by" || resp.ID != "resp-1" {
		t.Errorf("model/id = %q/%q, want what the metadata delta said", resp.Model, resp.ID)
	}
	if resp.Usage != (Usage{Input: 10, Output: 4}) {
		t.Errorf("usage = %+v, want the two halves merged rather than one erasing the other", resp.Usage)
	}
	if resp.StopReason != StopToolUse {
		t.Errorf("stop reason = %q, want %q", resp.StopReason, StopToolUse)
	}
	if resp.Text() != "answer" || resp.Thinking() != "hmm" {
		t.Errorf("content = %+v, want the thinking and the answer kept apart", resp.Content)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].ID != "c1" {
		t.Errorf("tool calls = %+v, want the one the driver sent", calls)
	}
}

// A range loop may stop at any point, and a range function that yields after
// its body returned false panics — including on the block a failure flushed.
func TestAConsumerMayBreakOnAnyEvent(t *testing.T) {
	runs := map[string]script{
		"a stream that completes": {deltas: []Delta{
			{Block: TextBlock("one")},
			{EndBlock: true},
			{Block: ToolCallBlock(ToolCall{ID: "c1", Name: "search"})},
		}},
		"a stream that fails after producing text": {
			deltas: []Delta{{Block: TextBlock("partial")}},
			err:    &Error{Kind: KindOverloaded, Message: "overloaded"},
		},
	}

	for name, run := range runs {
		t.Run(name, func(t *testing.T) {
			total, _, _ := collectEvents(t, drive(run))
			// Plus the terminal EventDone, which collectEvents keeps apart.
			for stop := range len(total) + 1 {
				var count int
				for range drive(run).Stream(context.Background(), []Message{UserMessage("hi")}) {
					if count == stop {
						break
					}
					count++
				}
			}
		})
	}
}

// A failure mid-answer must still hand back the text and the tokens it cost,
// because both were already paid for.
func TestAFailedStreamKeepsWhatItProduced(t *testing.T) {
	boom := &Error{Kind: KindOverloaded, Message: "overloaded"}
	events, resp, err := collectEvents(t, drive(script{
		deltas: []Delta{{Usage: &Usage{Input: 7}}, {Block: TextBlock("partial")}},
		err:    boom,
	}))
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the driver's failure", err)
	}
	last := events[len(events)-1]
	if last.kind != EventBlockEnd || last.text != "partial" {
		t.Errorf("last event = %v, want the interrupted block flushed", last)
	}
	if resp == nil || resp.Text() != "partial" || resp.Usage.Input != 7 {
		t.Fatalf("response = %+v, want the partial answer and its tokens", resp)
	}
	if resp.StopReason != StopError || !resp.Failed() {
		t.Errorf("stop reason = %q, want %q", resp.StopReason, StopError)
	}
}

// A driver that closes its iterator having produced nothing is a bug in the
// driver, and saying so beats handing back what looks like an empty answer.
func TestAStreamThatEndsWithoutAResponseIsReported(t *testing.T) {
	resp, err := Collect(func(yield func(Event, error) bool) {})
	if resp != nil || !IsKind(err, KindNetwork) {
		t.Fatalf("Collect = %v, %v; want a network failure and no response", resp, err)
	}
}

// A cancel is a cancel wherever it was noticed: a bare context.Canceled would
// make IsKind(err, KindCanceled) false for the one caught before the request.
func TestACancelBeforeTheRequestIsClassified(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := drive(script{deltas: []Delta{{Block: TextBlock("never")}}}).
		Complete(ctx, []Message{UserMessage("hi")})
	if !IsKind(err, KindCanceled) {
		t.Errorf("err = %v, want it classified as %q", err, KindCanceled)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want the cause still reachable with errors.Is", err)
	}
}

// Repair never sees the system prompt — it is not a message — so the prompt
// would reach the wire carrying bytes a provider rejects.
func TestPrepareCleansTheSystemPrompt(t *testing.T) {
	d := &scripted{scripts: []script{{deltas: []Delta{{Block: TextBlock("ok")}}}}}
	c := NewClientWithDriver(d, stubModel())

	if _, err := c.Complete(context.Background(), []Message{UserMessage("hi")},
		WithSystem("be brief"+string([]byte{0xff}))); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	got := d.reqs[0].System
	if !utf8.ValidString(got) || !strings.HasPrefix(got, "be brief") {
		t.Errorf("system = %q, want the invalid byte replaced and the text kept", got)
	}
}

// Headers layer the way every other setting does: the client's, then the
// call's over them, name by name. Restating a whole set to change one of them
// is what a caller would otherwise build a second client to avoid.
func TestACallsHeadersLayerOverTheClients(t *testing.T) {
	base := map[string]string{"X-Tenant": "client", "X-Fixed": "kept"}
	d := &scripted{scripts: []script{{}}}
	c := NewClientWithDriver(d, stubModel(), WithHeaders(base))

	if _, err := c.Complete(context.Background(), []Message{UserMessage("hi")},
		WithHeaders(map[string]string{"X-Tenant": "call"}),
		WithHeaders(map[string]string{"X-Turn": "1"})); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	want := map[string]string{"X-Tenant": "call", "X-Fixed": "kept", "X-Turn": "1"}
	if got := d.reqs[0].Headers; !maps.Equal(got, want) {
		t.Errorf("Headers = %v, want %v", got, want)
	}
	// And the client's own map is not the request's, or one turn's header
	// would be on every turn after it.
	if base["X-Tenant"] != "client" || len(base) != 2 {
		t.Errorf("the client's headers = %v, want them untouched by the call", base)
	}
}

// The conversation is the same conversation whichever model is asked, which is
// what Agent.SetClient and Inference.Client are for. A reasoning model leaves
// its own state in it — a signed thinking block, an opaque reasoning item —
// and the next model is usually not the one that produced them.
//
// That state is the model's, not the caller's, so it is dropped for a model
// that cannot replay it. Refusing instead would make switching model mid-
// conversation impossible for exactly the models where it matters most, and
// would refuse over something no caller put there or can take out.
func TestModelStateThatCannotBeReplayedIsDroppedNotRefused(t *testing.T) {
	// Anthropic's own turn: thinking with the signature that proves it.
	history := []Message{
		UserMessage("think about it"),
		{Role: RoleAssistant, Content: Content{
			ThinkingBlock("step one, step two", "sig-from-anthropic"),
			TextBlock("the answer"),
		}},
		UserMessage("and now?"),
	}

	for _, tc := range []struct {
		name  string
		model Model
	}{
		{"a Chat endpoint that cannot carry it", Model{ID: "m", API: APIOpenAIChat}},
		{"the Responses protocol, which replays reasoning items instead",
			Model{ID: "m", API: APIOpenAIResponses}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := &scripted{scripts: []script{{deltas: []Delta{{Block: TextBlock("fine")}}}}}
			c := NewClientWithDriver(d, tc.model)

			if _, err := c.Complete(context.Background(), history); err != nil {
				t.Fatalf("switching to this model failed: %v", err)
			}

			sent := d.reqs[0].Messages
			for _, m := range sent {
				for _, b := range m.Content {
					if b.Type == BlockThinking {
						t.Errorf("a thinking block this model cannot replay was sent anyway")
					}
				}
			}
			// And what the caller actually said is still there.
			if len(sent) != 3 || sent[1].Text() != "the answer" {
				t.Errorf("the conversation came out as %d messages, want the three that went in", len(sent))
			}
		})
	}
}
