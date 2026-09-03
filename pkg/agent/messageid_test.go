package agent_test

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// counter is a generator whose output says how many times it ran, so a test
// can tell one message from the next and count the calls at the same time.
func counter() func() string {
	n := 0
	return func() string {
		n++
		return fmt.Sprintf("m%d", n)
	}
}

func ids(msgs []ai.Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.ID)
	}
	return out
}

// Everything the loop makes is named, not only what the caller handed in: a
// conversation half of which can be pointed at is one an application cannot
// build a store or an editable transcript on.
func TestEveryMessageInTheConversationIsNamed(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo the argument.",
		func(_ context.Context, a struct {
			S string `json:"s"`
		}) (agent.Result, error) {
			return agent.TextResult(a.S), nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("c1", "echo", `{"s":"hi"}`),
		text("done"),
	}}, agent.WithTools(echo), agent.WithMessageIDs(counter()))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	// The input, the model's tool call, the batch of results, and the answer.
	want := []string{"m1", "m2", "m3", "m4"}
	if got := ids(a.Messages()); !slices.Equal(got, want) {
		t.Errorf("conversation is named %v, want %v", got, want)
	}

	// And what was announced is what is held: a consumer folding MessageAdded
	// has to end up with the same names the agent has.
	var announced []string
	for _, e := range events {
		if v, ok := e.(agent.MessageAdded); ok {
			announced = append(announced, v.Message.ID)
		}
	}
	if !slices.Equal(announced, want) {
		t.Errorf("MessageAdded announced %v, want %v", announced, want)
	}
}

// A name that arrived with the message is kept. This is what makes a restore
// work: the conversation comes back pointing at the entries it was stored as,
// rather than being renamed into a transcript nobody has a record of.
func TestAMessageThatArrivedNamedKeepsItsName(t *testing.T) {
	stored := []ai.Message{
		{ID: "from-storage", Role: ai.RoleUser, Content: ai.TextContent("earlier")},
		{Role: ai.RoleAssistant, Content: ai.TextContent("unnamed, from before IDs")},
	}

	// Either order: seeding a conversation and asking for names are two
	// options, and an agent built with them the other way round is the same
	// agent.
	for _, tc := range []struct {
		name string
		opts []agent.Option
	}{
		{"messages first", []agent.Option{
			agent.WithMessages(stored), agent.WithMessageIDs(counter())}},
		{"ids first", []agent.Option{
			agent.WithMessageIDs(counter()), agent.WithMessages(stored)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}}, tc.opts...)

			got := ids(a.Messages())
			if want := []string{"from-storage", "m1"}; !slices.Equal(got, want) {
				t.Errorf("conversation is named %v, want %v", got, want)
			}
		})
	}
}

// Naming is the agent's business and the slice is the caller's. Writing names
// into the messages they still hold would rename rows under an interface that
// had already drawn them.
func TestNamingDoesNotTouchTheCallersSlice(t *testing.T) {
	mine := []ai.Message{{Role: ai.RoleUser, Content: ai.TextContent("hi")}}

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
		agent.WithMessageIDs(counter()))
	a.SetMessages(mine)

	if mine[0].ID != "" {
		t.Errorf("the caller's message was renamed to %q", mine[0].ID)
	}
	if a.Messages()[0].ID == "" {
		t.Error("the agent's copy went unnamed")
	}
}

// Off by default, and then every ID is empty. An agent nothing outside the
// loop has to point at should not be calling a generator on every message.
func TestMessagesAreUnnamedUnlessAsked(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}})

	if _, err := outcome(t, a, ai.UserMessage("hi")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for i, m := range a.Messages() {
		if m.ID != "" {
			t.Errorf("message %d is named %q with no generator given", i, m.ID)
		}
	}
}

// A conversation a hook replaced is named on the way in, like any other. A
// compaction whose summary had no name would put the one message a session
// most needs to point at beyond pointing at.
func TestAReplacedConversationIsNamedToo(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{text("ok")}},
		agent.WithMessageIDs(counter()),
		agent.WithHooks(agent.Hook{
			PreStep: func(context.Context, agent.PreStepContext) ([]ai.Message, error) {
				return []ai.Message{ai.UserMessage("(the summary)")}, nil
			},
		}))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	at := slices.IndexFunc(events, func(e agent.Event) bool {
		_, ok := e.(agent.MessagesReplaced)
		return ok
	})
	if at < 0 {
		t.Fatal("the hook replaced the conversation and nothing was announced")
	}
	for i, m := range events[at].(agent.MessagesReplaced).Messages {
		if m.ID == "" {
			t.Errorf("replacement message %d went unnamed", i)
		}
	}
}
