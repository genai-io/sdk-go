package agent_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
)

// ToolEnd is emitted the moment the tool lands, before PostTool runs, so a
// reader never waits on hooks. The consequence is pinned here rather than left
// to be discovered: a hook that replaces a result changes what the model is
// told and not what the stream reported, and the two are allowed to differ.
func TestToolEndCarriesTheToolsOwnResult(t *testing.T) {
	echo := agent.ToolFunc("echo", "Echo it.",
		func(_ context.Context, _ struct{}) (agent.Result, error) {
			return agent.TextResult("raw"), nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("call-1", "echo", `{}`),
		text("done"),
	}}, agent.WithTools(echo), agent.WithHooks(agent.Hook{
		PostTool: func(context.Context, agent.PostToolContext) (*agent.Result, error) {
			replacement := agent.TextResult("replaced")
			return &replacement, nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	var ended *agent.ToolEnd
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok {
			ended = &v
		}
	}
	if ended == nil {
		t.Fatal("no ToolEnd was emitted")
	}
	if got := ended.Result.Text(); got != "raw" {
		t.Errorf("ToolEnd result = %q, want the tool's own — the emit waited on the hook", got)
	}

	results := a.Messages()[2].ToolResults()
	if len(results) != 1 || results[0].Text() != "replaced" {
		t.Errorf("the model was told %+v, want the hook's replacement", results)
	}
}

// Two orders, both load-bearing: completion order for the spans, source order
// for what lands in the conversation.
func TestParallelToolsEndInCompletionOrderButRecordInSourceOrder(t *testing.T) {
	release := map[string]chan struct{}{
		"a": make(chan struct{}),
		"b": make(chan struct{}),
	}
	wait := agent.ToolFunc("wait", "Wait for a signal.",
		func(ctx context.Context, args struct {
			Key string `json:"key"`
		}) (agent.Result, error) {
			<-release[args.Key]
			return agent.TextResult("done: " + args.Key), nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "wait", Input: `{"key":"a"}`})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "2", Name: "wait", Input: `{"key":"b"}`})},
			{StopReason: ai.StopToolUse},
		},
		text("both done"),
	}}, agent.WithTools(wait))

	// b finishes first, though the model asked for a first.
	go func() {
		time.Sleep(10 * time.Millisecond)
		close(release["b"])
		time.Sleep(10 * time.Millisecond)
		close(release["a"])
	}()

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	var ended []string
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok {
			ended = append(ended, v.ID)
		}
	}
	if want := []string{"2", "1"}; !slices.Equal(ended, want) {
		t.Errorf("ToolEnd order = %v, want completion order %v", ended, want)
	}

	results := a.Messages()[2].ToolResults()
	var recorded []string
	for _, r := range results {
		recorded = append(recorded, r.ToolCallID)
	}
	if want := []string{"1", "2"}; !slices.Equal(recorded, want) {
		t.Errorf("recorded order = %v, want source order %v", recorded, want)
	}
}

func TestAnUnknownToolIsReportedRatherThanFatal(t *testing.T) {
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("call-1", "nosuchtool", `{}`),
		text("sorry"),
	}})

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("an unknown tool should not fail the turn: %v", err)
	}
	results := a.Messages()[2].ToolResults()
	if len(results) != 1 || !results[0].IsError {
		t.Fatalf("unknown tool produced %+v", results)
	}
}

func TestBadArgumentsAreCaughtBeforeTheToolRuns(t *testing.T) {
	strict := agent.ToolFunc("strict", "Needs a count.",
		func(_ context.Context, args struct {
			Count int `json:"count"`
		}) (agent.Result, error) {
			t.Fatal("the tool ran on arguments that do not match its schema")
			return agent.Result{}, nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("call-1", "strict", `{"count":"not a number"}`),
		text("retrying"),
	}}, agent.WithTools(strict))

	if _, err := collect(t, a, ai.UserMessage("go")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if results := a.Messages()[2].ToolResults(); len(results) != 1 || !results[0].IsError {
		t.Fatalf("schema violation produced %+v", results)
	}
}

// A tool that takes a while shows its work: Report reaches the consumer as
// ToolUpdate, and the finished result still arrives on ToolEnd.
func TestAToolReportsWhileItWorks(t *testing.T) {
	slow := agent.ToolFunc("build", "Build the project.",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			agent.Report(ctx, agent.TextResult("compiling…"))
			agent.Report(ctx, agent.TextResult("linking…"))
			return agent.TextResult("built"), nil
		})

	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("c1", "build", `{}`),
		text("done"),
	}}, agent.WithTools(slow))

	var partials []string
	var final string
	for e, err := range a.Run(context.Background(), ai.UserMessage("build it")) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		switch v := e.(type) {
		case agent.ToolUpdate:
			partials = append(partials, v.Partial.Text())
		case agent.ToolEnd:
			final = v.Result.Text()
		}
	}

	if want := []string{"compiling…", "linking…"}; !slices.Equal(partials, want) {
		t.Errorf("reported %q, want %q", partials, want)
	}
	if final != "built" {
		t.Errorf("ToolEnd result = %q, want the finished one", final)
	}
}

// A tool that reports where nobody is listening — outside an exchange, or in a
// test — is not a tool that panics.
func TestReportOutsideAToolDoesNothing(t *testing.T) {
	agent.Report(context.Background(), agent.TextResult("into the void"))
}

// A tool declares its arguments once, as a Go type, and that type is both what
// the model is shown and what the arguments are decoded into. An argument the
// schema does not have is the model inventing something: it comes back as a
// tool error the model can read and correct, not as a field quietly dropped on
// the way to code that then acts on a request it never received.
func TestAToolRejectsAnArgumentItsSchemaDoesNotHave(t *testing.T) {
	type args struct {
		Path string `json:"path"`
	}
	var ran bool
	tool := agent.ToolFunc("read", "read a file", func(_ context.Context, a args) (agent.Result, error) {
		ran = true
		return agent.TextResult(a.Path), nil
	})

	if _, err := tool.Run(context.Background(), ai.ToolCall{
		Name: "read", Input: `{"path":"main.go","recursive":true}`,
	}); err == nil {
		t.Fatal("an unknown argument was accepted; the model would never learn it was ignored")
	}
	if ran {
		t.Error("the tool ran on arguments that did not match its schema")
	}

	// The same call without the invention is fine, and an absent optional
	// field keeps whatever the target already held.
	got, err := tool.Run(context.Background(), ai.ToolCall{Name: "read", Input: `{"path":"main.go"}`})
	if err != nil {
		t.Fatalf("valid arguments were rejected: %v", err)
	}
	if got.Text() != "main.go" {
		t.Errorf("got %q, want the decoded path", got.Text())
	}
}

// The two halves of a tool are the schema the model reads and the function
// that answers it, and ToolFunc derives the first from the second's argument
// type — so they cannot come to describe different things.
func TestAToolsSchemaComesFromItsArgumentType(t *testing.T) {
	type args struct {
		City string `json:"city" description:"which city"`
	}
	tool := agent.ToolFunc("weather", "look up the weather", func(context.Context, args) (agent.Result, error) {
		return agent.Result{}, nil
	})

	schema := tool.Schema()
	if schema.Name != "weather" || schema.Description != "look up the weather" {
		t.Errorf("Schema = %+v, want the name and description it was given", schema)
	}
	props, _ := schema.DefinitionMap()["properties"].(map[string]any)
	if _, ok := props["city"]; !ok {
		t.Errorf("properties = %v, want the field from the argument type", props)
	}
	if err := schema.Validate(`{"city":42}`); err == nil {
		t.Error("a wrongly typed argument passed validation")
	}
}

// Sequential is a promise that a tool never runs beside another, and it holds
// through a caller's own wrapper as long as the mark stays outermost — which
// is the rule Sequential documents, because no marker survives a decorator
// that embeds the Tool interface.
func TestASequentialToolRunsAloneThroughADecorator(t *testing.T) {
	var live, peak atomic.Int64
	slow := agent.ToolFunc("touch", "touch shared state",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			n := live.Add(1)
			for {
				old := peak.Load()
				if n <= old || peak.CompareAndSwap(old, n) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			live.Add(-1)
			return agent.TextResult("ok"), nil
		})

	d := &scripted{Scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "touch", Input: "{}"})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "2", Name: "touch", Input: "{}"})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "3", Name: "touch", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("done"),
	}}
	a := newAgent(t, d, agent.WithTools(agent.Sequential(logged{slow})))

	for _, err := range a.Run(context.Background(), ai.UserMessage("touch it three times")) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := peak.Load(); got != 1 {
		t.Errorf("%d of them ran at once; Sequential promised 1", got)
	}
}

// logged is the kind of wrapper a caller writes: it passes everything through
// and adds nothing the agent knows about.
type logged struct{ agent.Tool }

// FromAI is the bridge that lets a tool written against pkg/ai run in an
// agent without being rewritten. Its schema and its answer both have to
// survive the crossing, and a failing one has to fail the agent's way — as a
// result the model can read, not as a turn that dies.
func TestAToolWrittenForTheClientRunsInAnAgent(t *testing.T) {
	type args struct {
		City string `json:"city" description:"which city"`
	}
	plain := ai.ToolFunc("weather", "Look up the weather.",
		func(_ context.Context, a args) (string, error) {
			if a.City == "" {
				return "", errors.New("no city given")
			}
			return "mild in " + a.City, nil
		})

	lifted := agent.FromAI(plain)
	if got := lifted.Schema(); got.Name != "weather" || got.Description != "Look up the weather." {
		t.Errorf("Schema = %+v, want the ai.Tool's", got)
	}
	if _, ok := lifted.Schema().DefinitionMap()["properties"]; !ok {
		t.Error("the derived schema did not cross over")
	}

	got, err := lifted.Run(context.Background(),
		ai.ToolCall{ID: "1", Name: "weather", Input: `{"city":"Delhi"}`})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Text() != "mild in Delhi" {
		t.Errorf("Run = %q, want what the ai.Tool returned", got.Text())
	}

	// A tool offered with nothing to run it is a configuration mistake, and
	// says so rather than panicking.
	empty := agent.FromAI(ai.Tool{Schema: ai.Schema{Name: "hollow"}})
	if _, err := empty.Run(context.Background(), ai.ToolCall{Name: "hollow"}); err == nil {
		t.Error("a tool with no Run was called without complaint")
	}
}

// A tool is the caller's code, but it runs on a goroutine this package
// created — the one place a panic cannot be recovered by whoever wrote it.
// Unrecovered it takes the whole process down mid-conversation. A failing tool
// already has a way to fail, so a panic becomes that: the model is told, and
// the turn carries on.
func TestAPanickingToolDoesNotTakeTheProcessWithIt(t *testing.T) {
	boom := agent.ToolFunc("boom", "panics",
		func(context.Context, struct{}) (agent.Result, error) {
			var m map[string]int
			m["nil map write"] = 1 //nolint:staticcheck // SA5000: this panic is the subject
			return agent.TextResult("unreachable"), nil
		})
	fine := agent.ToolFunc("fine", "works",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.TextResult("still here"), nil
		})

	d := &scripted{Scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "1", Name: "boom", Input: "{}"})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "2", Name: "fine", Input: "{}"})},
			{StopReason: ai.StopToolUse},
		},
		text("carried on"),
	}}
	a := newAgent(t, d, agent.WithTools(boom, fine))

	var failed *agent.PanicError
	ended := map[string]bool{}
	for e, err := range a.Run(context.Background(), ai.UserMessage("run both")) {
		if err != nil {
			t.Fatalf("the turn died: %v", err)
		}
		if v, ok := e.(agent.ToolEnd); ok {
			ended[v.Name] = true
			if v.Name == "boom" {
				if !errors.As(v.Err, &failed) {
					t.Fatalf("boom ended with %v, want a *agent.PanicError", v.Err)
				}
			}
		}
	}

	if !ended["boom"] || !ended["fine"] {
		t.Errorf("tools that ended: %v, want both — one panic stalled the batch", ended)
	}
	if failed == nil {
		t.Fatal("the panic was never reported")
	}
	if len(failed.Stack) == 0 {
		t.Error("the stack was not kept; a recovered panic with no stack is undebuggable")
	}
	// The model is told one line, not a stack trace.
	if got := agent.ResultText(agent.Result{}, failed); strings.Contains(got, "goroutine") {
		t.Errorf("the model was told %q — that is a stack trace", got)
	}

	// And the conversation went on: the model answered after the tool results.
	last := a.Messages()[len(a.Messages())-1]
	if last.Text() != "carried on" {
		t.Errorf("the conversation ended at %q, want the model's answer", last.Text())
	}
}

// SetTools takes effect on the next inference, and a batch answers the last
// one. A gate that narrowed the toolset while vetting the first call would
// otherwise turn the second into "no tool named that".
func TestAToolsetChangedMidBatchStillAnswersTheBatch(t *testing.T) {
	var a *agent.Agent
	read := agent.ToolFunc("read", "Read a file.",
		func(context.Context, struct{}) (agent.Result, error) { return agent.TextResult("contents"), nil })
	write := agent.ToolFunc("write", "Write a file.",
		func(context.Context, struct{}) (agent.Result, error) { return agent.TextResult("written"), nil })

	d := &scripted{Scripts: [][]ai.Delta{
		{
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c1", Name: "read", Input: `{}`})},
			{Block: ai.ToolCallBlock(ai.ToolCall{ID: "c2", Name: "write", Input: `{}`})},
			{StopReason: ai.StopToolUse},
		},
		text("done"),
	}, Keep: true}
	a = newAgent(t, d, agent.WithTools(read, write), agent.WithHooks(agent.Hook{
		// The gate takes the write tool away while the batch is being vetted.
		PreTool: func(_ context.Context, c agent.PreToolContext) (agent.Decision, error) {
			a.SetTools(read)
			return agent.Decision{}, nil
		},
	}))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	for _, e := range events {
		if v, ok := e.(agent.ToolEnd); ok && v.Err != nil {
			t.Errorf("%s failed with %v; the batch was vetted against a toolset that changed under it",
				v.Name, v.Err)
		}
	}

	// And the change did land, on the inference after the batch.
	sent := d.Sent()
	if len(sent) != 2 {
		t.Fatalf("the model was called %d times, want 2", len(sent))
	}
	if names := toolNames(sent[1].Tools); len(names) != 1 || names[0] != "read" {
		t.Errorf("the second call offered %v, want just the tool SetTools left", names)
	}
}

// A report is droppable rather than something a tool can be held up by: it
// goes if the exchange is listening and is dropped if it is not, without ever
// blocking the tool or panicking — a report that arrives late included.
func TestAReportNobodyIsListeningForIsDropped(t *testing.T) {
	var late context.Context
	tool := agent.ToolFunc("scan", "Scan things.",
		func(ctx context.Context, _ struct{}) (agent.Result, error) {
			late = ctx
			agent.Report(ctx, agent.TextResult("halfway"))
			return agent.TextResult("done scanning"), nil
		})
	a := newAgent(t, &scripted{Scripts: [][]ai.Delta{
		toolCall("c1", "scan", `{}`),
		text("finished"),
	}}, agent.WithTools(tool))

	events, err := collect(t, a, ai.UserMessage("go"))
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	updates, ended := 0, false
	for _, e := range events {
		switch e.(type) {
		case agent.ToolUpdate:
			if ended {
				t.Error("a report arrived after the call it belongs to was closed")
			}
			updates++
		case agent.ToolEnd:
			ended = true
		}
	}
	if updates != 1 {
		t.Errorf("the exchange reported %d partial results, want the 1 the tool sent", updates)
	}

	// The exchange is over and its channel is nobody's now. Far more reports
	// than it could hold, from a goroutine that would deadlock if any blocked.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 100 {
			agent.Report(late, agent.TextResult("still going"))
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("reporting into a finished exchange blocked the tool")
	}
}

// A tool that looked at something answers with a picture, and the picture is
// what the model is told — the loop no longer flattens a result to its text on
// the way. What a log or a session keeps is that text, and a result that is
// only an image says so rather than reading as nothing.
func TestAToolsImageReachesTheModel(t *testing.T) {
	shot := agent.ToolFunc("screenshot", "Look at the page.",
		func(context.Context, struct{}) (agent.Result, error) {
			return agent.Result{Content: ai.Content{
				ai.TextBlock("the page, rendered"),
				ai.ImageBlock(ai.Image{MediaType: "image/png", Data: "AAAA"}),
			}}, nil
		})
	driver := &scripted{Scripts: [][]ai.Delta{
		toolCall("c1", "screenshot", `{}`),
		text("a login form"),
	}, Keep: true}
	a := newAgent(t, driver, agent.WithTools(shot))

	if _, err := collect(t, a, ai.UserMessage("what does it look like?")); err != nil {
		t.Fatalf("turn failed: %v", err)
	}

	sent := driver.Sent()[1].Messages
	results := sent[len(sent)-1].ToolResults()
	if len(results) != 1 {
		t.Fatalf("the second call carried %d tool results, want the one that ran", len(results))
	}
	if !results[0].Content.HasImages() {
		t.Error("the model was told the text and not the picture")
	}
	if got := results[0].Text(); got != "the page, rendered" {
		t.Errorf("the text beside it = %q", got)
	}
}

// The three answers a call comes to, in both forms: what the model is told,
// and what a record keeps.
func TestAResultSaysSomethingEvenWhenItSaysNothing(t *testing.T) {
	image := agent.Result{Content: ai.Content{
		ai.ImageBlock(ai.Image{MediaType: "image/png", Data: "AAAA"}),
	}}
	for _, tc := range []struct {
		name   string
		result agent.Result
		err    error
		want   string
	}{
		{"text", agent.TextResult("what it found"), nil, "what it found"},
		{"failure", agent.Result{}, errors.New("the file is gone"), "the file is gone"},
		{"neither", agent.Result{}, nil, "(no output)"},
		{"a picture", image, nil, "(image)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := agent.ResultText(tc.result, tc.err); got != tc.want {
				t.Errorf("ResultText = %q, want %q", got, tc.want)
			}
			if got := agent.ResultContent(tc.result, tc.err); len(got) == 0 {
				t.Error("ResultContent came back empty; several endpoints reject that")
			}
		})
	}
}
