// Command agent-session keeps a conversation across processes. Run it twice
// and the second run remembers the first.
//
// The agent knows nothing about storage, and the store knows nothing about
// agents. What joins them is the event stream: a Recorder consumes the same
// events you draw from, and session.Open folds what was recorded back into a
// conversation to hand the agent. Three lines carry the whole thing —
//
//	rec, history, _ := session.Open(ctx, store, id)   // restore
//	a.SetMessages(history)                            // seed
//	rec.Handle(e)                                     // record, in your loop
//
// — and one more matters the first time you compact: SetMessages is outside
// the fold, so a replaced history has to be announced. See compact below.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/agent-session "what is a goroutine?"
//	go run ./examples/agent-session "and how do they leak?"   # remembers
//	go run ./examples/agent-session -list
//	go run ./examples/agent-session -new "starting over"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/agent/session"
	"github.com/genai-io/sdk-go/pkg/agent/session/jsonl"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
)

func main() {
	model := flag.String("model", "anthropic/claude-sonnet-5", "model reference, vendor/id")
	dir := flag.String("dir", filepath.Join(os.TempDir(), "agent-session-example"), "where sessions live")
	resume := flag.String("resume", "", "session id to continue; the most recent one by default")
	fresh := flag.Bool("new", false, "start a new session instead of continuing one")
	list := flag.Bool("list", false, "list the sessions on disk and exit")
	keep := flag.Int("keep", 0, "compact to the last N messages before answering (0 = never)")
	flag.Parse()

	if err := run(*model, *dir, *resume, *fresh, *list, *keep, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintf(os.Stderr, "\n%v\n", err)
		os.Exit(1)
	}
}

func run(model, dir, resume string, fresh, list bool, keep int, prompt string) error {
	ctx := context.Background()

	// A store is a directory. Nothing about it is agent-specific: it holds
	// entries, and Close is what flushes the metadata it keeps in memory.
	store, err := jsonl.Open(dir)
	if err != nil {
		return err
	}
	defer store.Close()

	if list {
		return listSessions(ctx, store)
	}
	if prompt == "" {
		return fmt.Errorf("usage: agent-session [flags] <prompt>")
	}

	// Which session to continue. An empty id tells session.Open to start one.
	if !fresh && resume == "" {
		sessions, err := store.List(ctx) // most recently updated first
		if err != nil {
			return err
		}
		if len(sessions) > 0 {
			resume = sessions[0].ID
		}
	}
	if fresh {
		resume = ""
	}

	// Restore. What comes back is the conversation, folded out of the entries
	// that were recorded — the agent is handed a plain []ai.Message and never
	// learns where it came from.
	rec, history, err := session.Open(ctx, store, resume)
	if err != nil {
		return err
	}
	if len(history) > 0 {
		fmt.Printf("\033[2mcontinuing %s · %d messages\033[0m\n", rec.ID(), len(history))
	} else {
		fmt.Printf("\033[2mnew session %s\033[0m\n", rec.ID())
	}

	client, err := auth.Client(model)
	if err != nil {
		return err
	}
	a, err := agent.New(client,
		agent.WithSystem("You are terse. Answer in one or two sentences, and use what was said earlier."),
		agent.WithTools(clock),
	)
	if err != nil {
		return err
	}
	a.SetMessages(history)

	if keep > 0 {
		compact(a, rec, keep)
	}

	// One loop, and it owns the order: record first, then draw. Handle is
	// synchronous and sits on this goroutine by design — a program that cannot
	// afford that forwards to a buffer of its own.
	var end agent.TurnEnd
	for e, err := range a.Run(ctx, ai.UserMessage(prompt)) {
		if err != nil {
			return err
		}
		rec.Handle(e)
		render(e)
		if v, ok := e.(agent.TurnEnd); ok {
			end = v
		}
	}

	// Recording never fails a turn — a store that broke mid-conversation
	// should not take the answer down with it — so ask afterwards.
	if err := rec.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "\nthe session was not fully written: %v\n", err)
	}
	fmt.Printf("\n\n\033[2m— %s · %d tokens · %s\033[0m\n", end.StopReason, end.Usage.Total(), rec.ID())
	return nil
}

// compact is the case worth knowing about. A session is the fold of the
// messages that were announced, and SetMessages announces nothing — the agent
// cannot tell being compacted from being handed a history in the first place.
// So the caller who replaced it says so. Forget the second line and the next
// restore hands back everything this one threw away.
func compact(a *agent.Agent, rec *session.Recorder, keep int) {
	msgs := a.Messages()
	if len(msgs) <= keep {
		return
	}
	// A real one summarises with a model call; this one just drops the middle.
	kept := msgs[len(msgs)-keep:]

	a.SetMessages(kept)
	rec.Snapshot(kept) // the fold starts here now

	fmt.Printf("\033[2mcompacted %d messages to %d\033[0m\n", len(msgs), len(kept))
}

func listSessions(ctx context.Context, store *jsonl.Store) error {
	sessions, err := store.List(ctx)
	if err != nil {
		return err
	}
	if len(sessions) == 0 {
		fmt.Println("no sessions yet")
		return nil
	}
	for _, m := range sessions {
		fmt.Printf("%s  %5d entries  %s\n", m.ID, m.Entries, m.UpdatedAt.Local().Format(time.RFC822))
	}
	return nil
}

var clock = agent.ToolFunc("now", "The current date and time.",
	func(context.Context, struct{}) (agent.Result, error) {
		return agent.TextResult(time.Now().Format(time.RFC1123)), nil
	})

func render(e agent.Event) {
	switch v := e.(type) {
	case agent.MessageUpdate:
		if v.Delta.Type == ai.EventBlockDelta && v.Delta.Block.Type == ai.BlockText {
			fmt.Print(v.Delta.Block.Text)
		}
	case agent.ToolStart:
		fmt.Printf("\033[2m[%s]\033[0m ", v.Name)
	}
}
