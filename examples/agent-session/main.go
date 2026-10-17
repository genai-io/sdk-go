// Command agent-session keeps a conversation across processes. Run it twice
// and the second run remembers the first.
//
// The agent knows nothing about storage, and the store knows nothing about
// agents. A conversation is a []ai.Message: load one, hand it to the agent,
// and save what the agent holds whenever its conversation changes —
//
//	w, _ := store.Writer(ctx, id)            // this process's writer
//	history, _ := store.Load(ctx, id)
//	a.SetMessages(history)
//	for e, err := range a.Run(ctx, msg) {
//		switch e.(type) {
//		case agent.MessageAdded, agent.MessagesReplaced:
//			w.Save(a.Messages())
//		}
//	}
//
// — compaction included: the next save is the compacted conversation. See
// compact below.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/agent-session "what is a goroutine?"
//	go run ./examples/agent-session "and how do they leak?"   # remembers
//	go run ./examples/agent-session -list
//	go run ./examples/agent-session -new "starting over"
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/genai-io/sdk-go/pkg/agent"
	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"
	"github.com/genai-io/sdk-go/pkg/session"

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

func run(model, dir, resume string, fresh, list bool, keep int, prompt string) (err error) {
	ctx := context.Background()

	// A store is a directory, one file per conversation.
	store, err := session.Open(dir)
	if err != nil {
		return err
	}

	if list {
		return listSessions(ctx, store)
	}
	if prompt == "" {
		return fmt.Errorf("usage: agent-session [flags] <prompt>")
	}

	// Which session to continue: the one asked for, or the most recent, or
	// a new one. An id is any name the store accepts.
	id := resume
	if !fresh && id == "" {
		sessions, err := store.List(ctx) // most recently written first
		if err != nil {
			return err
		}
		if len(sessions) > 0 {
			id = sessions[0].ID
		}
	}
	if fresh || id == "" {
		id = time.Now().Format("20060102-150405")
	}

	// One writer per session: a second process on the same id gets
	// session.ErrBusy here rather than interleaving its lines with ours.
	w, err := store.Writer(ctx, id)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, w.Close()) }()

	history, err := store.Load(ctx, id)
	if err != nil {
		return err
	}
	if len(history) > 0 {
		fmt.Printf("\033[2mcontinuing %s · %d messages\033[0m\n", id, len(history))
	} else {
		fmt.Printf("\033[2mnew session %s\033[0m\n", id)
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
		compact(a, keep)
	}

	var end agent.TurnEnd
	for e, err := range a.Run(ctx, ai.UserMessage(prompt)) {
		if err != nil {
			// Outside-the-turn failures only; the turn reports its own on
			// TurnEnd.
			return err
		}
		render(e)
		switch v := e.(type) {
		case agent.MessageAdded, agent.MessagesReplaced:
			// Saved as it changes, so what a failed or interrupted turn
			// got through is still the conversation.
			if err := w.Save(a.Messages()); err != nil {
				return err
			}
		case agent.TurnEnd:
			end = v
		}
	}
	fmt.Printf("\n\n\033[2m— %s · %d tokens · %s\033[0m\n", end.StopReason, end.Usage.Total(), id)
	return end.Err
}

// compact throws most of the conversation away. The store needs telling
// nothing: the next save is the compacted conversation.
func compact(a *agent.Agent, keep int) {
	msgs := a.Messages()
	if len(msgs) <= keep {
		return
	}
	// A real one summarises with a model call, and runs from a PreStep hook so
	// that it can fire mid-turn, when a long run of tools is what filled the
	// window. This one just drops the middle, between turns.
	kept := msgs[len(msgs)-keep:]
	a.SetMessages(kept)

	fmt.Printf("\033[2mcompacted %d messages to %d\033[0m\n", len(msgs), len(kept))
}

func listSessions(ctx context.Context, store *session.Store) error {
	sessions, err := store.List(ctx)
	if err != nil {
		return err
	}
	if len(sessions) == 0 {
		fmt.Println("no sessions yet")
		return nil
	}
	for _, s := range sessions {
		fmt.Printf("%s  %s\n", s.ID, s.ModTime.Local().Format(time.RFC822))
	}
	return nil
}

var clock = agent.ToolFunc("now", "The current date and time.",
	func(context.Context, struct{}) (agent.Result, error) {
		return agent.TextResult(time.Now().Format(time.RFC1123)), nil
	})

func render(e agent.Event) {
	switch v := e.(type) {
	case agent.InferenceUpdate:
		fmt.Print(v.Text())
	case agent.ToolStart:
		fmt.Printf("\033[2m[%s]\033[0m ", v.Name)
	}
}
