package auth

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
)

// The default way to show a person what a sign-in needs from them.
//
// A host application that has its own UI supplies its own oauth.Interaction
// instead; this is what a command-line program gets for free.

// ConsoleInteraction prints the instruction to stderr and tries to open a
// browser.
//
// Stderr, not stdout: a CLI whose output is being piped must not have a
// sign-in prompt land in the middle of its data. Opening the browser is
// best-effort — over SSH or in a container there is nothing to open, and the
// printed URL is then the whole instruction, which is why it is always
// printed rather than only on failure.
func ConsoleInteraction() oauth.Interaction {
	return oauth.InteractionFunc(func(ctx context.Context, p oauth.Prompt) error {
		if p.UserCode != "" {
			fmt.Fprintf(os.Stderr, "\nOpen %s and enter the code:  %s\n", p.URL, p.UserCode)
		} else {
			fmt.Fprintf(os.Stderr, "\nOpen this page to sign in:\n  %s\n", p.URL)
		}
		if !p.ExpiresAt.IsZero() {
			fmt.Fprintf(os.Stderr, "This expires in %s.\n", time.Until(p.ExpiresAt).Round(time.Minute))
		}
		openBrowser(p.URL)
		return nil
	})
}

// openBrowser is best-effort and deliberately ignores its outcome: there is
// nothing useful to do when a machine has no browser, and the URL has already
// been printed.
func openBrowser(target string) {
	if target == "" {
		return
	}
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", target)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", target)
	default:
		cmd = exec.Command("xdg-open", target)
	}
	_ = cmd.Start()
}
