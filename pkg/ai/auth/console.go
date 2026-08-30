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

// ConsoleInteraction prints the instruction to stderr and tries to open a
// browser.
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
