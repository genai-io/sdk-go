package auth

import (
	"context"
	"net/http"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
)

// The Codex client is OpenAI's published public client for its CLI. The
// redirect port is fixed because a public client's redirect must be one the
// provider has registered.
const codexClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

var codexScopes = []string{"openid", "profile", "email", "offline_access"}

var codexDefaults = oauth.CodeEndpoints{
	Authorize: "https://auth.openai.com/oauth/authorize",
	Token:     "https://auth.openai.com/oauth/token",
	Redirect:  "http://localhost:1455/auth/callback",
}

func init() { RegisterFlow("openai-codex", newCodexFlow(codexDefaults)) }

func newCodexFlow(e oauth.CodeEndpoints) Flow {
	return Flow{
		Method: "browser (PKCE)",
		Login: func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error) {
			cfg := oauth.Config{ClientID: codexClientID, Scopes: codexScopes, HTTPClient: client}
			token, err := oauth.Code(ctx, cfg, e, ui)
			if err != nil {
				return Credential{}, err
			}
			return Credential{
				Access:    token.Access,
				Refresh:   token.Refresh,
				ExpiresAt: token.Expires,
			}, nil
		},
		Token: func(ctx context.Context, client *http.Client, c Credential) (string, time.Time, Credential, error) {
			if !c.Expired() {
				return c.Access, c.ExpiresAt, c, nil
			}
			cfg := oauth.Config{ClientID: codexClientID, HTTPClient: client}
			token, err := oauth.Refresh(ctx, cfg, e.Token, c.Refresh)
			if err != nil {
				return "", time.Time{}, c, err
			}
			c.Access, c.Refresh, c.ExpiresAt = token.Access, token.Refresh, token.Expires
			return c.Access, c.ExpiresAt, c, nil
		},
	}
}
