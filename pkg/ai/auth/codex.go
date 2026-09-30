package auth

import (
	"context"
	"net/http"
	"time"

	"golang.org/x/oauth2"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
)

// The Codex client is OpenAI's published public client for its CLI. The
// redirect port is fixed because a public client's redirect must be one the
// provider has registered.
var codexDefaults = oauth2.Config{
	ClientID:    "app_EMoamEEZ73f0CkXaXp7hrann",
	Scopes:      []string{"openid", "profile", "email", "offline_access"},
	RedirectURL: "http://localhost:1455/auth/callback",
	Endpoint: oauth2.Endpoint{
		AuthURL:   "https://auth.openai.com/oauth/authorize",
		TokenURL:  "https://auth.openai.com/oauth/token",
		AuthStyle: oauth2.AuthStyleInParams,
	},
}

func init() { RegisterFlow("openai-codex", newCodexFlow(codexDefaults)) }

func newCodexFlow(cfg oauth2.Config) Flow {
	return Flow{
		Method: "browser (PKCE)",
		Login: func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error) {
			token, err := oauth.Code(context.WithValue(ctx, oauth2.HTTPClient, client), &cfg, ui)
			if err != nil {
				return Credential{}, err
			}
			return Credential{
				Access:    token.AccessToken,
				Refresh:   token.RefreshToken,
				ExpiresAt: token.Expiry,
			}, nil
		},
		Token: func(ctx context.Context, client *http.Client, c Credential) (string, time.Time, Credential, error) {
			if !c.Expired() {
				return c.Access, c.ExpiresAt, c, nil
			}
			// A token with no access half is one TokenSource refreshes at once;
			// a provider that does not rotate the refresh token gets it back.
			ctx = context.WithValue(ctx, oauth2.HTTPClient, client)
			token, err := cfg.TokenSource(ctx, &oauth2.Token{RefreshToken: c.Refresh}).Token()
			if err != nil {
				return "", time.Time{}, c, err
			}
			c.Access, c.Refresh, c.ExpiresAt = token.AccessToken, token.RefreshToken, token.Expiry
			return c.Access, c.ExpiresAt, c, nil
		},
	}
}
