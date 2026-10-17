package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"golang.org/x/oauth2"
)

// codexVendor is the catalog row the ChatGPT subscription signs into.
const codexVendor = "openai-codex"

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

func newCodexFlow(cfg oauth2.Config) flow {
	return flow{
		method: "browser (PKCE)",
		login: func(ctx context.Context, client *http.Client, ui Interaction) (Credential, error) {
			token, err := codeGrant(ctx, &cfg, ui)
			if err != nil {
				return Credential{}, err
			}
			return Credential{
				Access:    token.AccessToken,
				Refresh:   token.RefreshToken,
				ExpiresAt: token.Expiry,
			}, nil
		},
		token: func(ctx context.Context, client *http.Client, c Credential) (string, time.Time, Credential, error) {
			if !c.Expired() {
				return c.Access, c.ExpiresAt, c, nil
			}
			// A token with no access half is one TokenSource refreshes at once;
			// a provider that does not rotate the refresh token gets it back.
			token, err := cfg.TokenSource(ctx, &oauth2.Token{RefreshToken: c.Refresh}).Token()
			if err != nil {
				return "", time.Time{}, c, err
			}
			c.Access, c.Refresh, c.ExpiresAt = token.AccessToken, token.RefreshToken, token.Expiry
			return c.Access, c.ExpiresAt, c, nil
		},
		identify: func(h http.Header, token, session string) {
			h.Set("session_id", session)
			// The backend rejects a request that does not name the account
			// the subscription belongs to.
			if account := codexAccountID(token); account != "" {
				h.Set("chatgpt-account-id", account)
			}
		},
	}
}

// codexAccountID reads the ChatGPT account out of an access token's claims,
// without verifying it: the token was issued to us over TLS, and the account
// only says which subscription a request is billed to.
func codexAccountID(token string) string {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return ""
	}
	var claims struct {
		Auth struct {
			AccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if json.Unmarshal(payload, &claims) != nil {
		return ""
	}
	return claims.Auth.AccountID
}
