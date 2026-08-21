package auth

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/ai/catalog"
)

// Copilot signs in with GitHub's device grant, then exchanges the resulting
// GitHub token for a Copilot API token that lives about half an hour. The
// GitHub token is what gets stored; the short-lived one is minted per session
// and renewed underneath the caller.
//
// The client identifier and endpoints below are GitHub Copilot's published
// public client, taken from the Copilot editor integrations rather than from
// a specification.
const copilotClientID = "Iv1.b507a08c87ecfe98"

// copilotEndpoints is a struct rather than four constants so a test can point
// the flow at a stub. Nothing else varies them.
type copilotEndpoints struct {
	device, token, exchange, api string
}

var copilotDefaults = copilotEndpoints{
	device:   "https://github.com/login/device/code",
	token:    "https://github.com/login/oauth/access_token",
	exchange: "https://api.github.com/copilot_internal/v2/token",
	api:      "https://api.individual.githubcopilot.com",
}

var copilotFlow = newCopilotFlow(copilotDefaults)

func newCopilotFlow(e copilotEndpoints) flow {
	return flow{
		method: "device code",
		login: func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error) {
			cfg := oauth.Config{ClientID: copilotClientID, Scopes: []string{"read:user"}, HTTPClient: client}
			token, err := oauth.Device(ctx, cfg, oauth.DeviceEndpoints{
				Code:  e.device,
				Token: e.token,
			}, ui)
			if err != nil {
				return Credential{}, err
			}
			// One exchange now, for two reasons: it is how the endpoint is
			// discovered — Copilot reveals it only after authentication, and an
			// enterprise account's differs from an individual's — and it is where
			// an account without a Copilot subscription is found out, at sign-in
			// rather than on the first request.
			api, _, _, err := copilotSessionToken(ctx, client, e, token.Access)
			if err != nil {
				return Credential{}, err
			}
			// The GitHub token does not expire, so it is what persists.
			// Storing the short-lived Copilot token instead would mean signing
			// in again every half hour.
			return Credential{Access: token.Access, Endpoint: api}, nil
		},
		token: func(ctx context.Context, client *http.Client, c Credential) (string, time.Time, Credential, error) {
			api, expires, token, err := copilotSessionToken(ctx, client, e, c.Access)
			if err != nil {
				return "", time.Time{}, c, err
			}
			// The endpoint is worth persisting — Copilot only reveals it
			// after authentication, and an enterprise account's differs from
			// an individual's. The short-lived token is not: it would be stale
			// before the next run.
			c.Endpoint = api
			return token, expires, c, nil
		},
	}
}

// copilotSessionToken exchanges the stored GitHub token for the short-lived
// one the Copilot API accepts, and reports where that API lives — Copilot
// tells you its endpoint only after you have authenticated, and an enterprise
// account's differs from an individual's.
func copilotSessionToken(ctx context.Context, client *http.Client, e copilotEndpoints, githubToken string) (api string, expires time.Time, token string, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.exchange, nil)
	if err != nil {
		return "", time.Time{}, "", err
	}
	req.Header.Set("Authorization", "Bearer "+githubToken)
	req.Header.Set("Accept", "application/json")
	for k, v := range catalog.CopilotHeaders {
		req.Header.Set(k, v)
	}

	res, err := client.Do(req)
	if err != nil {
		return "", time.Time{}, "", err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return "", time.Time{}, "", err
	}
	if res.StatusCode >= 400 {
		return "", time.Time{}, "", &oauth.Error{
			Code:        "copilot_token_denied",
			Description: "GitHub declined to issue a Copilot token; the account may not have a subscription",
			Status:      res.StatusCode,
		}
	}

	var out struct {
		Token     string `json:"token"`
		ExpiresAt int64  `json:"expires_at"`
		Endpoints struct {
			API string `json:"api"`
		} `json:"endpoints"`
	}
	if err := json.Unmarshal(body, &out); err != nil || out.Token == "" {
		return "", time.Time{}, "", &oauth.Error{
			Code: "invalid_response", Description: "no Copilot token in the response", Status: res.StatusCode,
		}
	}
	api = out.Endpoints.API
	if api == "" {
		api = e.api
	}
	if out.ExpiresAt > 0 {
		expires = time.Unix(out.ExpiresAt, 0)
	}
	return api, expires, out.Token, nil
}
