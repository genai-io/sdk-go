package auth

import (
	"cmp"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/genai-io/sdk-go/pkg/ai/catalog"
	"golang.org/x/oauth2"
)

// Copilot signs in with GitHub's device grant, then exchanges the resulting
// GitHub token for a Copilot API token that lives about half an hour. The
// GitHub token is what gets stored; the short-lived one is minted per session
// and renewed underneath the caller.
const copilotClientID = "Iv1.b507a08c87ecfe98"

// copilotVendor is the catalog row this flow signs into, and the row the
// editor headers below are read from.
const copilotVendor = "copilot"

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

func newCopilotFlow(e copilotEndpoints) flow {
	return flow{
		method: "device code",
		login: func(ctx context.Context, client *http.Client, ui Interaction) (Credential, error) {
			cfg := &oauth2.Config{
				ClientID: copilotClientID,
				Scopes:   []string{"read:user"},
				Endpoint: oauth2.Endpoint{DeviceAuthURL: e.device, TokenURL: e.token, AuthStyle: oauth2.AuthStyleInParams},
			}
			token, err := deviceGrant(ctx, cfg, ui)
			if err != nil {
				return Credential{}, err
			}
			// One exchange now, for two reasons: it is how the endpoint is
			// discovered — Copilot reveals it only after authentication, and an
			// enterprise account's differs from an individual's — and it is where
			// an account without a Copilot subscription is found out, at sign-in
			// rather than on the first request.
			api, _, _, err := copilotSessionToken(ctx, client, e, token.AccessToken)
			if err != nil {
				return Credential{}, err
			}
			// The GitHub token does not expire, so it is what persists.
			// Storing the short-lived Copilot token instead would mean signing
			// in again every half hour.
			return Credential{Access: token.AccessToken, BaseURL: api}, nil
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
			c.BaseURL = api
			return token, expires, c, nil
		},
		identify: func(h http.Header, _, _ string) {
			h.Set("X-Request-Id", newUUID())
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
	// The same editor headers the vendor row carries: this exchange refuses a
	// caller that does not identify itself as one, exactly as the API does.
	// Except the API version: on api.github.com it selects a REST API
	// version, and the Copilot API's date is not one.
	if v, ok := catalog.Find(copilotVendor); ok {
		for name, value := range v.Headers {
			req.Header.Set(name, value)
		}
	}
	req.Header.Del("X-GitHub-Api-Version")

	res, err := client.Do(req)
	if err != nil {
		return "", time.Time{}, "", err
	}
	defer func() { _ = res.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return "", time.Time{}, "", err
	}
	if res.StatusCode >= 400 {
		return "", time.Time{}, "", fmt.Errorf("auth: GitHub declined to issue a Copilot token (http %d); "+
			"the account may not have a subscription", res.StatusCode)
	}

	var out struct {
		Token     string `json:"token"`
		ExpiresAt int64  `json:"expires_at"`
		Endpoints struct {
			API string `json:"api"`
		} `json:"endpoints"`
	}
	if err := json.Unmarshal(body, &out); err != nil || out.Token == "" {
		return "", time.Time{}, "", fmt.Errorf("auth: no Copilot token in the response (http %d)", res.StatusCode)
	}
	api = cmp.Or(out.Endpoints.API, e.api)
	if out.ExpiresAt > 0 {
		expires = time.Unix(out.ExpiresAt, 0)
	}
	return api, expires, out.Token, nil
}

// newUUID returns a random UUIDv4, the shape Copilot's X-Request-Id and the
// ChatGPT backend's session_id take.
func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
