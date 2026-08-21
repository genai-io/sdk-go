package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
)

// A few vendors authenticate a person rather than a service: there is no API
// key to paste, only a subscription and a browser. Without an interactive
// sign-in those vendors are unreachable, which is why their catalog entries
// used to say "supply the exchanged token yourself".
//
// The two grants live in package oauth and know nothing about any provider.
// What is provider-specific — the endpoints, the client identifiers, and in
// Copilot's case a second exchange for a short-lived API token — is here.

// flow is one vendor's interactive sign-in.
type flow struct {
	// method describes the grant for a caller listing what is available.
	method string
	// login runs the grant and returns what should be stored.
	login func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error)
	// token returns the value to present on a request, renewing as needed.
	//
	// The three results are deliberately separate. present is what goes in the
	// header; expires is that value's own lifetime, which for Copilot is half
	// an hour while the credential behind it never expires; updated is what
	// should be persisted, which changes only when a refresh rotated
	// something.
	token func(ctx context.Context, client *http.Client, c Credential) (present string, expires time.Time, updated Credential, err error)
}

var flows = map[string]flow{
	"copilot":      copilotFlow,
	"openai-codex": codexFlow,
}

// Interactive reports whether a vendor signs in through a browser rather than
// an API key, and by which grant. The second result is empty for a vendor that
// takes an API key.
func Interactive(vendorID string) (method string, ok bool) {
	f, ok := flows[vendorID]
	if !ok {
		return "", false
	}
	return f.method, true
}

// LoginOptions configures Login.
type LoginOptions struct {
	// Store is where the credential is kept. Nil uses the default file store.
	Store Store
	// Interaction shows the person what to do. Nil prints to stderr and tries
	// to open a browser.
	Interaction oauth.Interaction
	// HTTPClient is used for the sign-in exchanges.
	HTTPClient *http.Client
}

// Login runs a vendor's interactive sign-in and stores the result.
//
// It blocks until the person finishes in a browser, or the context ends. The
// returned credential is also saved, so the next run does not sign in again.
func Login(ctx context.Context, vendorID string, opts LoginOptions) (Credential, error) {
	f, ok := flows[vendorID]
	if !ok {
		if _, known := catalog.Find(vendorID); !known {
			return Credential{}, &UnknownVendorError{Vendor: vendorID}
		}
		return Credential{}, fmt.Errorf("auth: %s does not sign in interactively; set its API key variable instead", vendorID)
	}

	ui := opts.Interaction
	if ui == nil {
		ui = ConsoleInteraction()
	}
	client := opts.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}

	credential, err := f.login(ctx, client, ui)
	if err != nil {
		return Credential{}, err
	}
	credential.Vendor = vendorID

	store, err := resolveStore(opts.Store)
	if err != nil {
		return credential, err
	}
	if err := store.Save(credential); err != nil {
		// The sign-in worked; only persisting it did not. Handing back the
		// credential lets the caller carry on with this session.
		return credential, err
	}
	return credential, nil
}

// Logout forgets a vendor's stored credential. It does not revoke it with the
// provider, which only the provider's own settings page can do.
func Logout(vendorID string, store Store) error {
	s, err := resolveStore(store)
	if err != nil {
		return err
	}
	return s.Delete(vendorID)
}

// DefaultStore is where Login, Config, and Provider keep credentials when
// they are not handed a store.
//
// A package-level variable because the choice belongs to the application and
// has to hold for calls that take no options — llm.Config resolution among
// them. Set it once at start-up. Nil means a file under the user's config
// directory, which is what a CLI wants and what a server should override.
var DefaultStore Store

func resolveStore(s Store) (Store, error) {
	if s != nil {
		return s, nil
	}
	if DefaultStore != nil {
		return DefaultStore, nil
	}
	return NewFileStore("")
}

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

// ── GitHub Copilot ──

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

// ── ChatGPT / Codex subscription ──

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

var codexFlow = newCodexFlow(codexDefaults)

func newCodexFlow(e oauth.CodeEndpoints) flow {
	return flow{
		method: "browser (PKCE)",
		login: func(ctx context.Context, client *http.Client, ui oauth.Interaction) (Credential, error) {
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
		token: func(ctx context.Context, client *http.Client, c Credential) (string, time.Time, Credential, error) {
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

// Transport injects a vendor's current token into every request.
//
// A RoundTripper rather than a fixed header because Copilot's token lasts
// about half an hour — well inside a long session — so it cannot be resolved
// once at start-up. It overwrites any Authorization the driver set, which is
// how a driver that insists on a placeholder key still ends up sending a real
// one.
type Transport struct {
	base   http.RoundTripper
	source *tokenSource
}

func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := t.source.token(req.Context())
	if err != nil {
		return nil, err
	}
	// Cloning: a RoundTripper must not modify the request it is given.
	clone := req.Clone(req.Context())
	clone.Header.Set("Authorization", "Bearer "+token)
	return t.base.RoundTrip(clone)
}

// HTTPClient returns a client that presents a vendor's stored credential,
// renewing it as it expires.
func HTTPClient(vendorID string, c Credential, store Store, base *http.Client) (*http.Client, error) {
	f, ok := flows[vendorID]
	if !ok {
		return nil, fmt.Errorf("auth: %s does not sign in interactively", vendorID)
	}
	if base == nil {
		base = &http.Client{Timeout: 10 * time.Minute}
	}
	inner := base.Transport
	if inner == nil {
		inner = http.DefaultTransport
	}
	out := *base
	out.Transport = &Transport{
		base: inner,
		source: &tokenSource{
			vendor: vendorID,
			flow:   f,
			store:  store,
			client: &http.Client{Timeout: 30 * time.Second},
			cred:   c,
		},
	}
	return &out, nil
}

// tokenSource keeps a vendor's presentable token current.
//
// Copilot's expires about every half hour, which is well inside a long
// session, so renewal cannot happen only at start-up. This caches the current
// token, renews it when it runs out, and persists whatever the renewal
// changed.
type tokenSource struct {
	vendor string
	flow   flow
	store  Store
	client *http.Client

	mu      sync.Mutex
	cred    Credential
	current string
	expires time.Time
}

func (s *tokenSource) token(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.current != "" && (s.expires.IsZero() || time.Now().Add(time.Minute).Before(s.expires)) {
		return s.current, nil
	}
	present, expires, updated, err := s.flow.token(ctx, s.client, s.cred)
	if err != nil {
		return "", err
	}
	if updated != s.cred {
		updated.Vendor = s.vendor
		s.cred = updated
		// A session that cannot write to disk can still run; losing the
		// refreshed credential costs one extra sign-in later, and failing the
		// request costs the turn.
		if s.store != nil {
			_ = s.store.Save(updated)
		}
	}
	s.current, s.expires = present, expires
	return present, nil
}
