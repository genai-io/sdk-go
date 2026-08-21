package auth

// These tests are in-package because they point the vendor flows at stub
// servers, which is what makes an end-to-end exercise of the exchange and the
// renewal possible without a GitHub or OpenAI account.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/llm/auth/oauth"
	"github.com/genai-io/sdk-go/pkg/llm/catalog"
)

func silent() oauth.Interaction {
	return oauth.InteractionFunc(func(context.Context, oauth.Prompt) error { return nil })
}

// ── credential store ──

func TestFileStoreRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "credentials.json")
	store, err := NewFileStore(path)
	if err != nil {
		t.Fatal(err)
	}

	// Nothing stored is not an error; it is the normal first run.
	if _, found, err := store.Load("copilot"); err != nil || found {
		t.Fatalf("Load on an empty store = %v, %v", found, err)
	}

	want := Credential{
		Vendor:    "copilot",
		Access:    "gho_x",
		Refresh:   "r",
		ExpiresAt: time.Now().Add(time.Hour).Truncate(time.Second),
		Endpoint:  "https://api.individual.githubcopilot.com",
	}
	if err := store.Save(want); err != nil {
		t.Fatal(err)
	}

	// Reading through a second store proves it reached disk, not just memory.
	reopened, err := NewFileStore(path)
	if err != nil {
		t.Fatal(err)
	}
	got, found, err := reopened.Load("copilot")
	if err != nil || !found {
		t.Fatalf("Load = %v, %v", found, err)
	}
	if got.Access != want.Access || got.Endpoint != want.Endpoint || !got.ExpiresAt.Equal(want.ExpiresAt) {
		t.Errorf("Load = %+v, want %+v", got, want)
	}

	if err := store.Save(Credential{Vendor: "openai-codex", Access: "b"}); err != nil {
		t.Fatal(err)
	}
	if list, _ := store.List(); len(list) != 2 {
		t.Errorf("List = %v, want both vendors", list)
	}

	if err := store.Delete("copilot"); err != nil {
		t.Fatal(err)
	}
	if _, found, _ := store.Load("copilot"); found {
		t.Error("credential survived Delete")
	}
	// Deleting what is not there is not a failure — logging out twice is fine.
	if err := store.Delete("copilot"); err != nil {
		t.Errorf("second Delete: %v", err)
	}
}

// The file holds long-lived tokens; on a shared machine the mode is the only
// thing standing between them and another account.
func TestFileStoreWritesPrivately(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("no POSIX permission bits")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "conf", "credentials.json")
	store, _ := NewFileStore(path)
	if err := store.Save(Credential{Vendor: "copilot", Access: "secret"}); err != nil {
		t.Fatal(err)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode := fi.Mode().Perm(); mode != 0o600 {
		t.Errorf("file mode = %o, want 600", mode)
	}
	di, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if mode := di.Mode().Perm(); mode != 0o700 {
		t.Errorf("directory mode = %o, want 700", mode)
	}
	// The temporary file used for the atomic rename must not be left behind
	// holding a copy of the secret.
	entries, _ := os.ReadDir(filepath.Dir(path))
	if len(entries) != 1 {
		t.Errorf("directory holds %d files, want only the credential file", len(entries))
	}
}

func TestFileStoreRejectsACredentialWithNoVendor(t *testing.T) {
	store, _ := NewFileStore(filepath.Join(t.TempDir(), "c.json"))
	if err := store.Save(Credential{Access: "x"}); err == nil {
		t.Error("expected an error saving a credential with no vendor")
	}
}

func TestDefaultStorePathFollowsXDG(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "/xdg")
	path, err := DefaultStorePath()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join("/xdg", "genai-io", "credentials.json"); path != want {
		t.Errorf("DefaultStorePath = %q, want %q", path, want)
	}
}

// ── GitHub Copilot ──

// copilotStub stands in for GitHub's device endpoints and the Copilot token
// exchange. It hands out a new short-lived token on every exchange so renewal
// is observable.
type copilotStub struct {
	server    *httptest.Server
	endpoints copilotEndpoints
	exchanges atomic.Int32
	lifetime  time.Duration
	deny      bool
}

func newCopilotStub(t *testing.T) *copilotStub {
	t.Helper()
	s := &copilotStub{lifetime: 30 * time.Minute}
	mux := http.NewServeMux()
	mux.HandleFunc("/device/code", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"device_code":"dc","user_code":"WXYZ-9999",
			"verification_uri":"https://github.test/login/device","expires_in":900,"interval":1}`)
	})
	mux.HandleFunc("/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"access_token":"gho_stored","token_type":"bearer"}`)
	})
	mux.HandleFunc("/copilot/token", func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer gho_stored" {
			t.Errorf("exchange Authorization = %q", got)
		}
		// Copilot refuses a client that does not identify itself as an editor.
		if r.Header.Get("Editor-Version") == "" {
			t.Error("exchange did not identify the editor")
		}
		w.Header().Set("Content-Type", "application/json")
		if s.deny {
			w.WriteHeader(http.StatusForbidden)
			fmt.Fprint(w, `{"message":"no subscription"}`)
			return
		}
		n := s.exchanges.Add(1)
		fmt.Fprintf(w, `{"token":"copilot-token-%d","expires_at":%d,
			"endpoints":{"api":"https://api.enterprise.githubcopilot.test"}}`,
			n, time.Now().Add(s.lifetime).Unix())
	})
	s.server = httptest.NewServer(mux)
	t.Cleanup(s.server.Close)
	s.endpoints = copilotEndpoints{
		device:   s.server.URL + "/device/code",
		token:    s.server.URL + "/oauth/access_token",
		exchange: s.server.URL + "/copilot/token",
		api:      "https://api.individual.githubcopilot.test",
	}
	return s
}

func TestCopilotLoginStoresTheDurableTokenAndTheEndpoint(t *testing.T) {
	stub := newCopilotStub(t)
	withFlow(t, "copilot", newCopilotFlow(stub.endpoints))

	store := NewMemoryStore()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cred, err := Login(ctx, "copilot", LoginOptions{Store: store, Interaction: silent()})
	if err != nil {
		t.Fatalf("Login: %v", err)
	}

	// The GitHub token is what persists: the Copilot token is good for half an
	// hour, so storing it would mean signing in again every half hour.
	if cred.Access != "gho_stored" {
		t.Errorf("stored Access = %q, want the durable GitHub token", cred.Access)
	}
	if !cred.ExpiresAt.IsZero() {
		t.Errorf("stored ExpiresAt = %v, want no expiry on the GitHub token", cred.ExpiresAt)
	}
	// The endpoint is only discoverable after authenticating, and an
	// enterprise account's is not the default one.
	if cred.Endpoint != "https://api.enterprise.githubcopilot.test" {
		t.Errorf("Endpoint = %q, want the account's own endpoint", cred.Endpoint)
	}
	if saved, found, _ := store.Load("copilot"); !found || saved.Access != "gho_stored" {
		t.Errorf("Login did not persist the credential: %+v", saved)
	}
}

// An account without a subscription should be found out at sign-in, not on the
// first request in the middle of some later turn.
func TestCopilotLoginReportsAMissingSubscription(t *testing.T) {
	stub := newCopilotStub(t)
	stub.deny = true
	withFlow(t, "copilot", newCopilotFlow(stub.endpoints))

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_, err := Login(ctx, "copilot", LoginOptions{Store: NewMemoryStore(), Interaction: silent()})
	var oerr *oauth.Error
	if !errors.As(err, &oerr) || oerr.Code != "copilot_token_denied" {
		t.Fatalf("err = %v, want the exchange refusal", err)
	}
}

// Copilot's token lasts about half an hour — well inside a long session — so
// the transport has to renew it underneath the caller.
func TestTransportRenewsAnExpiringToken(t *testing.T) {
	stub := newCopilotStub(t)
	// Short enough that the first token is already inside the renewal margin.
	stub.lifetime = 30 * time.Second
	withFlow(t, "copilot", newCopilotFlow(stub.endpoints))

	var seen []string
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusOK)
	}))
	defer api.Close()

	store := NewMemoryStore()
	client, err := HTTPClient("copilot", Credential{Vendor: "copilot", Access: "gho_stored"}, store, nil)
	if err != nil {
		t.Fatal(err)
	}

	for i := range 2 {
		req, _ := http.NewRequest(http.MethodGet, api.URL, nil)
		// A driver that insists on setting a placeholder key must not win.
		req.Header.Set("Authorization", "Bearer placeholder")
		res, err := client.Do(req)
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		res.Body.Close()
	}

	if len(seen) != 2 || seen[0] != "Bearer copilot-token-1" || seen[1] != "Bearer copilot-token-2" {
		t.Errorf("headers = %v, want two distinct minted tokens", seen)
	}
	// The endpoint the exchange revealed is worth keeping; the short-lived
	// token is not, since it would be stale before the next run.
	saved, _, _ := store.Load("copilot")
	if saved.Endpoint != "https://api.enterprise.githubcopilot.test" {
		t.Errorf("persisted Endpoint = %q", saved.Endpoint)
	}
	if saved.Access != "gho_stored" {
		t.Errorf("persisted Access = %q, want the durable token", saved.Access)
	}
}

func TestTransportReusesATokenThatIsStillGood(t *testing.T) {
	stub := newCopilotStub(t)
	withFlow(t, "copilot", newCopilotFlow(stub.endpoints))

	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer api.Close()

	client, _ := HTTPClient("copilot", Credential{Vendor: "copilot", Access: "gho_stored"}, nil, nil)
	for range 3 {
		req, _ := http.NewRequest(http.MethodGet, api.URL, nil)
		res, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
	}
	// Minting a token per request would be three round-trips of pure latency
	// on every turn.
	if n := stub.exchanges.Load(); n != 1 {
		t.Errorf("%d exchanges for three requests, want 1", n)
	}
}

// ── ChatGPT / Codex ──

func TestCodexLoginAndRefresh(t *testing.T) {
	var issued atomic.Int32
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		n := issued.Add(1)
		if r.Form.Get("grant_type") == "authorization_code" && r.Form.Get("code_verifier") == "" {
			t.Error("the exchange carried no verifier; an intercepted code would be usable")
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"access_token":"at-%d","refresh_token":"rt-%d","expires_in":3600}`, n, n)
	}))
	defer tokenServer.Close()

	redirect := freeLoopback(t)
	withFlow(t, "openai-codex", newCodexFlow(oauth.CodeEndpoints{
		Authorize: "https://auth.test/authorize",
		Token:     tokenServer.URL,
		Redirect:  redirect,
	}))

	ui := oauth.InteractionFunc(func(ctx context.Context, p oauth.Prompt) error {
		u, err := url.Parse(p.URL)
		if err != nil {
			return err
		}
		go func() {
			res, err := http.Get(redirect + "?code=c&state=" + url.QueryEscape(u.Query().Get("state"))) //nolint:noctx // test callback
			if err == nil {
				res.Body.Close()
			}
		}()
		return nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	store := NewMemoryStore()
	cred, err := Login(ctx, "openai-codex", LoginOptions{Store: store, Interaction: ui})
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if cred.Access != "at-1" || cred.Refresh != "rt-1" {
		t.Fatalf("credential = %+v", cred)
	}
	if cred.ExpiresAt.IsZero() {
		t.Error("expires_in was not turned into a deadline")
	}

	// A stored credential that has run out must renew rather than send a token
	// the provider will reject.
	stale := Credential{Vendor: "openai-codex", Access: "at-1", Refresh: "rt-1", ExpiresAt: time.Now().Add(-time.Hour)}
	client, err := HTTPClient("openai-codex", stale, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	var sent string
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sent = r.Header.Get("Authorization")
	}))
	defer api.Close()
	req, _ := http.NewRequest(http.MethodGet, api.URL, nil)
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()

	if sent != "Bearer at-2" {
		t.Errorf("sent %q, want the renewed token", sent)
	}
	// Losing the rotated refresh token means the next expiry forces a sign-in.
	saved, _, _ := store.Load("openai-codex")
	if saved.Refresh != "rt-2" {
		t.Errorf("persisted Refresh = %q, want the rotated one", saved.Refresh)
	}
}

// ── wiring ──

func TestInteractiveReportsTheGrant(t *testing.T) {
	for vendor, want := range map[string]string{
		"copilot":      "device code",
		"openai-codex": "browser (PKCE)",
	} {
		if method, ok := Interactive(vendor); !ok || method != want {
			t.Errorf("Interactive(%q) = %q, %v; want %q", vendor, method, ok, want)
		}
	}
	// A key-based vendor has no browser sign-in, and saying so is how a caller
	// knows to prompt for a key instead.
	if method, ok := Interactive("openai"); ok {
		t.Errorf("Interactive(\"openai\") = %q, %v; want no interactive grant", method, ok)
	}
}

func TestLoginRejectsAVendorThatTakesAKey(t *testing.T) {
	_, err := Login(context.Background(), "openai", LoginOptions{Store: NewMemoryStore()})
	if err == nil {
		t.Fatal("expected an error")
	}
	var unknown *UnknownVendorError
	if errors.As(err, &unknown) {
		t.Fatalf("err = %v, want a 'set its API key' error, not an unknown vendor", err)
	}

	if _, err := Login(context.Background(), "nope", LoginOptions{Store: NewMemoryStore()}); !errors.As(err, &unknown) {
		t.Errorf("err = %v, want UnknownVendorError", err)
	}
}

// Reaching for a subscription vendor before signing in should say so, not 401
// later with something opaque.
func TestConfigForASignedOutVendor(t *testing.T) {
	withStore(t, NewMemoryStore())
	_, err := Config("copilot/gpt-5.1")
	var out *NotSignedInError
	if !errors.As(err, &out) {
		t.Fatalf("err = %v, want NotSignedInError", err)
	}
	if _, err := Provider("copilot"); !errors.As(err, &out) {
		t.Errorf("Provider err = %v, want NotSignedInError", err)
	}
}

func TestConfigForASignedInVendorUsesTheStoredEndpoint(t *testing.T) {
	stub := newCopilotStub(t)
	withFlow(t, "copilot", newCopilotFlow(stub.endpoints))
	store := NewMemoryStore()
	withStore(t, store)
	if err := store.Save(Credential{
		Vendor:   "copilot",
		Access:   "gho_stored",
		Endpoint: "https://api.enterprise.githubcopilot.test",
	}); err != nil {
		t.Fatal(err)
	}

	cfg, err := Config("copilot/gpt-5.1")
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.BaseURL != "https://api.enterprise.githubcopilot.test" {
		t.Errorf("BaseURL = %q, want the endpoint the sign-in resolved to", cfg.BaseURL)
	}
	// Building a Config must not talk to the network: the token is minted
	// lazily, on the first request.
	if n := stub.exchanges.Load(); n != 0 {
		t.Errorf("Config performed %d token exchanges, want 0", n)
	}
	if cfg.HTTPClient == nil {
		t.Error("Config returned no client, so nothing would present the token")
	}

	// A signed-in subscription vendor counts as available; a signed-out one
	// does not.
	if !slicesContainsVendor(Available(), "copilot") {
		t.Error("copilot missing from Available after signing in")
	}
	if err := Logout("copilot", store); err != nil {
		t.Fatal(err)
	}
	if slicesContainsVendor(Available(), "copilot") {
		t.Error("copilot still in Available after Logout")
	}
}

// ── helpers ──

// withFlow points a vendor at stub endpoints for the duration of a test.
func withFlow(t *testing.T, vendor string, f flow) {
	t.Helper()
	previous, existed := flows[vendor]
	flows[vendor] = f
	t.Cleanup(func() {
		if existed {
			flows[vendor] = previous
			return
		}
		delete(flows, vendor)
	})
}

func withStore(t *testing.T, s Store) {
	t.Helper()
	previous := DefaultStore
	DefaultStore = s
	t.Cleanup(func() { DefaultStore = previous })
}

func slicesContainsVendor(vendors []catalog.Vendor, id string) bool {
	for _, v := range vendors {
		if v.ID == id {
			return true
		}
	}
	return false
}

func freeLoopback(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return fmt.Sprintf("http://127.0.0.1:%d/auth/callback", port)
}

// A flow for a vendor the catalog does not carry is unreachable, and a vendor
// that lists key variables while signing in through a browser would tell a
// caller to look for a key that does not exist.
func TestEveryInteractiveVendorIsInTheCatalogAndNeedsNoKey(t *testing.T) {
	for id := range flows {
		v, ok := catalog.Find(id)
		if !ok {
			t.Errorf("flow %q has no catalog vendor", id)
			continue
		}
		if len(v.KeyEnv) > 0 {
			t.Errorf("%s signs in interactively but also lists key variables %v", id, v.KeyEnv)
		}
		if v.Note == "" {
			t.Errorf("%s has no note saying how to sign in", id)
		}
	}
}
