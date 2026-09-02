package oauth

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// DeviceEndpoints are the two URLs a device authorization grant uses —
// RFC 8628.
type DeviceEndpoints struct {
	// Code issues the device and user codes.
	Code string
	// Token is polled until the person finishes signing in.
	Token string
}

// deviceCodeResponse is the wire shape of the code endpoint's reply. The
// specification allows either a JSON body or a form-encoded one, and providers
// differ, so both are handled.
type deviceCodeResponse struct {
	DeviceCode      string `json:"device_code"`
	UserCode        string `json:"user_code"`
	VerificationURI string `json:"verification_uri"`
	// Some providers send the alternate spelling from an early draft.
	VerificationURL         string `json:"verification_url"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
	Error                   string `json:"error"`
	ErrorDescription        string `json:"error_description"`
}

func (r deviceCodeResponse) verificationURI() string {
	// The complete form embeds the code, so a person who can open it does not
	// have to type anything.
	for _, candidate := range []string{r.VerificationURIComplete, r.VerificationURI, r.VerificationURL} {
		if candidate != "" {
			return candidate
		}
	}
	return ""
}

// Device runs the device authorization grant: ask for a code, show it to the
// person, then poll until they have finished in a browser.
func Device(ctx context.Context, cfg Config, endpoints DeviceEndpoints, ui Interaction) (Token, error) {
	code, err := requestDeviceCode(ctx, cfg, endpoints.Code)
	if err != nil {
		return Token{}, err
	}

	lifetime := code.ExpiresIn
	if lifetime <= 0 {
		lifetime = defaultDeviceLifetimeSeconds
	}
	expires := time.Now().Add(time.Duration(lifetime) * time.Second)
	if ui != nil {
		prompt := Prompt{
			URL:       code.verificationURI(),
			UserCode:  code.UserCode,
			ExpiresAt: expires,
		}
		if err := ui.Prompt(ctx, prompt); err != nil {
			return Token{}, err
		}
	}

	interval := max(time.Duration(code.Interval)*time.Second, pollFloor)
	for {
		select {
		case <-ctx.Done():
			return Token{}, ctx.Err()
		case <-time.After(interval):
		}
		if time.Now().After(expires) {
			return Token{}, &Error{Code: "expired_token", Description: "the sign-in attempt timed out"}
		}

		res, err := postForm(ctx, cfg.client(), endpoints.Token, url.Values{
			"client_id":   {cfg.ClientID},
			"device_code": {code.DeviceCode},
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		})
		if err == nil {
			return res.token(), nil
		}

		var oauthErr *Error
		if !errors.As(err, &oauthErr) {
			return Token{}, err
		}
		switch oauthErr.Code {
		case "authorization_pending":
			// The person has not finished yet. Keep waiting.
		case "slow_down":
			// The provider is telling us we are polling too fast; ignoring it
			// is how an attempt gets refused outright.
			interval += slowDownStep
		default:
			return Token{}, oauthErr
		}
	}
}

// requestDeviceCode asks for the pair of codes, tolerating both reply shapes.
func requestDeviceCode(ctx context.Context, cfg Config, endpoint string) (deviceCodeResponse, error) {
	form := url.Values{"client_id": {cfg.ClientID}}
	if len(cfg.Scopes) > 0 {
		form.Set("scope", strings.Join(cfg.Scopes, " "))
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return deviceCodeResponse{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	res, err := cfg.client().Do(req)
	if err != nil {
		return deviceCodeResponse{}, err
	}
	defer func() { _ = res.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return deviceCodeResponse{}, err
	}

	var out deviceCodeResponse
	if json.Unmarshal(body, &out) != nil {
		// Fall back to the form-encoded shape some providers still send.
		values, parseErr := url.ParseQuery(string(body))
		if parseErr != nil {
			return out, &Error{Code: "invalid_response", Description: truncate(string(body), 200), Status: res.StatusCode}
		}
		out = deviceCodeResponse{
			DeviceCode:      values.Get("device_code"),
			UserCode:        values.Get("user_code"),
			VerificationURI: values.Get("verification_uri"),
			ExpiresIn:       atoiDefault(values.Get("expires_in"), 0),
			Interval:        atoiDefault(values.Get("interval"), 0),
			Error:           values.Get("error"),
		}
	}
	if out.Error != "" {
		return out, &Error{Code: out.Error, Description: out.ErrorDescription, Status: res.StatusCode}
	}
	if out.DeviceCode == "" || out.UserCode == "" {
		return out, &Error{Code: "invalid_response", Description: "no device code was issued", Status: res.StatusCode}
	}
	return out, nil
}

// defaultDeviceLifetimeSeconds stands in when a provider states none. Fifteen
// minutes is the specification's own example and what every provider seen here
// uses.
const defaultDeviceLifetimeSeconds = 900

// pollFloor is the shortest interval Device waits between polls whatever the
// provider asks for, and slowDownStep is what it adds when the provider says
// to back off — RFC 8628's own figure. They are variables rather than
// constants only so a test can drive several polls without spending seconds on
// each; nothing in this package changes them.
var (
	pollFloor    = 1 * time.Second
	slowDownStep = 5 * time.Second
)
