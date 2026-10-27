package auth

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/genai-io/sdk-go/pkg/ai"
	"golang.org/x/oauth2"
)

// NotSignedInError reports a vendor that needs an interactive sign-in and has
// no stored credential.
type NotSignedInError struct{ Vendor string }

func (e *NotSignedInError) Error() string {
	return fmt.Sprintf("auth: not signed in to %s; run auth.Login (%s)", e.Vendor, flows[e.Vendor].method)
}

// errNoVendor reports a credential that names no vendor. Every Store rejects
// one: it is keyed by vendor, so a credential without one could never be
// loaded back, and accepting it would silently lose a sign-in.
var errNoVendor = errors.New("auth: credential has no vendor")

func refreshError(vendor string, err error) error {
	rejected, ok := errors.AsType[*oauth2.RetrieveError](err)
	if !ok || rejected.Response == nil || (rejected.Response.StatusCode != http.StatusBadRequest && rejected.Response.StatusCode != http.StatusUnauthorized) {
		return err
	}
	code := rejected.ErrorCode
	if code == "" {
		var body struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if json.Unmarshal(rejected.Body, &body) == nil {
			code = body.Error.Code
		}
	}
	switch code {
	case "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused":
		return &ai.Error{Driver: vendor, Kind: ai.KindAuth, Status: rejected.Response.StatusCode,
			Code: code, Message: "session can no longer be refreshed; sign in again", Err: err}
	}
	return err
}
