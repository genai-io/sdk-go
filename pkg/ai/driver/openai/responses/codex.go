package responses

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3/option"

	"github.com/genai-io/sdk-go/pkg/ai"
)

const (
	codexLatestURL       = "https://registry.npmjs.org/@openai%2Fcodex/latest"
	codexFallbackVersion = "0.169.1"
	codexVersionTTL      = 24 * time.Hour
	codexVersionRetry    = 5 * time.Minute
	codexVersionTimeout  = 2 * time.Second
	codexVersionMaxBytes = 1 << 20
)

var codexVersions = &codexVersionCache{}

// Discovery never uses the provider's authenticated client or cookie jar.
var codexMetadataClient = &http.Client{Transport: &http.Transport{
	Proxy:             http.ProxyFromEnvironment,
	ForceAttemptHTTP2: true,
	IdleConnTimeout:   90 * time.Second,
}}

type codexVersionCache struct {
	mu      sync.Mutex
	version string
	expires time.Time
	pending chan struct{}
	fetch   func(context.Context) (string, error)
}

func (c *codexVersionCache) resolve(ctx context.Context) string {
	for {
		c.mu.Lock()
		version := cmp.Or(c.version, codexFallbackVersion)
		if ctx.Err() != nil || time.Now().Before(c.expires) {
			c.mu.Unlock()
			return version
		}
		if pending := c.pending; pending != nil {
			c.mu.Unlock()
			select {
			case <-ctx.Done():
				return version
			case <-pending:
				continue
			}
		}
		c.pending = make(chan struct{})
		c.mu.Unlock()

		fetch := c.fetch
		if fetch == nil {
			fetch = func(ctx context.Context) (string, error) {
				return fetchCodexVersion(ctx, codexMetadataClient, codexLatestURL)
			}
		}
		latest, err := fetch(ctx)

		c.mu.Lock()
		if err == nil {
			c.version = latest
			c.expires = time.Now().Add(codexVersionTTL)
		} else if ctx.Err() == nil {
			c.expires = time.Now().Add(codexVersionRetry)
		}
		version = cmp.Or(c.version, codexFallbackVersion)
		close(c.pending)
		c.pending = nil
		c.mu.Unlock()
		return version
	}
}

func fetchCodexVersion(ctx context.Context, client *http.Client, url string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, codexVersionTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	res, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return "", fmt.Errorf("codex version: http %d", res.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(res.Body, codexVersionMaxBytes+1))
	if err != nil {
		return "", err
	}
	if len(raw) > codexVersionMaxBytes {
		return "", fmt.Errorf("codex version: metadata exceeds %d bytes", codexVersionMaxBytes)
	}
	var metadata struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return "", fmt.Errorf("codex version: invalid metadata")
	}
	if !stableCodexVersion(metadata.Version) {
		return "", fmt.Errorf("codex version: invalid stable version")
	}
	return metadata.Version, nil
}

func stableCodexVersion(version string) bool {
	parts := strings.Split(version, ".")
	if len(parts) != 3 || len(version) > 32 {
		return false
	}
	for _, part := range parts {
		if part == "" || len(part) > 1 && part[0] == '0' {
			return false
		}
		for _, digit := range part {
			if digit < '0' || digit > '9' {
				return false
			}
		}
	}
	return true
}

// codexModels reads the lineup a ChatGPT subscription is entitled to.
func (d *driver) codexModels(ctx context.Context) ([]ai.Model, error) {
	version := d.compat.CodexClientVersion
	if version == "" {
		version = d.versions.resolve(ctx)
	}
	var listing struct {
		Models []struct {
			Slug                     string `json:"slug"`
			DisplayName              string `json:"display_name"`
			Description              string `json:"description"`
			ContextWindow            int    `json:"context_window"`
			ShowInPicker             *bool  `json:"show_in_picker"`
			DefaultReasoningLevel    string `json:"default_reasoning_level"`
			SupportedReasoningLevels []struct {
				Effort string `json:"effort"`
			} `json:"supported_reasoning_levels"`
		} `json:"models"`
	}
	err := d.client.Get(ctx, "models", nil, &listing, option.WithQuery("client_version", version))
	if err != nil {
		return nil, fail.Wrap(err)
	}
	out := make([]ai.Model, 0, len(listing.Models))
	for _, m := range listing.Models {
		// A null show_in_picker means shown; only an explicit false hides one.
		if m.Slug == "" || (m.ShowInPicker != nil && !*m.ShowInPicker) {
			continue
		}
		model := ai.Model{
			ID:            m.Slug,
			Name:          cmp.Or(m.DisplayName, m.Slug),
			Description:   m.Description,
			API:           ai.APIOpenAIResponses,
			Vendor:        d.model.Vendor,
			ContextWindow: m.ContextWindow,
		}
		for _, level := range m.SupportedReasoningLevels {
			if level.Effort == "" {
				continue
			}
			model.Reasoning = append(model.Reasoning, ai.ReasoningLevel{
				Effort:  ai.Effort(level.Effort),
				Value:   level.Effort,
				Default: level.Effort == m.DefaultReasoningLevel,
			})
		}
		out = append(out, model)
	}
	if len(out) == 0 {
		return nil, &ai.Error{Driver: name, Kind: ai.KindUnknown, Message: "the subscription catalog listed no models"}
	}
	return out, nil
}
