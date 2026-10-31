package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// CodexUsageWindow is an account rate-limit window, not token billing usage.
// Duration identifies the window; primary and secondary positions do not.
type CodexUsageWindow struct {
	Duration    time.Duration
	UsedPercent float64
	ResetsAt    time.Time
}

// CodexUsage contains only windows actually reported by ChatGPT.
type CodexUsage struct {
	Windows []CodexUsageWindow
}

// FetchCodexUsage reads the signed-in account's Codex limits. client should be
// constructed with HTTPClient("openai-codex", ...), which supplies bearer and
// account headers and refreshes credentials. The caller owns timeout/caching.
// No response body is included in errors, since it may contain account data.
func FetchCodexUsage(ctx context.Context, client *http.Client) (CodexUsage, error) {
	if client == nil {
		return CodexUsage{}, fmt.Errorf("codex usage: authenticated client required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://chatgpt.com/backend-api/wham/usage", nil)
	if err != nil {
		return CodexUsage{}, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return CodexUsage{}, fmt.Errorf("codex usage: request failed")
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return CodexUsage{}, fmt.Errorf("codex usage: HTTP %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024+1))
	if err != nil || len(data) > 1024*1024 {
		return CodexUsage{}, fmt.Errorf("codex usage: unreadable response")
	}
	return parseCodexUsage(data)
}

func parseCodexUsage(data []byte) (CodexUsage, error) {
	type window struct {
		Used    *float64 `json:"used_percent"`
		Seconds int64    `json:"limit_window_seconds"`
		Reset   int64    `json:"reset_at"`
	}
	var wire struct {
		RateLimit *struct {
			Primary   *window `json:"primary_window"`
			Secondary *window `json:"secondary_window"`
		} `json:"rate_limit"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return CodexUsage{}, fmt.Errorf("codex usage: invalid response")
	}
	result := CodexUsage{}
	if wire.RateLimit == nil {
		return result, nil
	}
	for _, w := range []*window{wire.RateLimit.Primary, wire.RateLimit.Secondary} {
		if w == nil || w.Used == nil || w.Seconds <= 0 || w.Seconds > int64((1<<63-1)/int64(time.Second)) || *w.Used < 0 || *w.Used > 100 {
			continue
		}
		entry := CodexUsageWindow{Duration: time.Duration(w.Seconds) * time.Second, UsedPercent: *w.Used}
		if w.Reset > 0 {
			entry.ResetsAt = time.Unix(w.Reset, 0)
		}
		result.Windows = append(result.Windows, entry)
	}
	return result, nil
}
