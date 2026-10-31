package auth

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestParseCodexUsage(t *testing.T) {
	tests := []struct {
		name, body string
		count      int
		bad        bool
	}{
		{"windows", `{"rate_limit":{"primary_window":{"used_percent":25,"limit_window_seconds":604800,"reset_at":2000000000},"secondary_window":{"used_percent":0,"limit_window_seconds":18000}}}`, 2, false},
		{"missing", `{}`, 0, false},
		{"null", `{"rate_limit":null}`, 0, false},
		{"no windows", `{"rate_limit":{}}`, 0, false},
		{"incomplete", `{"rate_limit":{"primary_window":{"limit_window_seconds":18000}}}`, 0, false},
		{"invalid percentage", `{"rate_limit":{"primary_window":{"used_percent":-1,"limit_window_seconds":18000}}}`, 0, false},
		{"invalid duration", `{"rate_limit":{"primary_window":{"used_percent":12,"limit_window_seconds":9223372036854775807}}}`, 0, false},
		{"secondary only", `{"rate_limit":{"primary_window":null,"secondary_window":{"used_percent":100,"limit_window_seconds":3600}}}`, 1, false},
		{"malformed", `{`, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseCodexUsage([]byte(tt.body))
			if (err != nil) != tt.bad || len(got.Windows) != tt.count {
				t.Fatalf("got %+v, %v", got, err)
			}
			if tt.name == "windows" && (got.Windows[0].Duration != 7*24*time.Hour || got.Windows[0].UsedPercent != 25 || got.Windows[0].ResetsAt.Unix() != 2000000000 || got.Windows[1].UsedPercent != 0) {
				t.Fatalf("wrong windows: %+v", got)
			}
		})
	}
}

type usageTransport func(*http.Request) (*http.Response, error)

func (f usageTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestFetchCodexUsageBounded(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	client := &http.Client{Transport: usageTransport(func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	})}
	if _, err := FetchCodexUsage(ctx, client); err == nil {
		t.Fatal("canceled request succeeded")
	}
	for _, body := range []string{"not JSON", strings.Repeat(" ", 1024*1024+1)} {
		client.Transport = usageTransport(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
		})
		if _, err := FetchCodexUsage(context.Background(), client); err == nil {
			t.Fatal("invalid response accepted")
		}
	}
}

func TestFetchCodexUsage(t *testing.T) {
	for _, status := range []int{200, 401, 403, 429, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			client := &http.Client{Transport: usageTransport(func(r *http.Request) (*http.Response, error) {
				if r.Method != "GET" || r.URL.String() != "https://chatgpt.com/backend-api/wham/usage" {
					t.Fatalf("wrong request: %s %s", r.Method, r.URL)
				}
				return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(`{"private_account_detail":"secret","rate_limit":{}}`)), Header: make(http.Header)}, nil
			})}
			_, err := FetchCodexUsage(context.Background(), client)
			if (err != nil) != (status != 200) {
				t.Fatalf("error: %v", err)
			}
			if err != nil && strings.Contains(err.Error(), "secret") {
				t.Fatal("response leaked")
			}
		})
	}
	if _, err := FetchCodexUsage(context.Background(), nil); err == nil {
		t.Fatal("nil client accepted")
	}
}
