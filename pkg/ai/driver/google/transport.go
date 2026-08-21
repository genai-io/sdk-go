package google

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
)

// The HTTP plumbing. Gemini has no Go SDK this driver depends on, so the
// request/response mechanics — URL construction, headers, and reading a
// server-sent-event stream — live here rather than in a vendor library.

// methodURL builds the URL for a model method, e.g. ":countTokens".
func (d *Driver) methodURL(method, query string) string {
	u := fmt.Sprintf("%s/%s/models/%s:%s", d.baseURL, apiVersion, d.model.ID, method)
	if query != "" {
		u += "?" + query
	}
	return u
}

func (d *Driver) post(ctx context.Context, url string, body any) (*http.Response, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return d.do(req)
}

func (d *Driver) get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	return d.do(req)
}

func (d *Driver) do(req *http.Request) (*http.Response, error) {
	// The key travels in a header rather than the query string the REST
	// examples use: a URL ends up in proxy logs and error reports, and a
	// credential should not go with it.
	if d.apiKey != "" {
		req.Header.Set("x-goog-api-key", d.apiKey)
	}
	for k, v := range d.headers {
		req.Header.Set(k, v)
	}
	res, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= 400 {
		defer res.Body.Close()
		return nil, readAPIError(res)
	}
	return res, nil
}

// sseEvents yields the payload of each "data:" event until the stream ends.
func sseEvents(r io.Reader) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		scanner := bufio.NewScanner(r)
		// A single event can carry a whole turn's worth of tool arguments; the
		// default 64KiB limit would cut one in half.
		scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		for scanner.Scan() {
			// Bytes, not Text: this runs once per streamed delta, and the
			// consumer only unmarshals the payload, which does not retain it.
			// Going through a string would copy every line twice.
			line := bytes.TrimRight(scanner.Bytes(), "\r")
			payload, ok := bytes.CutPrefix(line, []byte("data:"))
			if !ok {
				continue
			}
			payload = bytes.TrimSpace(payload)
			if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			if !yield(payload, nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
		}
	}
}
