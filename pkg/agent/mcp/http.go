package mcp

import "net/http"

// withHeaders is a client that adds the same headers to every request — a
// gateway token, a tenant tag. The MCP transports take an *http.Client and
// nothing else, so a header that has to be on every request has to ride on
// one.
func withHeaders(base *http.Client, headers map[string]string) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	c := *base
	c.Transport = headerTransport{inner: c.Transport, headers: headers}
	return &c
}

type headerTransport struct {
	inner   http.RoundTripper
	headers map[string]string
}

func (t headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	inner := t.inner
	if inner == nil {
		inner = http.DefaultTransport
	}
	// Clone: a RoundTripper is not allowed to modify the request it is given,
	// and a retry would otherwise accumulate headers on the same one.
	req = req.Clone(req.Context())
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	return inner.RoundTrip(req)
}
