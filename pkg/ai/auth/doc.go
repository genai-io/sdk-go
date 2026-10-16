// Package auth turns a model reference into a ready-to-use client by finding
// the credential for it.
//
//	client, err := auth.Client("anthropic/claude-opus-4-6")
//
// It is deliberately a separate import from package ai, which never reads the
// environment or the filesystem. A server handling several tenants must not
// inherit a process-wide key by accident; a command-line tool wants exactly
// that, and opts in here.
//
// # Two kinds of vendor
//
// Key-based vendors are resolved from the environment. The variables consulted
// are the ones each vendor documents, recorded in the catalog as KeyEnv,
// BaseURLEnv and VertexEnv; a caller with a secret store of its own reads them
// through catalog.Vendor's KeyFromEnv, BaseURLFromEnv and VertexFromEnv
// instead. Nothing is stored, and nothing is written back.
//
// Interactive vendors authenticate a person rather than a service: there is no
// key to paste, only a subscription and a browser. Login runs the grant and
// keeps the result in a Store so the next run does not sign in again — see
// login.go. The default Store is a 0600 file under the user's config
// directory, which is what a CLI wants; a server passes its own Store to
// Login, Logout and HTTPClient. That file is the only thing this SDK writes to
// disk.
//
// # Where things live
//
//	auth.go        resolving a reference into an ai.Config or a client
//	endpoint.go    the same, for a whole vendor's model listing
//	login.go       interactive sign-in: the flows and the Login verb
//	grant.go       the OAuth grants those flows run, and the Prompt they show
//	copilot.go     GitHub Copilot's grant
//	codex.go       the ChatGPT/Codex grant
//	console.go     the default terminal prompt for a sign-in
//	credential.go  what a sign-in produces, and where it is kept
//	filestore.go   the default Store: one 0600 file under the config dir
//	transport.go   presenting a stored credential on every request
//	errors.go      what a caller gets when nobody has signed in
package auth
