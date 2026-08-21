// Package endpoint is a configured endpoint and the models it serves.
//
// It sits above ai.Client rather than inside it: nothing in the core knows a
// Endpoint exists. A Client talks to one model; a Endpoint is the layer a model
// picker needs — the list of models an endpoint offers, refreshed against the
// endpoint itself, with one credential and one base URL behind all of them.
//
//	e := endpoint.New(endpoint.Config{ID: "acme", API: ai.APIOpenAIChat, APIKey: key})
//	models := e.Models()          // synchronous, never fails
//	err := e.Refresh(ctx)         // explicit network call
//	client, err := e.Open("acme-pro")
//
// Reading the list and fetching it are separate verbs on purpose. A picker has
// to render immediately, and a dead endpoint must not hang it — so Models
// returns what is known now, and Refresh is the only thing that reaches the
// network.
//
// # Where things live
//
//	provider.go  one endpoint: its models, its config, and merging a live listing
//	set.go       a set of them, and the fan-out refresh over it
//	clone.go     the snapshots taken wherever a model crosses in or out
package endpoint
