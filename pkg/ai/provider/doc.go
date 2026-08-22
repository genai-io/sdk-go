// Package provider reconciles what a host actually serves with what the
// catalog says about it.
//
// An ai.Client talks to one model. This is the layer above: one configured
// host — its base URL, its credential, its protocol — and the list of models
// on it, which is what a model picker needs and what a Client cannot give you.
//
// That list has two sources and neither is enough alone. The catalog knows
// pricing, context windows, reasoning ladders and protocol quirks; the host
// knows which models exist today and little else. Provider merges them, with
// the host authoritative about what exists and about any figure it reported
// and the catalog filling the rest — because a model stripped of its quirks
// stops working, and a model the catalog has never heard of still has to open.
//
//	p := provider.New(provider.Config{ID: "acme", API: ai.APIOpenAIChat, APIKey: key})
//	models := p.Models()        // what is known now: never blocks, never fails
//	err := p.Refresh(ctx)       // the only call that reaches the network
//	client, err := p.Client("acme-pro")
//
// Reading the list and fetching it are separate verbs on purpose. A picker has
// to render immediately and a host that is down must not hang it, so Models
// answers from what it already knows and Refresh is the one thing that goes
// out to ask.
//
// This package sits above package ai, which does not know it exists.
//
// # Where things live
//
//	provider.go  one host: its models, its config, and merging a live listing over them
//	set.go       several of them, and the fan-out refresh across them
package provider
