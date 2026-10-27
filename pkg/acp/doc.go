// Package acp provides the bounded NDJSON connection used by ACP peers.
// This experimental kernel does not yet provide typed ACP methods or an agent.
// Session storage, capability negotiation and authorization remain outside Conn.
//
// New takes ownership of its streams. Their Close methods must interrupt
// concurrent reads and writes. Handlers must honor their context; Wait joins
// them rather than pretending cancellation stopped application work.
// Malformed/oversized frames and queue overflow close the connection; EOF aborts
// pending delivery rather than promising a graceful notification drain.
package acp
