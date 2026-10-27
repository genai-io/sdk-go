// Package acp provides the experimental connection kernel for ACP (Agent Client
// Protocol), which connects editors and other clients to agents. It provides
// bounded UTF-8, newline-delimited JSON-RPC 2.0 transport, not a complete agent.
//
// # Connection boundary
//
// Applications supply closable streams and [Handlers] to [New]. The connection
// handles framing, request/response correlation, ordered notifications and
// bounded work; handlers implement the application's protocol semantics.
//
//	ACP peer <--> Conn <--> application handlers <--> application runtime
//
// The protocol subpackage contains generated stable ACP v1 wire types.
// [Conn.Initialize], [Conn.NewSession], [Conn.Prompt], [Conn.Cancel],
// [Conn.SessionUpdate] and [Conn.RequestPermission] validate message shapes
// against the pinned schema. Incoming dispatch still uses [Handlers].
// These convenience methods do not track initialization, serialize prompts,
// negotiate capabilities, manage sessions or apply authorization policy.
// New starts I/O workers; it does not launch a process, listen on a port or
// connect an agent runtime. Applications may use child-process stdio streams.
//
// # Test conversation
//
// TestACPWireConversation in integration/acp_test.go uses real OS pipes and
// fixed handlers to demonstrate the following exchange:
//
//	Mock client                         Conn + test handlers
//	    | -- initialize --------------------> |
//	    | <---------------- protocol version |
//	    | -- session/new -------------------> |
//	    | <---------------- test session ID  |
//	    | -- session/prompt ----------------> |
//	    | <------ session/request_permission |  handler calls Conn.Call
//	    | -- allow once --------------------> |
//	    | <----------------- session/update  |  handler calls Conn.Notify
//	    | <------------- stopReason=end_turn |
//
// This fixture runs no model or tools and does not establish ACP compatibility
// with an editor. The method-specific responses come from the test handlers.
//
// # Ownership and shutdown
//
// New takes ownership of its streams on success. Their Close methods must
// interrupt concurrent I/O. Handlers must honor their context; [Conn.Wait]
// joins them rather than forcing application work to stop. [Conn.Close] and
// [Conn.Done] signal shutdown, not completed cleanup; Wait joins cleanup.
//
// Cancelling a Call context does not send session/cancel or stop remote work.
// Cancelling an active frame write closes the connection to avoid reusing a
// partially written frame. Malformed/oversized input and exhausted inbound or
// write capacity also close it; EOF aborts pending delivery, without draining
// all previously received notifications.
package acp
