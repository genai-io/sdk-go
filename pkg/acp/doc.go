// Package acp provides experimental ACP (Agent Client Protocol) endpoints,
// connecting editors and other clients to agents. It provides bounded UTF-8,
// newline-delimited JSON-RPC 2.0 transport and a typed receiving-side core.
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
// [Conn.Initialize], [Conn.CreateSession], [Conn.Prompt], [Conn.CancelPrompt],
// [Conn.SendSessionUpdate] and [Conn.RequestPermission] validate message shapes
// against the pinned schema. Raw Conn dispatch uses [Handlers].
// These convenience methods do not track initialization, serialize prompts,
// negotiate capabilities, manage sessions or apply authorization policy.
//
// # Typed receiving endpoint
//
// [NewAgent] supplies initialization and session state over the same transport.
// [AgentHandlers] implements Initialize, CreateSession and Prompt. Initialization
// accepts protocol version 1; successful initialization is immutable. Independent
// capability snapshots are available through [Agent.Initialization]; enforcing
// content capabilities, MCP configuration and authorization remains application
// work. Unsupported requests receive method-not-found; unknown notifications are
// ignored. Malformed session/cancel notifications close the connection.
//
// Session slots are bounded by [AgentOptions.MaxSessions], including creations
// in flight. The application allocates unique nonempty session IDs and owns all
// runtime/storage resources, including cleanup after failed creation or shutdown.
// There is no automatic eviction, persistence, resume or session-close operation.
//
// Admission occurs in wire order on the reader without invoking application
// callbacks: overlapping prompts for a session are rejected, not queued. Cancel
// binds to the turn active at admission and never to a later successor. Idle or
// unknown-session cancellation is a no-op. Callbacks still run concurrently
// across sessions. Initialization/session readiness and prompt-slot retirement
// are committed before their responses become observable, alongside request-ID
// retirement. Clients should await initialization/creation responses before
// sending dependent work.
//
// Session-limit and active-prompt rejections use implementation-defined server
// error code -32001, not ACP's -32000 (authentication required). Prompts for an
// unknown session use -32002 (resource not found).
//
// Each Prompt receives a [Turn]. Its SendSessionUpdate and RequestPermission
// methods bind the session ID and reject work after the callback returns. The
// response waits for callback unwind and all admitted outbound operations, so
// final updates precede completion. Cancellation requests cooperative unwind;
// it does not manufacture a response while application execution is still live.
// A cancelled callback returning nil or context.Canceled produces stopReason
// cancelled; other callback errors retain the usual Conn error handling.
// Applications must join their own goroutines before returning from callbacks.
//
// Turn writes deliberately use connection lifetime: final updates can be sent
// after cancellation, and cancelling a permission wait cannot tear a frame.
// Permission cancellation takes effect after its request write finishes; a
// stalled peer requires Close (or cancellation of NewAgent's parent context).
// Callback return cancels outstanding permission waits and joins admitted work.
// This does not send remote permission cancellation or apply permission policy.
// Retained Turn values cannot publish into a successor. Raw Conn methods retain
// their original active-write cancellation semantics.
//
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
