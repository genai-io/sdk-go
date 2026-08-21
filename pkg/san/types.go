package san

import "github.com/genai-io/sdk-go/pkg/llm"

// EventType identifies an agent lifecycle event.
type EventType string

// Agent lifecycle events — emitted to the Outbox.
const (
	OnStart   EventType = "AgentStart" // agent begins
	OnStop    EventType = "AgentStop"  // agent ends
	OnChunk   EventType = "Chunk"      // streaming delta (llm.Event in Data)
	OnTurn    EventType = "Turn"       // think+act cycle completed (Result in Data)
	OnMessage EventType = "Message"    // message received on inbox (llm.Message in Data)

	PreInfer  EventType = "PreInfer"  // before the model call
	PostInfer EventType = "PostInfer" // after the model responds
	PreTool   EventType = "PreTool"   // before tool execution (llm.ToolCall in Data)
	PostTool  EventType = "PostTool"  // after tool execution (llm.ToolResult in Data)
)

// Event is a lifecycle event emitted by the agent during Run.
type Event struct {
	Type   EventType // which event
	Source string    // who triggered it (agent ID, tool name)
	Data   any       // payload — the type depends on Type
}

// Convenience accessors for Event.Data.
func (e Event) Chunk() (llm.Event, bool)           { c, ok := e.Data.(llm.Event); return c, ok }
func (e Event) Result() (Result, bool)             { r, ok := e.Data.(Result); return r, ok }
func (e Event) ToolCall() (llm.ToolCall, bool)     { tc, ok := e.Data.(llm.ToolCall); return tc, ok }
func (e Event) ToolResult() (llm.ToolResult, bool) { tr, ok := e.Data.(llm.ToolResult); return tr, ok }
func (e Event) Message() (llm.Message, bool)       { m, ok := e.Data.(llm.Message); return m, ok }
func (e Event) Error() (error, bool)               { err, ok := e.Data.(error); return err, ok }

// Result is the outcome of one completed turn.
type Result struct {
	Content    string         // final text output of this turn
	Messages   []llm.Message  // full conversation history
	Steps      int            // inference steps in this turn
	ToolUses   int            // tool calls in this turn
	Usage      llm.Usage      // tokens consumed across every step of the turn
	StopReason llm.StopReason // why the loop stopped
}
