# Examples

Two kinds of folder. Most show one vendor, and what is worth seeing is the
vendor-specific thing each demonstrates — the request code barely changes
between them, which is the point. The rest show a capability that works the
same everywhere.

## Capabilities

| | What it shows |
| --- | --- |
| [`quickstart/`](quickstart) | The smallest thing that works: one prompt, streamed, against any model the catalog can name |
| [`tools/`](tools) | A tool-calling loop, and the four rules that make one work: the schema comes from the Go type, arguments are checked before anything runs, every call is answered in the turn that follows, and the model's own state is carried forward |
| [`structured/`](structured) | Asking for an answer shaped like a Go type and getting one back decoded, in a single call — with the struct tags that tell the model what each field means and what it may contain |

Run `go run ./examples/structured -schema` to see what a Go type actually
becomes on the wire.

## Agents

`pkg/agent` runs the loop around a model call. The first three each isolate one
idea; the last is a working assistant with all of them in it.

| | What it shows |
| --- | --- |
| [`agent-chat/`](agent-chat) | One exchange per line you type: `Run` advances the conversation once and returns, so repeating it is your `for` loop rather than a mode the agent is in. Ctrl-C ends the exchange in flight and hands the prompt back |
| [`agent-progress/`](agent-progress) | A tool that takes a while and shows its work — `agent.Report` from inside the tool arriving as `ToolUpdate` — and the split between `Result.Content`, which the model reads, and `Result.Details`, which only your interface does |
| [`agent-session/`](agent-session) | Record and resume: run it twice and the second run continues the first. The agent is handed a plain `[]ai.Message` and never learns where it came from — and `-keep` shows the one case that needs saying out loud, compacting a history the session would otherwise hand straight back |
| [`agent/`](agent) | All of it at once: a system prompt built from the environment, a `PreTool` gate that refuses anything that writes, and a session on disk that the next run resumes from |

```sh
go run ./examples/agent-chat
go run ./examples/agent-progress "how many Go files are here, and how many Markdown ones?"
go run ./examples/agent-session "what is a goroutine?"
go run ./examples/agent-session "and how do they leak?"   # remembers the first
go run ./examples/agent "how many Go files are in this directory?"
```

## Vendors

| | Vendor | Protocol | What it shows |
| --- | --- | --- | --- |
| [`anthropic/`](anthropic) | Anthropic | Messages | Thinking blocks you can read, and aiming the prompt cache — with what it cost |
| [`openai/`](openai) | OpenAI | Responses | Carrying a reasoning model's own state across turns, so it resumes instead of starting over |
| [`qwen/`](qwen) | Alibaba (Qwen) | Chat Completions | A vendor that is pure data — and how to reach an OpenAI-compatible endpoint the catalog has never heard of |
| [`deepseek/`](deepseek) | DeepSeek | Chat Completions | A model that reasons unless told not to, and what that costs if you do not know |
| [`gemini/`](gemini) | Google | Gemini | Sending an image, and a driver with no vendor SDK behind it |
| [`ollama/`](ollama) | local | Chat Completions | No credential at all, and asking a server what it actually serves |

Every one needs a credential in the environment, except `ollama`:

```sh
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export DASHSCOPE_API_KEY=...   # Qwen, via DashScope
export DEEPSEEK_API_KEY=...
export GEMINI_API_KEY=...

go run ./examples/qwen "用两句话解释 goroutine 泄漏"
```

To see which ones you can run right now:

```sh
go run ./examples/quickstart -list
```
