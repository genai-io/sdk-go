# Examples

Each folder shows something the others do not. The request code barely changes
between them — that is the point — so what each one is actually for is the
vendor-specific thing it demonstrates.

| | Vendor | Protocol | What it shows |
| --- | --- | --- | --- |
| [`quickstart/`](quickstart) | any | any | The smallest thing that works: one prompt, streamed, against any model the catalog can name — and the only folder here that is not about one vendor |
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
