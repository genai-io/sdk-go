# sdk-go

用于大语言模型推理的 Go 客户端库，在 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses 和 Google Gemini 四套协议之上提供一套带类型的统一 API。

不管请求最终由哪家服务，你拿到的都是同一套 `Message`、`Response` 和流式事件类型。库里自带一份 27 家厂商、55 个模型的目录——端点、限额、价格和各端点的怪癖都以数据形式存在——并且**除非你主动选择，否则它不读任何凭证**。

> English: [README.md](README.md)

## 环境要求

Go 1.24 或更高。

## 安装

```sh
go get github.com/genai-io/sdk-go
```

## 用法

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)

func main() {
	client, err := auth.Client("openai/gpt-4.1")
	if err != nil {
		log.Fatal(err)
	}

	response, err := client.Complete(context.Background(),
		[]ai.Message{ai.UserMessage("解释一下 goroutine 泄漏。")},
		ai.WithSystem("回答尽量简洁。"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response.Text())
}
```

`auth.Client` 从各厂商自己文档里写明的那个环境变量读取凭证——这里是 `OPENAI_API_KEY`。想自己提供，见 [凭证](#凭证)。

那行 blank import 注册的是**一套线协议**。只 import 你用到的协议，因为每一个都会把对应厂商的 SDK 拖进来。

### 换一家厂商

改模型引用，上面的代码一个字都不用动。

```go
client, err := auth.Client("anthropic/claude-opus-5")
client, err := auth.Client("google/gemini-3.5-flash")
client, err := auth.Client("deepseek/deepseek-v4-pro")
client, err := auth.Client("ollama/llama4")
```

把 blank import 换成那个模型的协议；如果模型是运行时才决定的，用 `pkg/ai/driver/all`。

各家厂商的可运行示例在 [`examples/`](examples)。

## 流式

`Stream` 返回一个事件迭代器。**每一种内容——文字、思考、工具调用、图片——都走同一套 start/delta/end 生命周期**，所以一个循环处理全部：

```go
for event, err := range client.Stream(ctx, messages) {
	if err != nil {
		return err
	}
	switch event.Type {
	case ai.EventBlockStart:
		ui.Open(event.Index, event.Block.Type)

	case ai.EventBlockDelta:
		switch event.Block.Type {
		case ai.BlockText:
			ui.AppendText(event.Index, event.Block.Text)
		case ai.BlockThinking:
			ui.AppendThinking(event.Index, event.Block.Text)
		}

	case ai.EventBlockEnd:
		if event.Block.Type == ai.BlockToolCall {
			go runTool(*event.Block.ToolCall)
		}

	case ai.EventDone:
		response = event.Response
	}
}
```

`EventBlockDelta` 带的是片段；`EventBlockEnd` 带的是完整的块，一旦像工具调用这种原子值拼装完成就立刻到达。`EventDone` 带的是聚合好的 `Response`——跟 `Complete` 返回的是同一个值。

客户端会关闭它开启过的每一个块，失败时也一样。中途放弃迭代器会取消请求。

## 请求选项

对话就是普通的 `[]ai.Message`。其余一切都是 `Option`，而且**同一个 option 放在 `ai.New` 里是默认值、放在调用里是覆盖值**：

```go
client := ai.New(driver, model, ai.WithEffort(ai.EffortHigh)) // 默认值

response, err := client.Complete(ctx, messages,
	ai.WithTemperature(0),
	ai.WithMaxTokens(4096),
	ai.WithEffort(ai.EffortLow),                // 覆盖上面的默认值
	ai.WithStopSequences("\n\n"),
	ai.WithCacheRetention(ai.CacheLong))
```

**调用这个 option 本身就是"显式设置"的标记**，所以 `WithTemperature(0)` 是确定性采样，不写它才是继承。解析顺序是：模型默认值 → 客户端默认值 → 单次调用覆盖。

`Model` 和返回的模型列表跨边界时都是深拷贝快照，历史修复也绝不写你传进来的 messages——所以你事后再改自己那份数据，改不到一个正在飞行中的请求。

### 推理档位

推理是按**归一化的档位**来请求的。每个模型自带一份梯子，把这个档位映射到它的端点想要的东西——token 预算、级别字符串、或者一个开关标志——所以同一个请求到哪都能跑：

```go
ai.WithEffort(ai.EffortHigh)
```

模型如果没有你要的那一档，会被吸附到它有的最近一档。`Model.Efforts()` 告诉你某个模型实际提供哪些档。

## 消息与内容

一条消息装的是**按顺序排列的带类型的块**，而不是并行字段——因为那个顺序正是下一次请求必须回放的东西。

```go
messages := []ai.Message{
	ai.UserMessage("这张图里是什么？", image),
	ai.AssistantMessage("一只 gopher。"),
	ai.ToolResultsMessage(result),
}
```

构造器产生不了的顺序，就把消息直接写出来：

```go
ai.Message{Role: ai.RoleUser, Content: ai.Content{
	ai.TextBlock("比较 "),
	ai.ImageBlock(imageA),
	ai.TextBlock(" 和 "),
	ai.ImageBlock(imageB),
}}
```

不在乎顺序的时候，访问器可以只投影出某一类：

```go
text := response.Text()
thinking := response.Thinking()
calls := response.ToolCalls()

history = append(history, response.Message()) // 保留每一个块，保序
```

**用 `response.Message()`，不要用 `ai.AssistantMessage(response.Text())`**：前者会把模型的 thinking 和 reasoning 状态一并带到下一轮，而那正是让一个推理模型能够接着想、而不是每轮从头想的东西。

标签与载荷不匹配的块、出现在错误角色上的块，在 driver 被调用之前就会失败。

## 工具调用

工具参数是从 Go 类型推导出来的，所以**模型看到的 schema 和参数解码进去的那个结构体不可能对不上**：

```go
type SearchArgs struct {
	Query string `json:"query" jsonschema:"description=要找什么"`
	Limit int    `json:"limit,omitempty"`
}

search := ai.ToolFor[SearchArgs]("search", "搜索知识库")

response, err := client.Complete(ctx, messages, ai.WithTools(search))

for _, call := range response.ToolCalls() {
	if err := search.ValidateArgs(call.Input); err != nil {
		// 把它作为 tool result 交回去，模型能自己改对。
	}
	args, err := ai.UnmarshalArgs[SearchArgs](call)
	// ...
}
```

`ValidateArgs` 在你真正执行之前，拿工具自己的 schema 校验这次调用——**这样模型的错误会以"它能修正的形式"回到它那里**，而不是变成你的工具拿着一个缺失字段做出的任何事。

每次调用都必须在紧接着的那一轮里被应答：

```go
messages = append(messages, response.Message())
messages = append(messages, ai.ToolResultsMessage(ai.ToolResult{
	ToolCallID: call.ID,
	ToolName:   call.Name,
	Content:    output,
}))
```

## 结构化输出

```go
type Person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

person, err := ai.CompleteAs[Person](ctx, client, messages)
```

`CompleteAs` 从 `Person` 推导 schema、约束生成、再把答案解码回来——**这个类型只写一次**。想给模型一段关于这个形状的说明，或者想约束成一个 Go 类型表达不了的形状，再加 `ai.WithSchema`。

支持的每套协议都能原生约束生成。模型如果不支持，就在 prompt 里描述形状，然后用 `Response.Unmarshal` 解码——它能容忍被 markdown 围栏包住、或者前面带一段废话的答案。

## Token 计数

```go
count, err := client.CountTokens(ctx, messages)
if count.Exact {
	// 是厂商自己的分词器数出来的。
}

left, count, err := client.Headroom(ctx, messages)
```

Anthropic 和 Gemini 提供计数端点，结果是精确的；OpenAI 那一系协议没有，而**返回值会告诉你到底是哪种情况**，不会把估算值当成测量值端给你。上下文窗口未知的模型，报告的 headroom 是零，不是一个猜出来的数。

调用之后，`ai.IsOverflow(response, client.Model())` 能识别溢出——**包括那些不报错、直接静默截断的厂商**。

## 错误处理

失败是分好类的，所以"接下来怎么办"写在类型里，而不是藏在错误文本里：

```go
switch {
case ai.IsAuth(err):
	// 凭证错了或者没设。
case ai.IsContextExceeded(err):
	// 压缩对话后重试。
case ai.IsRetryable(err):
	time.Sleep(ai.RetryAfter(err))
case ai.IsUnsupported(err):
	// 这个模型做不到你要它做的事；在到达网络之前就被拦下了。
}
```

一轮失败会**同时**返回一个非 nil 的 error 和一个非 nil 的 `*Response`，里面装着先到的东西——已经流出来的文字、已经计费的 token。所以部分答案和它的成本不会跟着错误一起丢掉。

请求在发出之前就会被校验：往纯文本模型发图片、有工具调用却没有对应结果、给一个不支持约束输出的模型传 schema。每一种失败都会给出**一句点名了模型和原因的话**。

**这个库不会为了让请求跑通而改写它。** 一个没有 system 角色的模型、或者一个不能把输出约束成 schema 的模型，得到的是一个点名该模型的错误——把指令挪进 user 轮、或者用文字要求 JSON，是**关于你产品的决定，不是关于线格式的**。

## 重试与其他执行策略

**这个库不重试。** 重试、缓存、日志、成本计量都是 `Middleware`，因为只有你的应用知道一轮的预算、什么可以缓存、什么绝不能记进日志：

```go
client := ai.New(driver, model).Use(retry, costMeter)
```

有一条规则不该让你自己去踩出来：**重试只能重放一个在产出任何 delta 之前就失败的调用。** 一旦有输出到了你手里，答案就已经开始了，重发要么把已经显示的文字重复一遍，要么把它丢掉。

## 凭证

`pkg/ai` 不读任何环境变量、不读任何文件。这正是它能安全地待在一台握着多个租户密钥的服务器里的原因——直接给它一个 `Config`：

```go
client, err := ai.NewClient(ai.Config{
	Model:      model,
	APIKey:     key,
	BaseURL:    "https://gateway.internal/v1",
	HTTPClient: httpClient,
	Headers:    map[string]string{"X-Tenant": tenant},
})
```

`pkg/ai/auth` 是那个**明确选择去读环境**的包，命令行工具要的正是它。它会解析各厂商文档里写明的那些变量，并且为那些"认证的是人而不是服务"的厂商（GitHub Copilot、ChatGPT/Codex）跑浏览器登录流程。这个库唯一会写到磁盘上的东西，就是那份凭证存储——用户配置目录下的一个 `0600` 文件。

## Provider 与模型列表

三层，各自以"区别它们的东西"命名：

| 类型 | 是什么 |
| --- | --- |
| `catalog.Vendor` | 数据。不联网就能读的一行。 |
| `provider.Provider` | 那一行配好了凭证，带一份你可以刷新的模型列表。 |
| `ai.Client` | 它上面的一个模型。 |

```go
p, err := auth.Provider("ollama")
models := p.Models()       // 同步，不阻塞，不会失败
err = p.Refresh(ctx)       // 唯一会联网的调用
client, err := p.Client("llama4")
```

**读列表和取列表是两个分开的动词**，这样模型选择器可以立刻渲染，而一台挂掉的主机卡不住它。

## 支持的协议

| 协议 | 包 | 目录里的厂商数 |
| --- | --- | --- |
| OpenAI Chat Completions | `pkg/ai/driver/openai/chat` | 18 |
| OpenAI Responses | `pkg/ai/driver/openai/responses` | 3 |
| Anthropic Messages | `pkg/ai/driver/anthropic` | 4 |
| Anthropic on Vertex AI | `pkg/ai/driver/anthropic/vertex` | 1 |
| Google Gemini | `pkg/ai/driver/google` | 1 |

**一个厂商是目录里的一行，不是一个包。** 绝大多数厂商提供的是一个说别人协议的端点，所以新增一家是 `pkg/ai/catalog` 里的一次数据改动——不是又一份 HTTP 实现。

## 实现一套协议

driver 接口只有两个方法：

```go
type Driver interface {
	Name() string
	Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error]
}
```

调用方还需要的其他一切——把 delta 聚合成 `Response`、应用默认值、修复历史、校验、重试——都归 `Client`，而且**对每套协议只写一次**。模型列举和 token 计数是可选接口，靠类型断言发现；driver 不实现它们从来不是错误。

在 `init` 里注册，这样一行 blank import 就足以让这套协议可达。**只有你的协议才有的设置**，放进一个带类型的 `ProtocolOptions` 值，而不是往 `Request` 上加字段：

```go
response, err := client.Complete(ctx, messages,
	ai.WithProtocolOptions(anthropic.Options{ThinkingDisplay: "omitted"}))
```

你的类型实现 `ai.ProtocolOptions`——一个一行的标记方法——所以这个字段不是裸的 `any`，**一个本来就不该放进去的值是编译期错误**。而放进去的是**另一个 driver 的**类型、或者送给一个根本没定义这类设置的协议时，那是一次无效请求，在那个 driver 读取它的那一刻被抓住。它绝不会被静默忽略。

构造期的设置走 `ai.ProtocolConfig`，规则完全一样。

## 测试

因为 `Driver` 只有两个方法，拿一个模型来测你自己的代码，就是写一个你这个用例需要的桩：

```go
type scripted struct{ deltas []ai.Delta }

func (scripted) Name() string { return "scripted" }

func (s scripted) Stream(context.Context, *ai.Request) iter.Seq2[ai.Delta, error] {
	return func(yield func(ai.Delta, error) bool) {
		for _, d := range s.deltas {
			if !yield(d, nil) {
				return
			}
		}
	}
}
```

这个库自己的测试是 `test/` 下的一个黑盒包。它像应用一样 import 这个 SDK，并且**只断言两件事：到达端点的字节，和回来的那个值**。每个端点都是一个桩 HTTP 服务器，所以整套测试不需要网络、不需要凭证。

```sh
go test ./test/
```

## 设计说明

理由待在它管的那段代码旁边：

```sh
go doc github.com/genai-io/sdk-go/pkg/ai         # 请求、结果，以及每个文件各自负责什么
go doc github.com/genai-io/sdk-go/pkg/ai/driver  # 为什么 driver 是包、而厂商是表里的一行
go doc github.com/genai-io/sdk-go/pkg/ai/auth    # 为什么凭证解析是单独的一次 import
```

两条规则解释了大部分布局。**包是代码的单位，所以一个线格式一个包，一个厂商零个包。** 以及**一个文件以它负责的主题命名，并且负责这个主题的全部**——如果两个文件各拿着一个想法的一部分，那么其中一个是错的。

[`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md) 讲的是没有任何单个文件负责的那部分：各部分怎么拼在一起、依赖朝哪个方向走、一个请求从你的调用到线上的字节之间经过了什么。

## 版本

API 尚在早期开发中，可能会变。在有 tag 的发布出现之前，请 pin 到某个 commit。

## 许可

[Apache 2.0](LICENSE)
