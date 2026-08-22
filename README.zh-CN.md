# sdk-go

[![Go Reference](https://pkg.go.dev/badge/github.com/genai-io/sdk-go.svg)](https://pkg.go.dev/github.com/genai-io/sdk-go)
[![CI](https://github.com/genai-io/sdk-go/actions/workflows/ci.yml/badge.svg)](https://github.com/genai-io/sdk-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/github/go-mod/go-version/genai-io/sdk-go)](go.mod)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

一个用于大模型推理的 Go 客户端库，在 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses 和 Google Gemini 四种协议之上提供**同一套带类型的 API**。

- **一套 API，五种协议。**不管请求最终由哪家服务，拿到的都是同样的 `Message`、`Response` 和流式事件。
- **流式。**文本、thinking、工具调用、图片共用同一套 start/delta/end 生命周期，**一个循环处理所有内容类型**。
- **工具调用。**一个工具就是一个 struct 加一个函数。schema 从 struct 推导，参数在你的代码运行之前先校验。
- **结构化输出。**`CompleteAs[T]` 推导 schema、约束生成、解码答案，**类型只写一次**。
- **带类型的错误。**认证、限流、超上下文、不支持是**四个不同的答案**，不是四个要去匹配的子串。
- **模型目录。**27 家厂商、55 个模型——端点、限额、价格和各端点的怪癖都是数据，**不联网就能读**。
- **不会顺手读凭证。**`pkg/ai` 不读任何环境变量、不读任何文件。`pkg/ai/auth` 是那个**选择性开启**的、确实会读的入口。

[安装](#安装) ·
[快速开始](#快速开始) ·
[流式](#流式) ·
[工具调用](#工具调用) ·
[结构化输出](#结构化输出) ·
[请求选项](#请求选项) ·
[消息与内容](#消息与内容) ·
[错误](#错误与执行策略) ·
[凭证](#凭证) ·
[协议](#支持的协议) ·
[English](README.md)

## 安装

```sh
go get github.com/genai-io/sdk-go
```

需要 Go 1.24 或更高版本。

## 快速开始

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
		[]ai.Message{ai.UserMessage("Explain goroutine leaks.")},
		ai.WithSystem("You are concise."))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response.Text())
}
```

`auth.Client` 会从厂商文档规定的那个环境变量里读凭证——这里是 `OPENAI_API_KEY`。那个空导入注册的是**一种线协议**；只导入你要用的，或者在模型要到运行时才确定时用 `pkg/ai/driver/all`。

这是**一条链的最短端**：`auth.Client` 就是 `auth.Config` 加 `ai.New`，后者又是 `ai.NewDriver` 加 `ai.NewClientWithDriver`。需要的话就早点停——绝不能顺手读凭证的服务端停在 `ai.New`，要套 middleware 就停在 `ai.NewDriver`。见[构造客户端](docs/clients.zh-CN.md)。

### 换一家服务

**只改模型引用，程序里其他什么都不用动。**

```go
ref := "anthropic/claude-opus-5" // 或 google/gemini-3.5-flash、
                                 //    deepseek/deepseek-v4-pro、ollama/llama4
client, err := auth.Client(ref)
```

引用的格式是 `厂商/模型`。两半都来自目录，所以 `catalog.Models()` **不联网**就能列出全部可解析的模型。

可直接运行的例子在 [`examples/`](examples)——每家厂商一个，外加 [`tools/`](examples/tools) 和 [`structured/`](examples/structured)。

## 流式

`Stream` 返回一个事件迭代器。**每一种内容**——文本、thinking、工具调用、图片——都走同一套 start/delta/end 生命周期，所以一个循环就能全部处理。

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

`EventBlockDelta` 带的是**片段**；`EventBlockEnd` 带的是**完整的块**，而且像工具调用这种原子值一凑齐就会到达。`EventDone` 带的是聚合好的 `Response`——和 `Complete` 返回的是同一个值。客户端会关闭它开启的每一个块，**出错时也一样**；中途放弃迭代器会取消请求。

## 工具调用

一个工具就是**一个名字、一句话、一个函数**。struct 里放的正好是模型可以发来的东西。

```go
type SearchArgs struct {
	Query string `json:"query" description:"要找什么，用大白话"`
	Limit int    `json:"limit,omitempty" description:"最多返回几段" maximum:"10"`
}

search := ai.ToolFunc("search", "搜索文档，返回匹配的段落。",
	func(ctx context.Context, a SearchArgs) (string, error) {
		return docs.Search(ctx, a.Query, a.Limit) // 依赖走闭包
	})

response, history, err := client.Run(ctx,
	[]ai.Message{ai.UserMessage(question)}, []ai.Tool{search, fetch})
```

模型不会执行你的工具：它请求你执行、你回答、它继续。`Run` 就是这个循环，`history` 是整段对话，所以追问直接从它接着走。

**`SearchArgs` 只写了一次**，所以发给模型的 schema 和参数解码进去的那个 struct 不可能各说各话：

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "要找什么，用大白话"},
    "limit": {"type": ["integer", "null"], "description": "最多返回几段", "maximum": 10}
  },
  "required": ["query", "limit"],
  "additionalProperties": false
}
```

参数在你的函数运行之前先按这份 schema 校验过，于是**模型的错误会变成它能自己改正的东西**，而不是变成"你的工具拿着一个缺字段的输入去干活"。工具名不存在、参数不对、工具自己失败——都作为 `IsError` 结果回给模型，而不是终止整段对话。

约束模型调哪个，是**一个值、四种状态**，正好是这里每个协议都能表达的那四种：

```go
ai.WithToolChoice(ai.ToolChoiceNone)            // 这一轮不许调工具
ai.WithToolChoice(ai.ToolChoiceRequired)        // 必须调一个，调哪个模型定
ai.WithToolChoice(ai.ToolChoiceNamed("search")) // 必须调这个
```

`ToolFunc` 返回的就是一个普通 `ai.Tool`——上线的三个字段加一个函数——所以手写 schema 是一次赋值，到运行时才知道形状的工具就是直接写出那个值。轮次本身是你的业务时（要流式输出、要按条件停、要每轮记账），用 `Complete` 加 `RunTools` 自己写这个循环。

## 结构化输出

```go
type Person struct {
	Name string `json:"name" description:"完整法定姓名，姓在后"`
	Age  int    `json:"age" description:"周岁" minimum:"0"`
}

person, err := ai.CompleteAs[Person](ctx, client, messages)
```

`CompleteAs` 从 `Person` 推导 schema、约束生成、解码答案，**类型只写一次**。这里支持的每一个协议都是原生约束生成。

**标签的 key 就是它要设的那个 JSON Schema 关键字**——`description`、`enum`、`minimum`、`maximum` 等共 11 个——而且里面每一个字都是 prompt 文本。schema 是按**"能被接受"**推导的，不只是"合法"：所有字段进 `required`、所有对象封闭、可选表示成 `["T","null"]`，因为各家的 strict 模式就是这么要求的。词表和规则见 [`pkg/ai/jsonschema`](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/ai/jsonschema)。

## 请求选项

对话就是一个普通的 `[]ai.Message`。其余一切都是 `Option`，而且**同一个 option 在构造时是默认值、在调用处是覆盖值**：

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh)) // 默认值

response, err := client.Complete(ctx, messages,
	ai.WithTemperature(0),
	ai.WithMaxTokens(4096),
	ai.WithEffort(ai.EffortLow), // 覆盖上面的默认值
	ai.WithCacheRetention(ai.CacheLong))
```

**传了 option 本身就是"显式"的标记**，所以 `WithTemperature(0)` 是确定性采样，不传才是继承。解析顺序是：模型默认 → 客户端默认 → 调用处覆盖。

推理强度是按**归一化的档位**请求的。每个模型自带一张梯子，把这个档位映射到它的端点想要的东西——token 预算、级别字符串、开关标志——所以同一个请求到哪都能跑；模型没有你要的那一档时会**贴到最近的一档**。

## 消息与内容

一条消息装的是**有序的、带类型的块序列**，而不是一堆平行字段——因为**下一次请求要重放的正是这个顺序**。

```go
messages := []ai.Message{
	ai.UserMessage("What is in this image?", image),
	ai.AssistantMessage("A gopher."),
	ai.ToolResultsMessage(result),
}

text := response.Text()
history = append(history, response.Message()) // 保留每一个块，保序
```

**用 `response.Message()`，不要用 `ai.AssistantMessage(response.Text())`**：前者会把模型的 thinking 和 reasoning 状态一并带到下一轮，而那正是让推理模型能接着想、而不是每轮从头想的东西。构造函数产不出的顺序，就把 `ai.Message` 连同它的 `Content` 直接写出来。

## 错误与执行策略

失败是**分类过的**，所以"现在该怎么办"的答案在**类型**里，而不在错误文本里：

```go
switch {
case ai.IsAuth(err):            // 凭证不对或缺失
case ai.IsContextExceeded(err): // 压缩对话后重试
case ai.IsRetryable(err):       time.Sleep(ai.RetryAfter(err))
case ai.IsUnsupported(err):     // 这个模型做不了要求它做的事
}
```

失败的一轮会**同时**返回非 nil 的 error 和非 nil 的 `*Response`，里面装着先到的那部分，所以**部分答案和它的花费不会跟着错误一起丢掉**。请求在发出之前就会校验——图片发给纯文本模型、工具调用没有对应结果——而且**这个库不会替你改写请求去让它能跑**：模型没有 system 角色就报错并指名道姓，因为把那些指令挪进 user 轮是**关于你的产品的决定**，不是关于线协议的。

执行策略是装饰 driver 的 `Middleware`，因为只有你的应用知道一轮的预算、什么能缓存、什么绝不能记日志：

```go
client := ai.NewClientWithDriver(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), model)
```

`Retry` 是这里唯一自带的策略，而且**每个 driver 都关掉了它厂商 SDK 自己的重试**，所以不加它就一次重试都没有。缓存、日志、成本统计都归你。

## 凭证

**`pkg/ai` 永远不读环境变量、不读文件。**正是这一点让它在一台握着多个租户密钥的服务器上是安全的：

```go
client, err := ai.NewClient(ai.Config{
	Model: model, APIKey: key, BaseURL: "https://gateway.internal/v1",
	HTTPClient: httpClient, Headers: map[string]string{"X-Tenant": tenant},
})
```

`pkg/ai/auth` 是那个**选择性开启**的、确实会读环境的入口，命令行工具要的就是它。`pkg/ai/provider` 是目录行和客户端之间的那一层：一个配好的 host 加上它上面的模型列表，其中**"读列表"和"拉列表"是两个动词**——所以模型选择器能立刻渲染出来，而一个挂掉的端点卡不住它。

## 支持的协议

| 协议 | 包 | 目录中的厂商数 |
| --- | --- | --- |
| OpenAI Chat Completions | `pkg/ai/driver/openai/chat` | 18 |
| OpenAI Responses | `pkg/ai/driver/openai/responses` | 3 |
| Anthropic Messages | `pkg/ai/driver/anthropic` | 4 |
| Anthropic on Vertex AI | `pkg/ai/driver/anthropic/vertex` | 1 |
| Google Gemini | `pkg/ai/driver/google` | 1 |

**厂商是目录里的一行，不是一个包。**大多数厂商提供的端点说的是别人的协议，所以加一家是在 `pkg/ai/catalog` 里改数据——不是再写一份 HTTP 实现。

## 文档

| | |
| --- | --- |
| [API 参考](https://pkg.go.dev/github.com/genai-io/sdk-go/pkg/ai) | 每个类型和函数，设计理由就写在它管辖的代码旁边 |
| [构造客户端](docs/clients.zh-CN.md) | 从模型引用到 `ai.Client` 的一条链，以及在哪里停下（[English](docs/clients.md)） |
| [架构](docs/architecture.zh-CN.md) | 各部分如何拼合，一个请求要经过什么（[English](docs/architecture.md)） |
| [贡献指南](CONTRIBUTING.md) | 开发环境、实现一套协议、测试套件 |
| [更新日志](CHANGELOG.md) | 每个版本改了什么 |
| [`examples/`](examples) | 可运行的程序，每家厂商一个，外加工具调用和结构化输出 |

## 版本

遵循[语义化版本](https://semver.org/lang/zh-CN/)。主版本号还是 `0`，所以 API 在次版本之间仍可能变动；[更新日志](CHANGELOG.md)会写清楚动了什么、该改成什么。

```sh
go get github.com/genai-io/sdk-go@v0.1.0
```

## 许可

[Apache 2.0](LICENSE)
