# 构造客户端

只有一个 `ai.Client`，但到达它有好几条路。这几条路是**一架梯子**：每上一级，你多拿到一分控制权、也多付出一分交代。**每一级产出的都是同一个 `*ai.Client`**，所以下游的代码不关心你走的是哪条。

| 你想要 | 用 |
| --- | --- |
| 一个模型，凭证从环境读 | [`auth.Client`](#1-authclient) |
| 一个模型选择器，或模型要到运行时才知道的 host | [`auth.Provider`](#2-authprovider) |
| 用环境里的凭证，但要改掉某一项 | [`auth.Config`](#3-authconfig-然后-ainewclient) |
| 一台握着多个租户密钥的服务器 | [`ai.NewClient`](#4-ainewclient) |
| 重试、成本统计、缓存、日志 | [`ai.Wrap`](#5-ainewdriver-然后-aiwrap) |
| 不联网的测试 | [`ai.New`](#6-ainew自带-driver) |

每条路都需要对应协议的 driver 已注册，一个空导入就够：

```go
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"           // 五种全要
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"   // 或只要一种
```

模型要到运行时才确定就导入 `all`；编译期就知道协议、又不想把其他厂商的 SDK 链进来，就只导入那一个。

## 1. `auth.Client`

最短的一条，命令行工具就该用它。

```go
client, err := auth.Client("anthropic/claude-opus-5")
```

它在目录里查这个引用，从那家厂商文档规定的环境变量读凭证，返回客户端。**厂商需要 key 而它的环境变量一个都没设时，就在这里失败**——而不是发一个空凭证出去、稍后变成一个莫名其妙的 401。

给每次调用的默认值加在后面：

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh))
```

对于那些认证的是**人**而不是服务的厂商——GitHub Copilot、ChatGPT/Codex——这条路会跑浏览器登录，并把结果以 `0600` 存在用户配置目录下。**那个凭证库是这个库唯一会写的文件。**

## 2. `auth.Provider`

一个 `Provider` 是**一个配好的 host 加上它上面的模型列表**。需要列表、而不只是一个模型时用它。

```go
p, err := auth.Provider("ollama")
if err != nil {
	return err
}

models := p.Models()              // 同步，从不阻塞，从不失败
if err := p.Refresh(ctx); err != nil {
	log.Printf("listing %s: %v", p.Name(), err) // 不致命
}

client, err := p.Client("llama4")
```

**"读列表"和"拉列表"是两个动词**，这是故意的：模型选择器直接从目录立刻渲染出来，而一个挂掉的 host 卡不住它。`Refresh` 是唯一会联网的调用，跑完之后 `Models()` 返回的是目录行与 host 实际提供的内容**合并后**的结果。

要配一个目录里没有的 host，自己建 provider：

```go
p := provider.New(provider.Config{
	ID:      "internal",
	Name:    "Internal gateway",
	API:     ai.APIOpenAIChat,
	BaseURL: "https://gateway.internal/v1",
	APIKey:  key,
})
```

## 3. `auth.Config` 然后 `ai.NewClient`

凭证在环境里、但**别的项要改**时——代理、网关、自定义 `http.Client`：

```go
cfg, err := auth.Config("openai/gpt-4.1")
if err != nil {
	return err
}
cfg.BaseURL = "https://gateway.internal/v1"
cfg.HTTPClient = instrumented

client, err := ai.NewClient(cfg)
```

`auth.Config` 只负责填好凭证和端点就停下，所以它返回的是一个**普通的 `ai.Config`**，用之前你可以随便改。

## 4. `ai.NewClient`

**`pkg/ai` 不读任何环境变量、不读任何文件。**正是这一点让它在一台握着多个租户密钥的服务器上是安全的：它做的任何事都不依赖环境状态，所以**两个请求不可能拿到彼此的凭证**。

```go
model, err := catalog.Model("anthropic/claude-opus-5")
if err != nil {
	return err
}

client, err := ai.NewClient(ai.Config{
	Model:      model,
	APIKey:     tenantKey,
	BaseURL:    "https://gateway.internal/v1",
	HTTPClient: httpClient,
	Headers:    map[string]string{"X-Tenant": tenant},
})
```

`catalog.Model` 是**单独的那一次查表**——端点、限额、价格和该说哪种协议，**不联网、不需要凭证**。目录里压根没有的模型，自己填一个 `ai.Model` 就行。

## 5. `ai.NewDriver` 然后 `ai.Wrap`

执行策略——重试、成本统计、缓存、日志——是**装饰 driver 的 `Middleware`**，所以它需要拿到 driver 这个值。`NewDriver` 就是 `NewClient` 提前一步停下：

```go
driver, err := ai.NewDriver(cfg)
if err != nil {
	return err
}

client := ai.New(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), cfg.Model)
```

`Retry` 是这里唯一自带的策略，而且**每个 driver 都关掉了它厂商 SDK 自己的重试**，所以不加它就一次重试都没有。顺序是**最外层在前**：列表里第一个 middleware 最先看到请求、最后看到响应。

一个 `Middleware` 是"包着 `Handler` 的 `Handler`"，和 `Driver.Stream` 是同一个形状——所以你写的任何东西都能和这里自带的组合起来。

## 6. `ai.New`，自带 driver

`ai.New` 收一个 driver 和一个 model，**不做 I/O、不查表、不校验凭证**。上面那几条路最终都落到它，而测试要的也正是它：

```go
client := ai.New(scripted{deltas: deltas}, ai.Model{
	ID: "test", API: ai.APIAnthropicMessages, MaxOutput: 1024,
})
```

因为 `Driver` 只有两个方法，给某个用例写的桩就是几行。例子见 [CONTRIBUTING.md](../CONTRIBUTING.md#testing-your-own-code)。

## 默认值与覆盖

不管你走哪条路，**同一个 `Option` 在构造时是默认值、在调用处是覆盖值**：

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh))

response, err := client.Complete(ctx, messages,
	ai.WithEffort(ai.EffortLow)) // 只影响这一轮
```

解析顺序是：模型默认 → 客户端默认 → 调用处覆盖。**传了 option 本身就是"显式"的标记**，所以 `ai.WithTemperature(0)` 是确定性采样，不传才是继承。
