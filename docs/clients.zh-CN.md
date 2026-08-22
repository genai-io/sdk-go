# 构造客户端

**只有一条链，你从自己已经在的位置接进去。**它不分叉，所以没有"选错"这回事。

```
   "openai/gpt-4.1"          ai.Config           ai.Driver           *ai.Client
          │                      │                   │                    │
          └─── auth.Config ──────┴─── ai.NewDriver ──┴─── ai.New ─────────┘
                     读环境                  建传输层              装模型和默认值
```

你实际会敲的那两个名字，是这条链上的**捷径**，而且每个都**literally 只有一行**：

```go
// auth.Client = auth.Config + ai.NewClient
func Client(ref string, opts ...ai.Option) (*ai.Client, error) {
	cfg, err := Config(ref)
	if err != nil {
		return nil, err
	}
	return ai.NewClient(cfg, opts...)
}

// ai.NewClient = ai.NewDriver + ai.New
func NewClient(cfg Config, opts ...Option) (*Client, error) {
	d, err := NewDriver(cfg)
	if err != nil {
		return nil, err
	}
	return New(d, cfg.Model, opts...), nil
}
```

三种写法产出的是同一个客户端：

```go
client, err := auth.Client("openai/gpt-4.1")

cfg, err := auth.Config("openai/gpt-4.1")
client, err := ai.NewClient(cfg)

driver, err := ai.NewDriver(cfg)
client := ai.New(driver, cfg.Model)
```

所以问题不是"用哪个构造函数"，而是**你需要在这条链上走到哪里就停**：

| 停在 | 什么时候 | 代价 |
| --- | --- | --- |
| `auth.Client(ref)` | 命令行工具，凭证在环境里 | 会读环境 |
| `auth.Config(ref)` | 同上，但要改端点或 `http.Client` | 会读环境 |
| `ai.NewClient(cfg)` | 服务端。凭证你给，**不读任何环境状态** | 无 |
| `ai.NewDriver(cfg)` | 需要拿到 driver 这个值，去套 middleware | 无 |
| `ai.New(driver, model)` | 你已经有 driver 了——包括测试里的桩 | 无 |

每条路都需要对应协议的 driver 已注册，一个空导入就够。模型要到运行时才确定就用 `all`；编译期就知道协议、又不想把其他厂商的 SDK 链进来，就只导入那一个。

```go
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
import _ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
```

## 停在 `auth.Client`

```go
client, err := auth.Client("anthropic/claude-opus-5", ai.WithEffort(ai.EffortHigh))
```

它在目录里查这个引用，从那家厂商文档规定的环境变量读凭证，返回客户端。**厂商需要 key 而它的环境变量一个都没设时，就在这里失败**——而不是发一个空凭证出去、稍后变成一个莫名其妙的 401。

对于那些认证的是**人**而不是服务的厂商——GitHub Copilot、ChatGPT/Codex——它会跑浏览器登录，把结果以 `0600` 存在用户配置目录下。**那个凭证库是这个库唯一会写的文件。**

## 停在 `auth.Config`

```go
cfg, err := auth.Config("openai/gpt-4.1")
cfg.BaseURL = "https://gateway.internal/v1"
cfg.HTTPClient = instrumented

client, err := ai.NewClient(cfg)
```

`auth.Config` 填好凭证和端点就停下，所以它返回的是一个**你可以随便改的普通 `ai.Config`**。

## 停在 `ai.NewClient`

**`pkg/ai` 不读任何环境变量、不读任何文件。**正是这一点让它在一台握着多个租户密钥的服务器上是安全的：它做的任何事都不依赖环境状态，所以**两个请求不可能拿到彼此的凭证**。

```go
model, err := catalog.Model("anthropic/claude-opus-5")

client, err := ai.NewClient(ai.Config{
	Model:      model,
	APIKey:     tenantKey,
	BaseURL:    "https://gateway.internal/v1",
	HTTPClient: httpClient,
	Headers:    map[string]string{"X-Tenant": tenant},
})
```

`catalog.Model` 是**单独的那一次查表**——端点、限额、价格和该说哪种协议，**不联网、不需要凭证**。目录里压根没有的模型，自己填一个 `ai.Model` 就行。

## 停在 `ai.NewDriver`

执行策略——重试、成本统计、缓存、日志——是**装饰 driver 的 `Middleware`**，所以它需要拿到 driver 这个值：

```go
driver, err := ai.NewDriver(cfg)
client := ai.New(ai.Wrap(driver, ai.Retry(3, time.Second), costMeter), cfg.Model)
```

`Retry` 是这里唯一自带的策略，而且**每个 driver 都关掉了它厂商 SDK 自己的重试**，所以不加它就一次重试都没有。顺序是**最外层在前**：第一个 middleware 最先看到请求、最后看到响应。一个 `Middleware` 是"包着 `Handler` 的 `Handler`"，和 `Driver.Stream` 同一个形状，所以你写的任何东西都能和自带的组合起来。

## 停在 `ai.New`

`ai.New` **不做 I/O、不查表、不校验凭证**。上面每一条路最终都落到它，而测试要的也正是它：

```go
client := ai.New(scripted{deltas: deltas}, ai.Model{
	ID: "test", API: ai.APIAnthropicMessages, MaxOutput: 1024,
})
```

因为 `Driver` 只有两个方法，给某个用例写的桩就是几行。例子见 [CONTRIBUTING.md](../CONTRIBUTING.md#testing-your-own-code)。

## Provider 不在这条链上

上面所有东西造出来的都是**一个模型**的客户端。`Provider` 是它旁边的一层：**一个配好的 host 加上它上面的模型列表**——模型选择器要的就是这个。

```go
p, err := auth.Provider("ollama")

models := p.Models()          // 同步，从不阻塞，从不失败
err = p.Refresh(ctx)          // 唯一会联网的调用

client, err := p.Client("llama4")
```

**"读列表"和"拉列表"是两个动词**，这是故意的：选择器直接从目录立刻渲染，而一个挂掉的 host 卡不住它。目录里没有的 host，用 `provider.New(provider.Config{…})`。

## 默认值与覆盖

不管你停在哪，**同一个 `Option` 在构造时是默认值、在调用处是覆盖值**：

```go
client, err := auth.Client("openai/gpt-4.1", ai.WithEffort(ai.EffortHigh))

response, err := client.Complete(ctx, messages,
	ai.WithEffort(ai.EffortLow)) // 只影响这一轮
```

解析顺序是：模型默认 → 客户端默认 → 调用处覆盖。**传了 option 本身就是"显式"的标记**，所以 `ai.WithTemperature(0)` 是确定性采样，不传才是继承。
