# `pkg/ai` 架构说明

三份文档，三件事。README 讲**怎么用**；`go doc` 把每个决定的理由放在**它管的那段代码旁边**；这一份讲**各部分怎么拼在一起**——哪个包知道什么、依赖朝哪个方向走、一个请求从你的调用到线上的字节之间经过了什么。

> English: [architecture.md](architecture.md)

## 命题

**包是代码的单位，所以一个线格式一个包，一个厂商零个包。**

catalog 里 27 家厂商，由 5 个协议服务：

| 协议 | 包 | 厂商数 |
| --- | --- | --- |
| OpenAI Chat Completions | `driver/openai/chat` | 18 |
| OpenAI Responses | `driver/openai/responses` | 3 |
| Anthropic Messages | `driver/anthropic` | 4 |
| Anthropic on Vertex AI | `driver/anthropic/vertex` | 1 |
| Google Gemini | `driver/google` | 1 |

绝大多数厂商提供的是一个说别人协议的端点。DeepSeek、Moonshot、Ollama 说 OpenAI Chat Completions；MiniMax、小米 MiMo、火山引擎说 Anthropic Messages。真正区分它们的是一个 base URL、一个环境变量、一套 reasoning 方言和一份模型清单——这四样全是数据。

所以**一个厂商是 `catalog/vendors.go` 里的一行，不是一个包**。加一个 OpenAI 兼容的端点是往表里加一条记录；只有新的线格式才需要写 Go 代码。

这条命题是否成立，有个可验证的判据：**没有任何 driver 里出现厂商名**。每个请求构造器分支依据的都是 `Compat` 字段和 `ReasoningLevel` 数据。

## 分层

调用方往下走的那条链：

```
catalog.Vendor      数据 - 不联网就能读的一行
      | .Provider(cfg)
      v
provider.Provider   那一行配好了凭证，带一份会刷新的模型列表
      | .Client(modelID)
      v
ai.Client           一个模型
```

每一级加一种知识，然后往下委托。`catalog` 不需要网络；`provider` 连一台主机；`Client` 跑一次调用。

**每一步都以它交还的东西命名**，所以不管你从哪一级进入，读法都一样：

| 从 | 调用 | 拿到 |
| --- | --- | --- |
| `catalog.Vendor` | `.Provider(cfg)` | `*provider.Provider` |
| `provider.Provider` | `.Client(modelID)` | `*ai.Client` |
| `ai.Config` | `ai.NewClient(cfg)` | `*ai.Client` |
| `ai.Config` | `ai.NewDriver(cfg)` | `ai.Driver` |
| 一个引用字符串 | `auth.Client(ref)` | `*ai.Client` |
| 一个厂商 ID | `auth.Provider(id)` | `*provider.Provider` |

**每一个包级构造函数都收 `Config`，并以它返回什么命名**——`ai.New` 给 `*Client`，`ai.NewDriver` 给 `Driver`，五个 driver 包和 `provider.New` 也是同一个形状。`ai.NewClientWithDriver(driver, model)` 是唯一的例外，而它的名字已经说明了原因：它从**你手里已有的 `Driver`** 出发。

两个包级入口，区别只有一个——**允不允许环境变量说话**：

```go
ai.NewClient(Config)         // 模型、密钥、主机都由你提供
auth.Client("vendor/model")  // catalog 和环境变量提供
```

`pkg/ai` 不读任何环境变量、不读任何文件。这是它能安全地待在一台握着多个租户密钥的服务器里的原因。`pkg/ai/auth` 是那个明确选择去读的包，命令行工具要的正是它。这条分界是 **import 边界，不是约定**。

**protocol 是第二根轴，不是链条的下一级。** 一个 Provider **拥有**一个 protocol，它不**是**一个 protocol。把两根轴画在一起，driver 的位置就清楚了——它挂在 model 上，不在链条上：

```
  auth.Client("deepseek/deepseek-v4-pro")
        |
        v
  catalog.Vendor ---> provider.Provider ---> ai.Client ---> ai.Driver ---> HTTPS
   一行数据             主机 + 凭证             一个模型        一个协议
   不联网              + 会刷新的模型列表           |             ^
                                                  |             |
                                                  +-------------+
                                              Model.API 选中它
```

选中 driver 的是 `Model.API`，**从来不是厂商名**。这正是"厂商可以是数据"的全部理由：关于 DeepSeek 的其他一切，无非是一个 base URL、一个环境变量和一套 reasoning 方言。

基数是区分这两个词的关键：

```
1 个 protocol  ->  多个 provider     18 家厂商说 OpenAI Chat Completions
1 个 provider  ->  多个 model
1 个 model     ->  1 个 client
```

所以 **provider** 命名的是"一台你能连上的、配好的主机"，**protocol** 命名的是"它说的那套请求格式"。`ProtocolOptions` 作用于后者——这就是为什么 `anthropic.Options` 也能作用到 MiniMax 和火山引擎：**它们说那个协议，但不是那个厂商**。

## 27 家厂商，5 个协议，5 个 driver 包

把命题画出来。左列全是数据，只有右列是 Go 代码。

```
  catalog 里的行                  Model.API                  driver 包
  ----------------------------    -----------------------    --------------------------
  deepseek    moonshot    zai  |
  alibaba     bigmodel    xai  |
  ollama      groq      nvidia |
  cerebras    together  agnesai+--> openai-chat-completions --> driver/openai/chat
  fireworks   copilot  sensenova            18 家
  openrouter  huggingface      |
  bedrock-openai               |

  anthropic   minmax           |
  mimo        volcengine       +--> anthropic-messages      --> driver/anthropic
                                            4 家

  openai      azure-openai     |
  openai-codex                 +--> openai-responses        --> driver/openai/responses
                                            3 家

  anthropic-vertex              --> anthropic-vertex        --> driver/anthropic/vertex
  google                        --> google-genai            --> driver/google
```

加第 19 家 OpenAI 兼容厂商，只动左列。

## 核心原语：有序的块

一条消息装的是 `Content []Block`——**按顺序排列的 tagged union**，而不是并行的 `Text`、`Images`、`ToolCalls` 字段。

```
	类型              装在哪                   谁产生的
	BlockText         Text                     两边都可能
	BlockImage        Image                    你
	BlockThinking     Text, Signature          模型
	BlockToolCall     ToolCall                 模型
	BlockToolResult   ToolResult               你
	BlockReasoning    Reasoning                模型
```

**顺序就是重点。** 一个模型先思考、再调工具、然后解释自己，它按那个次序产生了三个块，下一次请求必须原样带回去。并行字段会丢掉顺序，而没有任何协议接受一个靠猜重建出来的对话次序。

这也是 `Response.Message()` 存在的理由，以及为什么改用 `ai.AssistantMessage(resp.Text())` 是个 bug：前者把 thinking 和不可读的 reasoning 状态一并带上，丢掉它们会让一个推理模型每一轮都从头想起。

**实测结果**：交错内容在 `anthropic`、`openai/responses`、`google` 上保序。`openai/chat` 会**压平**它——文字被拼成一个字符串，调用被移进并行的 `tool_calls` 数组——因为那个协议表达不了交错。那是线格式的性质，不是 driver 的 bug。

## 一份对话，加若干选项

```go
client.Complete(ctx, messages, ai.WithSystem(s), ai.WithEffort(ai.EffortHigh))
```

对话就是普通的 `[]ai.Message`。其余一切都是 `Option`，而且同一个 `Option` 放在构造时是默认值、放在调用里是覆盖值——所以一个设置无论在哪里设，写法只有一种。

**选项字段上没有"是否设置"的包装类型。** 调用这个 option 本身就是那个标记位：`WithTemperature(0)` 就是确定性采样，不写它就是继承。早先的设计用 `Optional[T]` 加 `Some(...)` 来表达这个区别，它存在的唯一理由是服务那个"两个 struct"的调用形状，随着那个形状一起消失了。

解析是三层，按顺序应用：模型默认值、客户端默认值、单次调用覆盖。后面的盖住前面的。规则就这一条。

## 归一化，还是透传

每个设置的分界线：

| | 去哪 |
| --- | --- |
| 每个协议都能表达，只是拼写不同 | 一个 `With*` 选项，由 driver 翻译 |
| 只有一个协议有 | 那个 driver 的 `ProtocolOptions` 值，原样透传 |

reasoning 档位是第一类最清楚的例子。"该想多久"这个概念到处都有，而每家拼写都不同——Anthropic 要 `thinking.budget_tokens` 或 `output_config.effort`，Gemini 要 `thinkingLevel`，DashScope 要 `enable_thinking` 加 `thinking_budget`。所以它是 `ai.WithEffort(ai.EffortHigh)`，每个 `Model` 自带一份 `[]ReasoningLevel` 梯子，把档位映射到它的端点想要的东西。

**梯子是数据，不是代码**：没有任何 driver 里有 effort 映射表。一个模型还可以声明本包从没听说过的档位，按名字精确匹配就会原样发出去。一个既不在便携词表里、也不在该模型梯子里的名字会被拒绝，而且错误信息里会列出这个模型实际提供哪些档。

`thinking.display` 是第二类。别的协议没有这个概念，所以它是 `anthropic.Options.ThinkingDisplay`，不经翻译直接送出。

这两个字段**有类型，不是 `any`**：`ai.ProtocolOptions` 和 `ai.ProtocolConfig` 是标记接口，所以一个本来就不该放进去的值是**编译期错误**。而放进去的是**另一个 driver 的**类型时，会在那个 driver 读取的那一刻被抓住——Go 没有 union type，那一半没法做成编译期检查。

## 协议方言

两个端点可以说同一个协议却仍然不一致。`compat.go` 就是端点声明自己差异的地方，形式是一个值：

```go
Compat: ai.OpenAIChatCompat{Thinking: ai.ThinkingEffortOrDisable}   // DeepSeek
Compat: ai.OpenAIChatCompat{Thinking: ai.ThinkingType,
                            ReasoningContent: true}                 // Moonshot
Compat: ai.AnthropicCompat{BearerAuth: true}                        // 火山引擎 Ark
```

这是防止一个 driver 长成一棵厂商特例树的东西。`driver/openai/chat` 里的 `applyReasoning` 服务 18 家厂商，是一个纯粹的 `ThinkingFormat` switch，**里面一个厂商字面量都没有**。

Compat 值由 `catalog` 写入、由各 driver 读取，而 `pkg/ai` 自己在校验时也要读它。这就是这些类型必须待在 `pkg/ai` 的原因：**它是三方都能依赖的唯一一个包**。把它们挪进 driver 包，就会逼着 `catalog` import 每一个 driver，连带把每家厂商的 SDK 拖进来——注册表存在的意义（按需 blank import）就毁了。

## 这个 SDK 拒绝做的事

**只自带一条执行策略，不多给。** `Retry` 在这里，是因为每个 driver 都关掉了厂商 SDK 自带的重试——不给它，调用方一次重试都没有；也因为自己写这个东西错法很隐蔽：**重试只能重放一个在产出任何 delta 之前就失败的调用**，违反了表现是输出重复或凭空消失，不是报错。明知有这么危险的一条规则，却只发警告不发实现，是两者里更糟的那个。

缓存、日志、成本计量仍然归调用方，形式是装饰 driver 的 `Middleware`——只有应用知道一轮的预算、什么可以缓存、什么绝不能记日志。

**不压缩。** 判断一段对话太长、总结它、丢掉最老的几轮，都是应用层拿着这个包没有的信息去做的决定。它做的是**修复**：`Repair` 只删协议本身会拒绝的东西——Ctrl-C 留下的没人应答的工具调用、经过 JavaScript 运行时的对话带来的非法 UTF-8。**修复不是策略。**

**不改写你的请求去让它能跑通。** 一个没有 system 角色的模型、或者一个不能约束输出为 schema 的模型，会得到一个点名该模型的错误。把指令挪进 user 轮、或者用文字要求 JSON，是关于你产品的决定，不是关于线格式的。

**不猜。** 一个上下文窗口未知的模型报告的 headroom 是零，而不是一个替代数字——因为按猜出来的上限行事，两个方向上都会静默出错。

## 一次调用经过什么

从 `client.Complete(ctx, messages, opts...)` 到线上的字节之间：

```
  messages + options
        |
        |  +---------------------- Client.prepare ------------------------+
        +--| newRequest         模型默认值 -> 客户端 -> 单次调用，按此顺序   |
        |  | validateStructure  一个 Block 必须满足的 tagged-union 不变式   |
        |  | Repair      配对工具调用，替换非法 UTF-8                |
        |  | validate           设置本身，然后是这个模型做不到的事           |
        |  +--------------------------------------------------------------+
        v
    *ai.Request  ------------->  Driver.Stream
                                      |   翻译成线格式、发出去、
                                      |   一边读一边吐
                                      v
                               iter.Seq2[Delta, error]      原始片段
                                      |
                                      |   blockTracker 组装
                                      v
                               iter.Seq2[Event, error]      有序的块
                                      |
                                      v
                                 *ai.Response               Complete 抽干它得到
```

**框里的一切，对每个协议只发生一次。** 那就是 driver 待在后面的那条线：它翻译、它流式吐出，别的什么都不做。这也是为什么 `Client.Complete` 不是第二条代码路径——它抽干 `Stream`，所以一个请求到达端点的方式**有且只有一种**。

`CountTokens` 量的是同一个 `*ai.Request`，所以一个 prompt 不可能"量的时候是一个样、发出去是另一个样"。

## 在到达网络之前失败

请求发出前跑三层校验，顺序如下：

1. **结构** —— 一个 `Block` 必须满足的 tagged-union 不变式，以及回放一个块必须满足的协议不变式；
2. **设置** —— 值本身就不合法的，或者跟它所在的对话相矛盾的；
3. **能力** —— 这个具体模型声明自己做不到的事。

目的是给出一句**调用方能据此行动的话**——"model deepseek-v4-pro does not accept image input"——而不是一个含糊的 provider 拒绝，或者更糟：一个静默降级的答案。往一个纯文本端点发图片，以前的结果是图片被丢掉、模型回答了一个它从没见过的东西。

真的到了网络那一层才失败的，回来时是**分好类的**，所以"接下来怎么办"写在类型里：`IsAuth`、`IsContextExceeded`、`IsRetryable`、`IsUnsupported`。一轮失败会同时返回错误**和**一个 `*Response`，里面装着先到的东西——所以一个部分答案和它的花费不会跟错误一起丢掉。

## 哪些边界由编译器保证

这里有些规则是约定，有些是被检查的。值得知道哪个是哪个。

**编译器保证的：**

- **依赖图是单向的，根是 `pkg/ai`。** 它不 import 本仓库的任何东西：

  ```
  缩进表示"import 上面那一行"

  pkg/ai                  核心。不 import 仓库内任何包，不带厂商 SDK，
                          不读环境变量，不读文件。
    |
    +-- pkg/ai/jsonschema Go 类型变成一份 provider 会接受的 schema
    |
    +-- pkg/ai/driver/*   一个协议一个包。厂商 SDK 只被它拖进来，
    |                     别的地方都不碰。
    |
    +-- pkg/ai/provider   一台配好的主机和它会刷新的模型列表
          |
          +-- pkg/ai/catalog    厂商表
                |
                +-- pkg/ai/auth    环境变量、文件、OAuth
  ```

  `catalog` **不依赖任何 driver 包**——它的整棵树只有 `pkg/ai` 加 `pkg/ai/provider`。这就是"只跟一家厂商说话的程序不必链接所有厂商 SDK"的保证，也是 `Compat` 类型必须待在 `pkg/ai` 而不是各自 driver 旁边的原因。

- `driver/openai/internal/errs` 对 `driver/anthropic` 不可达，靠 `internal/`。
- `pkg/ai`、`pkg/ai/jsonschema`、`pkg/ai/catalog`、`pkg/ai/provider` 的**模块依赖数是 0**——除标准库外什么都不依赖。厂商 SDK 只能通过你 blank import 的那个 driver 进入构建。
- `ProtocolOptions` / `ProtocolConfig` 在编译期拒绝一个本来就不该是它的值。
- `Driver` 只有两个方法，所以测试里的桩也只有两个方法。

**只写在文档里的**（意味着只有 review 能拦住）：

- driver 不得修改或持有交给它的 `*Request`。`Client` 在调用返回之后还会再读它。
- middleware 不得重放一个已经产出过 delta 的调用。

## 已知的瑕疵

记在这里，因为一份只列成绩的设计文档不算设计文档。

**`APIAnthropicVertex` 不是一个线格式。** 五个 `API` 值里有四个是真正不同的请求形状。第五个是 Anthropic Messages 换了认证（Google ADC）、换了主机、换了模型 ID 形式；那个 driver 把构造完客户端之后的一切原样交回 `driver/anthropic`。它之所以还是一个独立的 `API` 值，是因为注册表按 `API` 查 driver，而它的 Google Cloud auth 依赖很重——**271 个第三方包，对比 `anthropic` 的 43 个**——所以它必须只落进主动要它的构建里。要正经修，得给注册表加第二个维度，代价大于这个瑕疵本身。

**OpenAI 的 cache-write token 没有被计入。** 端点在 `input_tokens_details.cache_write_tokens` 里报告它们，而当前锁定的 `openai-go` 版本没有暴露这个字段，所以在 GPT-5.6 及以后，那部分 token 被按普通 input 计价——**比真实数字少约四分之一**。厂商条目里写明了这一点。

**`ResolveLevel` 的吸附是静默的。** 在一个只提供 off 和 high 的模型上要 `medium`，会拿到 high，而没有任何东西告诉调用方。方向是有意的——**静默地想得比要求的少，是更让人意外的那种失败**——但这份静默目前调用方观察不到。

## 测试

一个黑盒包，`test/`。它像应用一样 import 这个 SDK，并且只断言两件事：**到达端点的字节，和回来的那个值。** 每个端点都是一个桩 HTTP 服务器，所以整套测试不需要网络、不需要凭证。

这个形状是刻意的。伸进内部去看的测试，验证的是它被写出来时对着的那份实现；这些测试验证的是契约，所以一个在相同线上行为背后被重写的 driver，仍然能跑过。
