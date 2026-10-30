# Khoj 集成 DeepSeek 后 HTTP 400 根因调查与代码审查报告

## 1. 执行摘要

**结论：已确认本地代码存在可确定复现的请求格式兼容缺陷；它是用户所报 HTTP 400 的高置信度根因。** 普通消息先进入数据源和输出格式识别，而非直接生成最终回答。`aget_data_sources_and_output_format()` 同时请求 `response_type="json_object"`、`response_schema=PickTools`；`get_structured_output_support()` 对官方 DeepSeek 地址及新注册模型默认返回 `TOOL=3`，满足 `>= SCHEMA(2)`，使 `openai_send_message_to_model()` 选择 `json_schema`。DeepSeek **Chat Completions** 文档只列出 `text`、`json_object`。[官方接口说明](https://api-docs.deepseek.com/api/create-chat-completion/)

本次用真实业务函数、OpenAI SDK 2.8.1 和 `httpx.MockTransport` 捕获了序列化后的上述请求，未请求真实模型。48 项隔离 pytest 检查通过；另一次直接执行的 DeepSeek 异步流式 SDK 检查通过。异步批量检查发生停滞，未宣称通过。未读取正式数据库、未取得历史 HTTP 请求，因此不把合成配置的捕获结果冒充故障现场记录。

推荐优先完善 `get_structured_output_support()`，让显式 DeepSeek Provider 的 Chat Completions 返回 `OBJECT`，同时在 `openai_send_message_to_model()` 传入 Provider 信息。保留业务层的结构化识别需求，另补本地 Pydantic 校验。**本次没有实施修复。**

证据级别全文统一如下：

- **A／已确认**：当前本地源码可直接证明，或已执行隔离验证。
- **B／高置信度推断**：源码、用户日志和文档一致，但缺故障现场完整配置或请求。
- **C／待验证假设**：需要额外运行证据；附验证方法，不作为直接根因。

## 2. 审查环境

### 2.1 源码基线及约束

| 项目 | 结果 |
|---|---|
| 审查日期 | 2026-10-27，用户时区 Asia/Shanghai |
| 工作目录 | `/home/agent/projects/khoj` |
| 分支 | `feature/three-tier-intent-recognition` |
| HEAD | `ae229ca894c0b80ad84664afcfdde523b5e87057` |
| `git describe --tags --always` | `2.0.0-beta.28-12-gae229ca8`，并非干净发布版本 |
| 实际审查版本 | 上述 HEAD **加本地已有未提交文件内容**，行号均指本地工作树 |
| Python | 3.12.3，`.venv/bin/python` |
| SDK／依赖 | openai 2.8.1、pydantic 2.11.5、httpx 0.28.1、Django 5.1.15、pytest 9.1.1 |
| 项目依赖声明 | `pyproject.toml:45`：`openai >= 2.0.0, < 3.0.0` |
| AGENTS.md | 在项目及父目录搜索未找到 |

任务开始时 `git status --short`：

```text
 M src/khoj/database/adapters/__init__.py
 M src/khoj/database/models/__init__.py
 M src/khoj/processor/conversation/openai/gpt.py
 M src/khoj/processor/conversation/openai/utils.py
 M src/khoj/processor/conversation/utils.py
 M src/khoj/routers/helpers.py
 M src/khoj/utils/constants.py
 M src/khoj/utils/initialization.py
?? compose.db.yml
?? documentation/docs/advanced/deepseek.md
?? src/khoj/database/management/commands/register_deepseek_models.py
?? src/khoj/database/migrations/0100_add_deepseek_chat_model_type.py
?? tests/test_deepseek_provider.py
```

这些文件均在本次任务之前存在。相关集成包括：模型枚举与适配器注册、DeepSeek 默认模型、初始化与管理命令、OpenAI 通用路径分发、token 参数及非思考模式适配、上下文预算、文档和测试。已有 tracked diff 为 8 个文件、186 行增加／71 行删除。本次未更改它们。

本次执行：Git 状态／diff／版本检查；按行读取本地调用链及 SDK 实现；读取官方文档、Issue #1336；在 `/tmp/khoj-response-format-audit/` 编写隔离测试与结果文件；对审查基线 707 个已有 Git tracked／非忽略 untracked 文件计算 SHA-256，交付前再次比较。没有切分支、同步远程、提交、安装依赖或执行业务修复。没有运行注册命令、迁移或数据库查询。

### 2.2 配置事实与未确认内容

**A：源码配置**：`utils/constants.py:14` 注册 `deepseek-flash`、`deepseek-v4-pro`；`database/models/__init__.py:225` 定义 `model_type="deepseek"`；管理命令 `register_deepseek_models.py:18-25` 使用 `DEEPSEEK_BASE_URL`，缺省 `https://api.deepseek.com`，禁用视觉。密钥仅说明来源，不读取或输出值。注册不等于用户／服务端已经选中该模型。

**A：Mock 配置**：模型 `deepseek-flash`、类型 `deepseek`、Base URL `https://api.deepseek.com`，使用完全合成的凭据与消息。网络和数据库连接在测试进程中被阻断，模型传输只经 MockTransport；Django 数据库设置为内存 SQLite，但禁止 `ensure_connection`，未建表或写入数据。

**B／缺口**：故障当时的实际模型名、数据库 `AiModelApi.api_base_url`、fast 模型槽位、Provider 类型、网关及响应头没有现场记录。本次没有读取 `.env` 或凭据，也不把默认模型名视为运行时实值。选中的前置模型未必等于最终回答模型：见第 4 节。

## 3. 错误复现与证据

### 3.1 用户提供的错误及堆栈线索

```text
ERROR khoj.routers.api_chat:
Error processing chat request: Error code: 400
{"error":{"message":"This response_format type is unavailable now",
"type":"invalid_request_error","param":null,"code":"invalid_request_error"}}
```

用户提供的是整理后的调用链，不是本次采集的原始堆栈。核实后的链条为：

```text
process_chat_request → event_generator
  → aget_data_sources_and_output_format
  → send_message_to_model_wrapper → send_message_to_model
  → openai_send_message_to_model → completion_with_backoff
  → client.beta.chat.completions.stream → SDK create → POST /chat/completions
```

`Error processing chat request` 对应当前 `api_chat.py:1727-1730` 的 WebSocket 请求处理器。HTTP POST 入口则另见 `api_chat.py:1733-1761`，不会先调用 `process_chat_request()`。

### 3.2 实际验证结果

**A**：真实 `aget_data_sources_and_output_format()`、真实异步 wrapper、真实分发、真实参数构造及 SDK 序列化组合，在 Mock DeepSeek 响应 400 时，捕获一次 `/chat/completions` 请求：`model=deepseek-flash`、`response_format.type=json_schema`、schema 名 `PickTools`，必需字段 `source`／`output`。即使配置了第二个 fallback 模型，也没有请求第二个模型。

**A**：另以真实 `event_generator()`、Mock 识别器和 Mock 最终生成器验证：普通问候及显式 `/default` 调用识别器；模拟 400 立即传播，最终生成和普通保存函数均未调用。`/general` 跳过识别器，能产生合成答案和调用 Mock 保存。

**未执行**：真实 DeepSeek API、正式数据库现场配置、浏览器端到端验证，以及故障历史请求恢复。没有声称完成真实服务复现或修复验证。

## 4. 完整调用链分析

### 4.1 从入口到 SDK 的层级结构

```text
WebSocket /api/chat
└─ process_chat_request                         api_chat.py:1592,1643-1653
   └─ event_generator                          api_chat.py:676
      ├─ get_conversation_command              helpers.py:259-281
      ├─ 会话／Agent／就绪状态／记忆／中断上下文   api_chat.py:941-998
      └─ 仅 [Default] → 前置选择                 api_chat.py:1000-1014
         └─ aget_data_sources_and_output_format helpers.py:366-442
            ├─ 来源与输出选项、Prompt、PickTools
            └─ send_message_to_model_wrapper  helpers.py:1640-1726
               ├─ primary／fast 槽位／fallback／上下文截断
               └─ send_message_to_model       helpers.py:1581-1610
                  └─ openai_send_message_to_model  openai/gpt.py:27-94
                     ├─ get_structured_output_support  openai/utils.py:735-744
                     ├─ 构造 response_format=json_schema  gpt.py:49-69
                     └─ completion_with_backoff   openai/utils.py:83-277
                        ├─ get_openai_client    utils/helpers.py:1141-1155
                        └─ SDK stream 上下文    openai/utils.py:175-181
                           └─ POST {base_url}/chat/completions
                              └─ HTTP 400／BadRequestError
      后续未到达：来源工具执行、最终 agenerate_chat_response (api_chat.py:1392)
      后续未到达：最终流式答案 (1413-1440)、正常保存 (1449-1470)
```

HTTP POST `chat()` 直接构建同一 `event_generator()`；`body.stream=True` 返回 StreamingResponse，否则 `read_chat_stream()` 聚合。前端传输是否流式与前置 LLM 是否流式是两层独立设置。

| 文件 | 函数 | 主要作用 | 关键参数 | 是否涉及错误 |
|---|---|---|---|---|
| `routers/api_chat.py:1592,1643-1653` | `process_chat_request` | WebSocket 缓冲、消费事件 | body、scope、headers | 接收传播异常并发送通用错误 |
| `routers/api_chat.py:676,1000-1014` | `event_generator` | 命令分流、准备工具、生成回答 | 默认命令调用识别器 | 此处前置 await 抛出，只有 ValueError 回退 |
| `routers/helpers.py:366-442` | `aget_data_sources_and_output_format` | 选择资料来源和呈现方式 | json_object、PickTools、fast=True | 请求不支持格式的业务触发点 |
| `routers/helpers.py:1640-1726` | `send_message_to_model_wrapper` | 选模型、fallback、上下文 | response_* 原样传递、tools=None | 普通 400 不跨模型回退 |
| `routers/helpers.py:1581-1610` | `send_message_to_model` | 按 Provider 分发 | name、API 配置、provider_type | DeepSeek 与 OpenAI 共用实现 |
| `conversation/openai/gpt.py:27-94` | `openai_send_message_to_model` | 选择接口、生成 kwargs | 能力等级、schema、type、tools | 不兼容 response_format 在此构造 |
| `conversation/openai/utils.py:735-744` | `get_structured_output_support` | 模型名前缀／host 规则 | model、base_url；无 provider_type | DeepSeek 落入默认 TOOL |
| `conversation/openai/utils.py:95-181` | `completion_with_backoff` | SDK 客户端、采样、token、流式 | model_kwargs、provider_type | 175-181 行 SDK 上下文发出请求 |
| `utils/helpers.py:1141-1155` | `get_openai_client` | 普通／Azure SDK 客户端 | API key、base_url | 官方 DeepSeek 使用普通 OpenAI 客户端 |

### 4.2 为什么普通消息需要 JSON

这是应用内部的控制请求：先判定用一般知识、个人笔记、在线资料或其他工具，再决定文字、图片等输出。`PickTools` 返回的结构用于后续路由，用户最终仍可收到普通文本。`helpers.py:383-408` 根据用户文档、可用功能、Agent 限制准备选项，`416-422` 拼接 Prompt，`426-442` 调用模型。因此“普通文本消息”并不意味着全过程只有一次普通文本 LLM 请求。

该阶段属于**前置工具／数据源与输出模式联合选择**，也可宽泛称意图规划；它不是最终答案生成，也不是已执行的实际 tool calling。

### 4.3 命令与模型来源

`get_conversation_command()` 使用 `startswith`：`/general` 返回 General（262-263）；`/notes`、`/online`、`/webpage`、`/image`、`/diagram`、`/code`、`/research` 等有专门分支，跳过本次统一识别，但后续仍可能各自请求模型。`/operator` 还依赖功能启用。没有独立的 `/default` 分支，它与无命令、未知命令一样落到 Default（280-281），仍触发识别。选定命令后才在 `api_chat.py:1026-1030` 移除相应命令文本。

前置调用 `fast_model=True`（helpers.py:438）。wrapper 从 `ConversationAdapters.aget_default_chat_model(user, agent_chat_model, fast=True)` 取主模型；适配器 `database/adapters/__init__.py:1295-1370` 按服务端设置、订阅／fast 配置、显式 Agent fallback、用户设置及首个模型选择；备用模型来自 `1421-1454`，槽位逻辑见 `1457-1483`。最终回答另用 `aget_valid_chat_model()`（helpers.py:1947）。**不能仅凭 UI 中最终模型的名称推定前置模型。**

## 5. `response_format` 参数构造过程

### 5.1 业务初始参数与 Schema

本地 `routers/helpers.py:426-442`：

```python
class PickTools(BaseModel):
    source: List[str] = Field(..., min_items=1)
    output: str

raw_response = await send_message_to_model_wrapper(
    relevant_tools_prompt,
    # query_files / query_images / memories 省略
    response_type="json_object",
    response_schema=PickTools,
    fast_model=True,
    agent_chat_model=agent_chat_model,
    user=user,
    tracer=tracer,
)
```

**A**：`response_type` 在该调用点确定为字符串 `"json_object"`；`response_schema` 为函数内定义的 Pydantic **类** `PickTools`，不是实例或已编码 schema。`source` 预期非空字符串列表，`output` 为字符串；字段值还须与可用选项匹配。未传 `tools`，wrapper 默认 `None`（1652）；`deepthought` 默认 False（1653）。wrapper `1708-1715` 和分发 `1600-1610` 未覆盖这些值。

### 5.2 能力识别及真值分析

`openai/utils.py:735-744` 完整判断：

```python
if model_name.startswith("deepseek-reasoner"):
    return StructuredOutputSupport.NONE
if api_base_url:
    host = urlparse(api_base_url).hostname
    if host and host.endswith(".ai.azure.com"):
        return StructuredOutputSupport.OBJECT
    if host == "api.deepinfra.com":
        return StructuredOutputSupport.OBJECT
return StructuredOutputSupport.TOOL
```

枚举 `conversation/utils.py:1273-1277`：`NONE=0`、`OBJECT=1`、`SCHEMA=2`、`TOOL=3`。这是 `int, Enum`，所以 **`TOOL >= SCHEMA` 为 True**。当前把能力当成累计等级使用，隐含了“支持工具就支持 schema”的假定；这个假定不适用于所有 OpenAI-compatible 服务。

| 模型／地址（合成或静态输入） | 真实函数结果 | 原因 |
|---|---|---|
| deepseek-flash／api.deepseek.com | TOOL | 没有命中例外 |
| deepseek-v4-pro／api.deepseek.com/v1 | TOOL | 路径不影响 host，仍无例外 |
| deepseek-chat／api.deepseek.com | TOOL | 仅 reasoner 前缀例外 |
| deepseek-reasoner／api.deepseek.com | NONE | 首个分支优先 |
| alias／自定义代理地址 | TOOL | 未知 host 默认最高等级 |
| gpt-4o-mini／None | TOOL | 默认值 |
| custom／api.deepinfra.com | OBJECT | host 例外 |
| custom／任意 .ai.azure.com | OBJECT | host 后缀例外 |

以上均已执行函数测试。`model_type` 不参与该函数；虽然新增分发已把 `provider_type` 传给 gpt.py，但 gpt.py:45 只向能力函数传 model 和 URL。**类型信息没有在整条链中丢失，而是在能力识别这一层未被使用。**

### 5.3 分支优先权

`openai/gpt.py:43-71`：

```python
model_kwargs = {}
use_responses_api = provider_type != "deepseek" and supports_responses_api(model, api_base_url)
json_support = get_structured_output_support(model, api_base_url)
strict = not is_cerebras_api(api_base_url)
if tools and json_support == StructuredOutputSupport.TOOL:
    model_kwargs["tools"] = to_openai_tools(...)
elif response_schema and json_support >= StructuredOutputSupport.SCHEMA:
    cleaned_response_schema = clean_response_schema(response_schema)
    # Responses 分支用 text.format；否则：
    model_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "schema": cleaned_response_schema,
            "name": response_schema.__name__,
            "strict": strict,
        },
    }
elif response_type == "json_object" and json_support == StructuredOutputSupport.OBJECT:
    model_kwargs["response_format"] = {"type": response_type}
```

实际 DeepSeek／PickTools 输入条件逐项计算：

1. `provider_type == "deepseek"` → `use_responses_api=False`，即使外层判断 URL 支持 Responses 也禁用。
2. 官方 DeepSeek host、新模型名 → `json_support=TOOL(3)`。
3. `tools=None` → 第一个 if 为 False。所谓“PickTools”只是输出 schema 名，不能据此认为传入了 SDK tools。
4. `response_schema=PickTools` 为真，且 `3 >= 2` → 第二分支为 True。
5. `strict=True`：官方 DeepSeek URL 不是 Cerebras。
6. 采用 Chat Completions schema 形状；后面的 json_object elif 不再执行。

这是 **分支优先级导致 json_object 请求意图未被采用**；并非先赋值 json_object 再由后续代码覆盖。三个 if／elif 分支互斥，tools 优先，但只有存在 tools 且等级恰为 TOOL 才进入工具分支。

**已执行的完整等级矩阵**（均无 tools、有 PickTools、`response_type=json_object`、Chat Completions）：

| 等级 | tools 条件 | schema 条件 | OBJECT 条件是否选中 | 结果 |
|---|---|---|---|---|
| NONE(0) | False | False | 否 | 不发送 response_format |
| OBJECT(1) | False | False | 是 | `{type: json_object}` |
| SCHEMA(2) | False | True | 未执行 | `{type: json_schema, json_schema: …}` |
| TOOL(3) | False | True | 未执行 | `{type: json_schema, json_schema: …}` |

额外已验证：有 tools 且 TOOL → 只构造 tools；无 schema、仅请求 json_object 且 TOOL → 不发送 response_format，因为最后一分支使用 `== OBJECT`。后者是独立于本次触发方式的兼容缺口。

### 5.4 Schema 清洗与最终 kwargs

`clean_response_schema()` 在 `openai/utils.py:1299-1323` 调用 `model_json_schema()`、SDK `_ensure_strict_json_schema()`，再删除属性上的 minItems／maxItems 等。因此 `source` 的最小长度约束没有完整保留到线上 schema；这也说明本地校验有价值。

随后 `completion_with_backoff()`：

| 本地位置 | 操作及结果 |
|---|---|
| utils.py:106-111 | deepcopy kwargs，使用 API key 与 Base URL 选缓存客户端；不输出缓存键 |
| 113-116 | `stream = not is_non_streaming_model(...)`；DeepSeek 为 True，加入 include_usage |
| 118-120 | temperature=0.6；top_p 缺省 0.95；max_completion_tokens 缺省 16000 |
| 122 | 格式化消息 |
| 125-127 | 显式 DeepSeek：max_completion_tokens 改为 max_tokens；extra_body 合入 thinking disabled |
| 142-164 | 因是 elif，显式 DeepSeek 不再走旧 DeepSeek 名称等其他分支 |
| 166-168 | 非本地 read timeout=60 秒；设置了 KHOJ_LLM_SEED 才额外加入 seed |
| 175-181 | `client.beta.chat.completions.stream(messages=…, model=…, timeout=…, **model_kwargs)` |

上述流程**没有删除或降级 response_format**。`api_base_url` 用于客户端构造，不在 HTTP JSON body 中；tools 没有构造，也不在本次 body 中。温度、top_p 与 token 默认值均来自同步 completion 函数，不由用户 `body.stream` 决定。

### 5.5 捕获的最终 HTTP 请求

以下来自**真实 PickTools 业务函数链 + 真实 SDK + MockTransport**，消息和认证头省略；它不是历史故障抓包。测试未设 seed。请求 URL 为 `https://api.deepseek.com/chat/completions`：

```json
{
  "model": "deepseek-flash",
  "max_tokens": 16000,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "properties": {
          "source": {
            "items": {"type": "string"},
            "title": "Source",
            "type": "array"
          },
          "output": {"title": "Output", "type": "string"}
        },
        "required": ["source", "output"],
        "title": "PickTools",
        "type": "object",
        "additionalProperties": false
      },
      "name": "PickTools",
      "strict": true
    }
  },
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": 0.6,
  "top_p": 0.95,
  "thinking": {"type": "disabled"}
}
```

`extra_body` 经 SDK 合并为顶层 `thinking`，并不以 extra_body 字段发送。`timeout=httpx.Timeout(30, read=60)` 是客户端超时设置，也不在请求体中。

SDK 本地证据：`.venv/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py:1494-1500` 中 stream helper 以 `self.create`、`stream=True`、转换后的 response_format 发请求；`create` 使用 `/chat/completions`（该文件 1190 行）。因此 Python 命名空间里的 `beta` **不代表访问 DeepSeek `/beta` 地址，也不代表 Responses API**。非流式分支 `openai/utils.py:219-225` 调 `beta.chat.completions.parse()`；不是这次默认 DeepSeek 分支。

### 5.6 与文档及 OpenAI 的对照

查阅日期 2026-10-27：DeepSeek Chat Completions 的 `response_format.type` 列举为 text／json_object，没有 json_schema 或通过该字段提交完整 schema 的契约；流式 include_usage、max_tokens、thinking disabled、temperature、top_p 均有相应定义。JSON Output 指南要求提示词提及 JSON、提供示例，合理控制输出长度，并提醒可能空内容。本地 Prompt `prompts.py:708-751` 已有多个示例和 JSON 指令，但 751 行的格式示意有引号笔误（`"output'`），是可修正的提示词质量问题，非本次 400 根因。[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)

合成官方 OpenAI 配置 `provider=openai, base_url=None, model=gpt-4o-mini` 实际捕获 `/v1/responses`，使用 `text.format.type=json_schema`；合成 OpenAI 类型代理配置则捕获 `/v1/chat/completions` 的 response_format schema。区别来自 `supports_responses_api()`（utils.py:868-879）按 Base URL 判断，而非模型名。DeepSeek 当前集成显式强制 Chat Completions，因此不能用其他接口的结构化输出能力为这个请求辩护。

**其他参数审查边界**：新 DeepSeek 分支已将 max_completion_tokens 转为 max_tokens，这一点通过 Mock 验证；没有证据表明它们造成当前 response_format 错误。若配置为旧 `model_type=openai` 或非官方网关，token／thinking／seed 接受程度仍须按服务版本验证（C）。不能把这些可能性与已明确的格式冲突并列为确定根因。

## 6. 根因结论

### 6.1 分层结论

| 层级 | 结论 | 证据强度 |
|---|---|---|
| 直接原因 | 向 DeepSeek Chat Completions 提交其文档未支持的 `response_format.type=json_schema` | 合成运行已捕获 A；对应历史故障为 B |
| 代码根因 | `get_structured_output_support` 对未知／DeepSeek host 默认 TOOL，调用方把 TOOL 隐含视为 SCHEMA 的超集 | A：utils.py:735-744、gpt.py:49、枚举 1273-1277 |
| 集成遗漏 | 已新增 Provider 分发、接口选择与 token 适配，但能力函数未接收 Provider，未加 DeepSeek 格式策略 | A：当前 diff 与 gpt.py:44-45、helpers.py:1610 |
| 触发条件 | 默认命令；前置实际模型落入上述能力规则；携带 PickTools；无 tools；请求到不支持该格式的接口 | A：分支／Mock，现场模型配置未核实 |
| 普通聊天受阻原因 | 前置选择先于最终回答，400 非 ValueError，未触发该处回退 | A：api_chat.py:1000-1014、1392 |

OpenAI 与 DeepSeek 共用传输实现并不意味着参数契约相同。普通 OpenAI 路径保留 schema 能力符合其已有设计；DeepSeek 需要针对实际 API 类型约束能力。

**根因证据是否完整**：代码缺陷及参数因果链完整；历史事件闭环尚缺当时模型选择、URL、响应头和请求记录。报告不声称真实 DeepSeek 已被重放。后续可在 SDK 出口采集脱敏的 provider、model、接口路径、response_format.type、schema 名、尝试次数；不记录密钥或完整消息，再与故障日志关联。

### 6.2 Issue #1336 的对应关系

[Khoj Issue #1336](https://github.com/khoj-ai/khoj/issues/1336) 报告 v1.42.10 的同名 400、前置选择失败及空回复。评论引用旧 `get_openai_api_json_support`／`JsonSupport`，建议官方 DeepSeek host 返回 OBJECT；本地已是 `get_structured_output_support`／`StructuredOutputSupport`，缺省由 SCHEMA 变成 TOOL，但比较仍进入 schema 分支。故障模式相同，版本及实现并不相同。新增 Provider 保留了旧类缺陷，不能据此断言此前曾修好又回归。Issue 的 host 修正适合官方地址；当前显式 Provider 也支持代理，更宜结合 provider 与实际接口能力。其重试和空消息叙述须重新验证，不能照搬。

## 7. 相关潜在问题

### P0：阻断核心聊天

1. **默认聊天前置请求使用不兼容 schema（A；历史归因为 B）**。见第 5、6 节。它是本次首要缺陷；UI 选中了另一个前置模型或使用 `/general` 时不一定触发。

### P1：功能或兼容问题

1. **缺少可靠的本地 PickTools 校验（A，已测试）**。`helpers.py:444-478` 只 clean_json、json.loads、取字段与过滤选项，没有 `PickTools.model_validate_json`。SDK 收到的是 dict 形式 response_format，未直接收到 Pydantic 类，不会自动执行 PickTools 类型校验。SDK `_parsing/_completions.py:233-246` 将 dict 排除在 rich response format 之外。合法但缺 `output` 的 JSON 会默认为 text，缺 `source` 则回退；`source` 为字符串时会被按字符迭代，最后落入回退，而不是明确报告结构错误。仅靠 json_object 不能保证业务字段正确。
2. **NONE 或 schema 缺省情况下可能没有 JSON 格式约束（A）**。reasoner 前缀直接 NONE；TOOL 且只有 json_object 无 schema 时也不发送 response_format。通用能力枚举的顺序不能准确表达独立的工具、JSON object、JSON schema 能力。限定本次修复范围，不要求立即大规模重构。
3. **`text=None` 可能触发异常处理中再次异常（A，直接 Mock 已验证）**。clean_json(None) 在 `helpers.py:445` 赋值前失败，except 日志第 476 行引用未赋值的 response，得到 UnboundLocalError。正常同步 completion 对空文本先抛 ValueError，因此本次正常路径不易到达；不要把它当作此次 HTTP 400 的根因。非法 JSON 字符串不会导致该问题，因为第 445 行已经赋值。
4. **无针对性格式降级（A）**。当前不会把 schema 400 转成 JSON object，也不会对该 400 跨模型回退。这避免了无区别重试，但不能恢复聊天。修复应先解决能力识别；如果加降级，必须受 Provider、接口和明确错误原因约束。

### P2：错误处理、体验及维护

1. **WebSocket 错误对用户信息不足（A 静态；浏览器表现 B）**。`api_chat.py:1728` 发送 `{"error":"Internal server error"}`，具体错误只在服务端；前端 `src/interface/web/app/chat/page.tsx:362-365` 对该 controlMessage 只 console.error 和重置状态，不显示模型错误。因此可能只留下空白／未完成的答案区域。断网分支另有 toast（285-335），不能表述为整个前端没有任何错误提示。未运行浏览器。
2. **空 Assistant 持久化有条件发生（A 静态；当前故障实际是否落库 C）**。正常保存 `api_chat.py:1449-1453` 不会在这个前置 400 后到达，Mock 也验证未调用。HTTP disconnect 监控 `753-785`、WebSocket interrupt／cancel 分支 `808-831` 明确传 `chat_response=""`。若前置失败期间同时断开或中断，可能保存空回复；没有证据证明本次一定发生。验证办法：Mock 断连／中断与 save，或经授权只读检索该 turn 的时序，不写数据库。
3. **日志可能带上用户／响应内容（A 静态）**。completion 空响应日志 `utils.py:268` 会包含最后一条消息；helpers.py:476 会记录解析响应。后续改善脱敏和诊断元数据即可，本次测试只用合成输入，没有打印真实内容。
4. **输出 schema 清洗会移除数组长度要求（A）**。minItems 清洗见第 5.4 节，不能依靠远端 schema 代替本地校验。

### 7.1 重试与异常传播的准确判断

- `completion_with_backoff` 的 tenacity 装饰器（utils.py:83-94）只重试 APITimeoutError、RateLimitError、InternalServerError、ValueError，最多 2 次，reraise=True。**不含普通 BadRequestError**。
- SDK 2.8.1 `_base_client.py:750-783` 缺省重试 408、409、429、5xx，不重试普通 400；若服务器响应头 `x-should-retry=true` 则优先要求重试。这是额外条件，现场响应头未知。
- wrapper `helpers.py:1717-1726` 只有 `is_retryable_exception(e)` 为真才跨模型继续；其定义 `conversation/utils.py:125-163` 对普通 OpenAI BadRequestError 返回 False。测试配置了 fallback，实际仍只发 1 次。
- 400 从 planner 请求 await 处（430-442）抛出，尚未进入 JSON 解析 try（444）；`event_generator` 1000-1014 只捕获 ValueError；最终由 process_chat_request 广义 except 处理并继续 raise。
- 因此**不能宣称当前普通 400 存在无意义重复请求**。旧 Issue 的重试记录不适用于本地这套装饰器。空响应则会在 utils.py:266-269 产生 ValueError，测试实际发出 2 次请求；wrapper 可再选 fallback，最坏请求数由模型数与每层重试共同决定。
- JSON 字符串格式错误、字段缺失不会在同步 SDK 路径自动重试；测试中返回原文本，交由上层解析回退。空字符串在真实同步路径先被 ValueError 拦住；直接 Mock planner 得到空字符串则解析失败后回退。这两个测试边界不能混为一谈。

## 8. 修复方案比较

**只提出方案，不执行。**

| 方案 | 修改位置 | 优点 | 缺点 | 回归风险 | 推荐程度 |
|---|---|---|---|---|---|
| A：Provider＋接口能力识别 | utils.py 的 get_structured_output_support；gpt.py 调用点 | 集中策略，显式 DeepSeek 与代理均可识别；保留其它默认行为 | 需传递 provider；不能将 JSON 与工具能力永久视为线性等级 | 低至中，取决于默认规则是否保留 | **首选** |
| B：参数构造处兼容分支 | gpt.py 的 openai_send_message_to_model | 改动集中，直接阻止不支持字段 | 同一 Provider 规则易分散；能力函数仍误报 | 低至中 | 可作短期窄修复 |
| C：仅增加官方 host OBJECT 特例 | utils.py 的能力函数 | 修改最小，沿用旧 Issue 思路 | 自定义代理漏判；reasoner 前缀优先返回 NONE | 官方直连低，覆盖不足 | 有条件使用 |

### 8.1 方案 A（推荐）

1. **修改思路**：能力函数增加可选 `provider_type`／接口标识，针对 DeepSeek Chat Completions 返回 OBJECT；必要时为旧 OpenAI 类型的官方 DeepSeek URL 增加兼容识别。保持 OpenAI、Azure／DeepInfra、Anthropic、Google 的既有分支。旧 reasoner 的 NONE 策略不能被无条件覆盖，应根据实际版本和模型能力单独确定。
2. **涉及文件和函数**：`processor/conversation/openai/utils.py:get_structured_output_support`；`openai/gpt.py:openai_send_message_to_model` 的能力调用；可在同函数定义明确的 object／schema 条件。现有 `routers/helpers.py:send_message_to_model` 已传 provider，通常不需要再改分发接口。
3. **预计范围**：核心 2 个业务文件、数十行以内；加本地输出校验需 `routers/helpers.py:aget_data_sources_and_output_format` 少量配套。测试另行新增，届时再授权实施。
4. **优点**：把能力判断集中到适当层；对 DeepSeek 名称别名和自定义代理更可靠；尽量复用原参数构造流程。
5. **风险**：显式 DeepSeek 指向具有不同契约的代理时需有清楚覆盖策略；不能以品牌代表其所有历史模型／所有接口；返回 OBJECT 后 tools 支持是否降级，需单独测试。若工具能力是后续功能，需要分离能力标志，但不扩大本次阶段一文本聊天修复。
6. **兼容性**：只改变目标 Provider＋Chat Completions，保持未显式传 provider 时的旧行为；不改 Anthropic／Google 实现。
7. **必要测试**：第 9 节全部核心矩阵，特别是 DeepSeek 官方／代理、旧 reasoner、schema-only 调用、tools 调用与已有 OpenAI Responses。
8. **推荐判断**：首选。针对 schema-only 的调用，若仅有 response_schema 而无 json_object，当前 OBJECT 分支不会发送格式；应明确统一为 JSON object 并确保 Prompt 提供结构说明，而不是仅改枚举就宣布所有结构化任务已兼容。

示意逻辑（非实现）：

```python
capability = get_structured_output_support(model, base_url, provider_type, api_kind)
if tools and capability.supports_tools:
    ...
elif response_schema and capability.supports_schema:
    ...
elif (response_schema or response_type == "json_object") and capability.supports_json_object:
    response_format = {"type": "json_object"}
# 接收结果后：业务层验证 schema 与可用选项
```

示意中的独立标志用于表达边界，不要求本次修复立即更换全项目枚举。

### 8.2 方案 B

1. **修改思路**：在 gpt.py 的 schema 分支前，按显式 `provider_type==deepseek` 与 Chat Completions 条件优先构造 json_object；保留其他 Provider 的原分支。不要在已经序列化后到处删除字段。
2. **涉及文件**：核心仅 `openai/gpt.py:openai_send_message_to_model`；本地 JSON 校验可配套 `routers/helpers.py`。
3. **预计范围**：核心约 5-20 行，加校验和测试另计。
4. **优点**：易审阅，直接复用已经传入的 provider，适合作为紧急窄修复。
5. **缺点及风险**：`get_structured_output_support` 仍返回错误能力；未来其他调用方可能重复犯错；tools 优先权必须明确，不能强行让任何带 tools 的请求也变成 JSON object。Provider 数量增长后分支会堆积。
6. **兼容性**：严格限于 DeepSeek 时不影响其它实现；若依据 `deepseek` 模型名前缀笼统处理，可能误伤其他厂商托管且支持 schema 的模型，因此不推荐纯名称判断。
7. **必要测试**：DeepSeek type＋官方／代理 URL、tools 与 schema 共存、OpenAI Chat／Responses、其它 Provider 分发回归。
8. **推荐判断**：次选，可用于快速修复，但应避免与能力函数长期维持两套冲突规则。

### 8.3 通用输出校验与 400 降级边界

`json_schema`（在服务支持时）可约束字段、类型和结构；`json_object` 主要约束 JSON 语法，不能保证 PickTools 必需字段、数组长度或值域。降级后仍须 Pydantic 验证，再做 Agent 选项过滤。建议分别处理：空文本、JSONDecodeError、ValidationError、不允许的来源／输出；失败时使用类型一致的 General／Text 安全默认值，或返回清晰错误。不要记录完整用户消息或模型原文。

不要优先修改 `aget_data_sources_and_output_format()` 去删除 schema 或绕过识别：那会让其它 Provider 的原有结构约束退化，也无法覆盖其他结构化业务调用。`send_message_to_model()` 是分发层，当前已经传 provider，通常不是最小修复处。

优先静态选择受支持格式，必要时才做**至多一次、可观测**的特定格式降级：须确认 Provider＋接口＋错误明确表示 unsupported response_format，且还没有输出部分内容。不能对所有 HTTP 400 自动移除 response_format：400 可能是无效模型、错误消息格式、token 限制或工具参数问题；盲目重试会掩盖配置问题、改变业务输出契约，还可能把原本的结构化请求变成无法解析的文本。保留其它 400 原因，计数降级并记录脱敏元信息。

## 9. 测试与验证结果

### 9.1 执行方式和隔离边界

审查脚本：`/tmp/khoj-response-format-audit/test_audit.py`；设置：同目录 `audit_settings.py`；输出：`results.json`、`real-planner-wire.json`、`pytest-final-safe.txt`。这些是系统临时文件，不是项目新增测试文件。依赖全部使用现有 `.venv`。

既有 pytest 配置有 `--reuse-db`，`tests/conftest.py:42-44` 的 autouse fixture 强制数据库访问；直接跑整个项目测试会建立／接触测试数据库。为遵守本次只读边界，没有运行该入口或数据库注册／初始化测试。先复用 `tests/test_deepseek_provider.py` 中无需 DB 的现有函数，由临时测试直接调用；再补 Mock 用例。原文件未修改。临时用例针对观察到的当前行为，不代表修复验收测试已经通过。

最终已完成的命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
USE_EMBEDDED_DB=false PYTHONPATH=src:/tmp/khoj-response-format-audit \
timeout 60 .venv/bin/python -m pytest -c /dev/null --noconftest \
-p no:cacheprovider /tmp/khoj-response-format-audit/test_audit.py -q \
-k 'not existing_async_sdk and not existing_general_event' --tb=short
```

**结果：48 passed, 5 deselected, 20 warnings in 6.08s。** 警告主要为 Pydantic 旧配置／min_items 弃用，以及禁用插件自动加载后原文件 asyncio marker 未注册；它们不等于测试失败。异步测试明确排除，不用已通过数字掩盖。

### 9.2 已运行矩阵

| 场景 | 方法与边界 | 结果 |
|---|---|---|
| A：普通默认消息 | 真实 event_generator，外部依赖 Mock，识别器抛合成 400 | 调识别，不到达最终生成／正常保存 |
| A2：显式 /default | 同上 | 与默认相同 |
| B：/general 普通消息 | 真实 event_generator，Mock 生成器／保存 | 跳过识别，产生合成文本 |
| C：DeepSeek json_object＋PickTools | 真实 planner→wrapper→分发→SDK；仅 DB、上下文构造及 HTTP 响应 Mock | 捕获 json_schema，400 后 1 次请求、不走 fallback |
| D：OpenAI 对照 | SDK MockTransport 捕获代理 Chat 和官方 Responses 参数 | 分别保持 response_format schema、text.format schema |
| E：NONE／OBJECT／SCHEMA／TOOL | 替换能力返回，其余参数构造执行真实函数 | 无 format／object／schema／schema |
| E2：tools 优先 | 替换工具转换输出并观察 kwargs | TOOL＋有 tools 时仅构造 tools |
| E3：仅 json_object、无 schema | 真实默认 TOOL 能力 | 不构造 response_format |
| F1：合法 source/output JSON | Mock wrapper 文本、真实 planner 解析 | 返回 General／Text |
| F2：非法 JSON 字符串 | 同上 | 返回 Default／Text，无 UnboundLocalError |
| F3：缺 output | 同上；另用 Pydantic 作对照 | planner 默认 text；Pydantic 拒绝缺字段 |
| F4：缺 source、空对象、错误字段类型 | 同上 | 当前回退，无完整 schema 校验 |
| F5：空字符串（planner 边界） | Mock wrapper 返回空文本 | 解析失败后 Default／Text |
| F6：text=None（planner 边界） | Mock wrapper 返回 None | UnboundLocalError；正常 completion 空响应保护另算 |
| F7：空文本（SDK 完整路径） | Mock SSE、真实同步 SDK 与 tenacity，取消测试睡眠 | 发 2 次后 ValueError |
| F8：非法／缺字段文本（SDK 路径） | Mock SSE、真实 SDK，发送 dict schema | 原样返回、1 次；无 Pydantic 自动校验 |
| F9：HTTP 400 | MockTransport 默认响应头、真实 SDK＋tenacity | 1 次 BadRequestError；is_retryable_exception=False |
| 既有文本／分发回归 | 原文件同步 SDK 4 组合、四 Provider 内部分发＋同步 wrapper、接口选择 | 纳入 48 项并通过 |

另一次直接运行原有 `test_async_sdk_parameters_and_text_chunks('deepseek', True)` 成功，输出 `ASYNC OK`。它验证的是原有合成文本流式场景，不能替代全部异步组合。

### 9.3 未通过、停滞和未运行记录

首轮临时同步集合为 **44 passed, 2 failed, 5 deselected**。两个失败源于审查测试错误地预期非法 JSON／空字符串必定触发 UnboundLocalError；源码中 clean_json 的结果已先赋值，因此实际进入正常回退。只修正 `/tmp` 中的测试预期，再加 `text=None` 边界、真实 planner 至 HTTP 用例，得到最终 48 项通过；未修改业务代码迎合测试。

包含原有异步 SDK wrapper 的批量 pytest 在前四个同步用例后停滞：有一次 50 秒 timeout（退出 124），另有手动中断（退出 130）；后续直接运行四组异步组合及 /general 的批量脚本也在 60 秒 timeout（退出 124），未产生通过记录。直接单项异步检查通过，但批量异步停滞根因未确定，**不计为产品缺陷，也不计为通过**。完整项目测试、真实 API、真实浏览器和正式模型配置核验均未执行；现有 online chat actor 测试会依赖聊天客户端／模型及数据库（如 `tests/test_online_chat_actors.py:603-631`），不满足本次离线隔离要求。

临时测试可在本环境用上方命令复现；`/tmp` 不保证长期保留。独立复现关键缺陷的方法是：以无真实凭据的 `openai.OpenAI(..., http_client=httpx.Client(transport=httpx.MockTransport(handler)))` 替换 `utils.get_openai_client`，清空仅测试进程内的客户端缓存；调用 `gpt.openai_send_message_to_model(..., response_type="json_object", response_schema=PickTools, provider_type="deepseek", api_base_url="https://api.deepseek.com")`；handler 只记录脱敏 body 并返回合成 400。应观察 `type=json_schema`、`stream=true` 及一次请求。禁止把 MockTransport 替换为真实传输后直接使用真实密钥运行。

### 9.4 后续修复验收必须覆盖

- DeepSeek 两个注册模型、别名、官方／代理 URL，默认问候和 /general；前置 fast 模型与最终模型不同的场景。
- json_object＋schema、仅 schema、仅 json_object、纯 text，以及 tools＋schema 共存的精确优先级。
- OpenAI 官方 Responses 和 Chat Completions／代理，Azure／DeepInfra 既有例外，Anthropic 与 Google 输出及分发；不能只测 DeepSeek 请求成功。
- 正确／非法／缺字段／空列表／错误类型 JSON、空字符串、None、截断、拒绝；本地 Pydantic 和选项过滤应明确处理。
- unsupported response_format 400 的有界降级，以及其他 400 不降级；429／5xx／超时保留合理重试；有 fallback 时验证尝试次数。
- WebSocket／HTTP 流式与非流式的错误信息、断开／中断保存；成功空答案不得伪装为正常完成。真实服务验证须另获授权。

## 10. 修复实施清单

- [P0] 在 `get_structured_output_support` 引入显式 Provider＋接口上下文，纠正 DeepSeek Chat Completions 的结构化输出能力；评估官方 host 兼容旧配置及旧 reasoner 例外。
- [P0] 在 `openai_send_message_to_model` 保证目标请求采用支持的 JSON object 策略，保留其它 Provider schema 和 tools 优先级；覆盖 schema-only 调用。
- [P0] 为普通默认命令补真实前置链＋Mock HTTP 回归；确认不再请求 json_schema，后续能继续生成回答。
- [P1] 在 `aget_data_sources_and_output_format` 加本地 JSON／PickTools 验证、空文本保护及类型一致的回退，避免未赋值 response 的异常日志。
- [P1] 增加 DeepSeek／OpenAI Chat／OpenAI Responses 对照和 Anthropic／Google 回归；将本次临时验证转为正式测试需后续实施授权。
- [P1] 如确有多版本服务需要，增加仅针对已识别格式错误的一次降级与计数，不对所有 400 重试或删除参数。
- [P2] 在 WebSocket control error 的 UI 分支展示适度的用户可见错误，并区分失败、完成与中断。
- [P2] 用 Mock 验证失败叠加断连／中断时的空 Assistant 保存策略，避免把中断占位当成功回复。
- [P2] 修正 Prompt 格式示意引号，改善解析／空响应日志脱敏。

本次任务在报告交付处结束，上述项目均未执行。

## 11. 参考资料与交付核对

### 11.1 本地源码索引

所有路径相对于 `/home/agent/projects/khoj`，行号对应审查工作树而非 GitHub master：

| 证据 | 位置 |
|---|---|
| 命令识别 | `src/khoj/routers/helpers.py:259-281` |
| PickTools 与前置请求 | `src/khoj/routers/helpers.py:366-480`，特别是 426-442 |
| 模型 wrapper／分发 | `src/khoj/routers/helpers.py:1581-1733` |
| 最终回答模型和分发 | `src/khoj/routers/helpers.py:1930-1993` |
| 前置执行／回退 | `src/khoj/routers/api_chat.py:941-1034` |
| 最终生成／正常保存 | `src/khoj/routers/api_chat.py:1392-1470` |
| 断连／中断保存 | `src/khoj/routers/api_chat.py:753-831` |
| WebSocket 错误与 HTTP 入口 | `src/khoj/routers/api_chat.py:1592-1761`，特别是 1727-1729 |
| 请求格式构造与接口选择 | `src/khoj/processor/conversation/openai/gpt.py:27-94` |
| 同步 completion／重试／空响应 | `src/khoj/processor/conversation/openai/utils.py:83-277` |
| 能力判断／接口与流式判断 | `src/khoj/processor/conversation/openai/utils.py:735-744,868-898` |
| Schema 清洗 | `src/khoj/processor/conversation/openai/utils.py:1299-1323` |
| 能力枚举／异常分类／clean_json | `src/khoj/processor/conversation/utils.py:1273-1277,125-163,957-959` |
| 客户端与命令枚举 | `src/khoj/utils/helpers.py:1141-1172,415-435` |
| 模型槽位与实际选择 | `src/khoj/database/adapters/__init__.py:1295-1370,1421-1483` |
| 前置识别 Prompt | `src/khoj/processor/conversation/prompts.py:691-758` |
| 前端错误显示 | `src/interface/web/app/chat/page.tsx:285-335,350-365` |
| 既有隔离 SDK 测试 | `tests/test_deepseek_provider.py:172-276,317-400,479-482` |
| pytest DB 约束 | `tests/conftest.py:42-44`、`pytest.ini` |

SDK 证据均来自已安装 `.venv/lib/python3.12/site-packages/openai/`：`resources/chat/completions/completions.py:1494-1500`、`lib/_parsing/_completions.py:233-246`、`_base_client.py:750-783`。SDK 版本在第 2 节记录。

### 11.2 外部资料

- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)：核对接口 `/chat/completions` 与 response_format 字段契约；没有把其他接口能力混用。
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)：核对 JSON object 模式、提示词配合与空内容边界。
- [Khoj Issue #1336](https://github.com/khoj-ai/khoj/issues/1336)：作为旧版本线索；与本地函数更名、枚举缺省及重试差异见第 6.2、7.1 节，不采用其结论代替本地验证。

### 11.3 交付时文件状态

报告生成前再次执行 `git status --short`，与第 2.1 节相同；707 个已有文件 SHA-256 比较无变化。报告生成后再次核对，预期仅新增 `reports/deepseek_response_format_root_cause_report.md`，原有修改和未跟踪文件全部保留。最终核对结果记录于本节下方。

最终核对：已有 707 个基线文件内容均未变化；无基线文件删除；Git tracked／非忽略 untracked 文件集合仅增加本报告。`git status --short` 新增 `?? reports/`，其余状态与开始时相同。本次没有发现额外意外改动。测试临时文件仅位于 `/tmp/khoj-response-format-audit/`。
