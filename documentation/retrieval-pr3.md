# PR-3：RRF、Hybrid Retrieval 与 Chat RAG 实施报告

日期：2026-10-29。工作分支：`feature/three-tier-intent-recognition`。

本次在包含未提交 PR-1/PR-2 和其他修改的现有工作区中实施；未重置分支、提交或覆盖其他任务。开始时的 diff 和状态分别保存在 `/tmp/khoj-pr3/baseline.diff` 与 `baseline-status.txt`。

## A. 完成情况

| 子任务 | 状态 | 主要文件 | 验证 |
|---|---|---|---|
| 3.1 RRF | 完成 | `search_type/rank_fusion.py` | 19 项独立测试，无数据库/模型依赖 |
| 3.2 Hybrid 编排 | 完成 | `search_type/hybrid_search.py` | 分支故障测试、真实 ParadeDB/pgvector 集成 |
| 3.3 execute_search | 完成 | `routers/helpers.py`、`search_type/text_search.py` | 三模式、余弦分数、阈值、精排、去重、数量限制 |
| 3.4 API mode | 完成 | `routers/api.py` | enum 校验、默认/覆盖、503、Hybrid 降级 |
| 3.5 Chat RAG | 完成 | `routers/helpers.py` | 多查询、references、真实模型消息构建、POST/WebSocket |

源文件均位于 `src/khoj/`。最终测试统计见 E。

## B. 修改文件与接口

新增：

- `src/khoj/search_type/rank_fusion.py`：纯函数 `reciprocal_rank_fusion()`，以 Entry PK 融合，不依赖数据库或模型。
- `src/khoj/search_type/hybrid_search.py`：`query()` 独立执行两路召回；`RetrievalUnavailable` 表示无法满足输出契约。
- `src/khoj/search_type/retrieval_config.py`：统一模式解析，以及自然语言查询与过滤指令的提取。
- `tests/test_rank_fusion.py`、`test_hybrid_search.py`、`test_hybrid_retrieval.py`、`test_retrieval_modes_api.py`、`test_hybrid_chat.py`、`test_hybrid_integration.py`：数学、编排、执行、API、聊天及真实数据库测试。
- 本报告。

修改：

- `src/khoj/search_type/retrieval_types.py`：复用 PR-1 的 `RetrievalHit`，新增集中配置 `HybridOptions`；保留旧 `RetrievalOptions` 的默认值 10。
- `src/khoj/search_type/text_search.py`：ORM 构造和求值都在同一线程敏感 worker 中执行；推理移出事件循环；新增 `hydrate_retrieval_hits()`、`collate_retrieval_hits()`；精排支持 RRF/BM25 次级排序；跨查询去重保留分数和查询来源。
- `src/khoj/routers/helpers.py`：`execute_search()` 增加末尾可选模式；分派三种检索；`search_documents()` 每条 inferredQuery 使用统一模式，保留过滤条件和查询顺序；关闭无可靠失效机制的检索结果缓存。
- `src/khoj/routers/api.py`：搜索路由支持 `mode`，非法模式返回 422，后端无法满足结果契约时返回 503。
- `tests/test_agents.py`：旧的空用户私有 Agent 测试改为显式授权身份，并断言无身份时拒绝访问。
- `src/khoj/database/adapters/__init__.py`：BM25 增加内部 `authorization_user`，区分授权用户和 Agent-only 文档范围；Dense 同距候选按 PK 稳定排序。

`SearchResponse` 类型及历史字段保持不变；BM25/Hybrid 的独立分数和排名放在可选的 `additional` 字典内，`score` 始终是真实 cosine distance。

## C. 实际检索链路

搜索 API：

`api.search()` → `execute_search()` → `resolve_retrieval_mode()`。

- Vector：`text_search.query()` → `EntryAdapters.search_with_embeddings()` → `apply_filters()` → 距离排序与截断。
- BM25：`EntryAdapters.search_with_bm25()` → `apply_filters()` → `ranked_bm25_entries()`（真实 `|||` 与 `pdb.score`）→ 独立排序与截断。
- Hybrid：`hybrid_search.query()` → 上述独立两路 → `reciprocal_rank_fusion()` → Top-40。

BM25/Hybrid 候选随后经 `hydrate_retrieval_hits()` 再次按原范围加载正文，用数据库 `CosineDistance` 计算真实响应距离；此处不应用 Dense 阈值。`collate_retrieval_hits()` → `rerank_and_sort_results()` → 现有 `cross_encoder_score()` → 最终去重 → `n`。

Hybrid 先按 RRF 降序、PK 升序排序；启用 Cross-Encoder 时，再按现有 `cross_score = 1 - prediction` 升序稳定排序。因此次级顺序仍为 RRF、PK。BM25 的次级顺序为原 BM25 分数、PK。Vector 保留原距离排序和现有精排行为。

聊天：POST `/api/chat` 与 WebSocket `/api/chat/ws` 共同进入 `event_generator()` → `search_documents()` → 对每条 inferredQuery 调用 `execute_search()` → `deduplicated_search_responses()` → `compiled_references`。同一 references 进入事件 `references.data.context`，并传给 `agenerate_chat_response()` → `build_conversation_context()` → `generate_chatml_messages_with_context()` → 现有模型 provider。

聊天测试在 provider 边界模拟输出，断言实际传给 provider 的消息含 BM25 独有证据；真实数据库测试另证明该证据由原生 BM25 独立召回。未更改 Prompt、Token Budget、工具选择或在线搜索逻辑。

## D. 参数与模式

| 参数 | 默认值 |
|---|---:|
| Hybrid Dense Top-K | 40 |
| BM25 Top-K | 40 |
| RRF k | 60 |
| Dense / BM25 权重 | 1.0 / 1.0 |
| Fusion Top-K | 40 |
| Final Top-N | 现有 n |
| 默认检索模式 | vector |

`HybridOptions` 统一管理召回与融合参数。为兼容历史行为，Vector 仍使用 `max(10, n)` 召回。

RRF 排名使用输入列表的 1-based 位置；同分按整数 PK 升序。同一路重复 PK 仅第一次计分，后续位置不压缩；不修改输入对象，不混加原始分数。

`KHOJ_RETRIEVAL_MODE=vector|bm25|hybrid` 控制服务器默认值。API 显式 `mode` 优先。无效服务器配置抛出清晰错误；不静默切换模式。聊天不需新增客户端参数。

## E. 测试报告

当前最终核心结果：**163 passed / 0 failed / 0 errors / 1 skipped**，报告 `/tmp/khoj-pr3/core-final.xml`。跳过项 `test_missing_extension_real_database` 需要无 pg_search 环境，另在 catalog 测试库单独验证。

| 检查 | 结果 | 证据 |
|---|---|---|
| RRF 独立运行 | 19 passed | `rrf.xml`；最后一次独立执行 0.07 秒 |
| 核心 PR-1/2/3 检索及真实 BM25 | 163 passed / 1 skipped | `core-final.xml` |
| 管理命令目录/索引冲突 | 50 passed | `catalog.xml` |
| Agent 本地索引/检索最终回归 | 10 passed | `agents-final.xml` |
| 缺少 pg_search 的真实错误契约 | 1 passed | `no-extension.xml` |
| PR-2 全新库索引生命周期 | 1 passed | 本次执行输出 |
| 授权修复针对性测试（较早一轮） | 23 passed | `review-fixes.xml` |
| 扩大搜索/用户/聊天/Agent 回归（较早一轮） | 213 passed / 3 skipped / 8 fixture errors | `final-targeted.xml`；8 项为缺少在线 API key 的旧 fixture |
| 全量尝试，遇首个失败停止 | 261 passed / 3 skipped / 1 failed | `full.xml`；`test_helpers.py::test_reading_webpage` 连接 Wikipedia SSL 超时 |
| Ruff / diff 空白检查 | 通过 | 最终命令输出 |

上述测试集有交集，不能把通过数量相加当作唯一测试总数。无扩展测试另在 catalog 库执行，1 passed（`no-extension.xml`）。Agent 使用离线占位配置重跑后 **10 passed / 0 failed / 0 errors**（`agents-final.xml`），先前 8 项 fixture 错误均不再出现。全量运行在第一个外部网络失败后停止，剩余测试未执行；未运行付费在线 LLM 质量评测。

实际执行的命令（从仓库根目录）：

```bash
PYTHONPATH=src .venv/bin/pytest --noconftest -c /dev/null tests/test_rank_fusion.py -q

.venv/bin/python /tmp/khoj-pr3/run.py --bm25 .venv/bin/pytest \
  tests/test_rank_fusion.py tests/test_hybrid_search.py tests/test_hybrid_retrieval.py \
  tests/test_retrieval_modes_api.py tests/test_hybrid_chat.py tests/test_hybrid_integration.py \
  tests/test_retrieval.py tests/test_bm25_retriever.py \
  tests/test_bm25_retriever_integration.py tests/test_bm25_index_updates.py \
  -q --tb=short --junitxml=/tmp/khoj-pr3/core-final.xml

.venv/bin/python /tmp/khoj-pr3/run.py --catalog .venv/bin/pytest \
  tests/test_bm25_index_command.py -q --tb=short --junitxml=/tmp/khoj-pr3/catalog.xml

.venv/bin/python /tmp/khoj-pr3/run.py --offline-api .venv/bin/pytest \
  tests/test_agents.py -q --tb=short --junitxml=/tmp/khoj-pr3/agents-final.xml

.venv/bin/python /tmp/khoj-pr3/run.py --catalog timeout --signal=INT --kill-after=20s 180 \
  .venv/bin/pytest -q --tb=short --maxfail=1 --junitxml=/tmp/khoj-pr3/full.xml

.venv/bin/ruff check src/khoj/search_type/rank_fusion.py \
  src/khoj/search_type/hybrid_search.py src/khoj/search_type/retrieval_config.py \
  src/khoj/search_type/retrieval_types.py src/khoj/search_type/text_search.py \
  src/khoj/routers/helpers.py src/khoj/routers/api.py src/khoj/database/adapters/__init__.py \
  tests/test_rank_fusion.py tests/test_hybrid_search.py tests/test_hybrid_retrieval.py \
  tests/test_retrieval_modes_api.py tests/test_hybrid_chat.py tests/test_hybrid_integration.py \
  tests/test_agents.py --output-format concise

git diff --check
```

`run.py` 明确设置 localhost:15432、`POSTGRES_DB=postgres` 为测试创建连接，以及单独 settings 内的 `DATABASES["default"]["TEST"]["NAME"]`。Django 实际执行测试使用下面的 test_ 数据库，不在 postgres 或业务库清理数据。它也启用模型离线缓存与 CPU 线程上限；`--bm25` 仅开启 opt-in 测试；`--offline-api` 仅供不调用 provider 的本地 Agent 回归建立占位配置，未设置启用在线质量测试的 `KHOJ_TEST_CHAT_PROVIDER`。

测试使用两套明确隔离的数据库：

- `test_khoj_pr3_20261029`：真实 pg_search/pgvector 与检索回归。
- `test_khoj_pr3_catalog_20261029`：要求不存在同名 BM25 索引的历史目录冲突测试。

PR-2 前置验证首次使用已有测试库时，41 passed / 1 failed / 1 skipped；失败为生命周期测试要求空库，但复用库已有索引。换到新的隔离库、仅在其中启用扩展后，生命周期测试 1 passed。此后合并运行又发现历史 B-tree 冲突测试与原生索引共库冲突；该文件分库运行后 50 passed。这两项环境前提均未通过删除已有索引来处理。

原生测试沿用 `KHOJ_TEST_BM25=1` 和 `tests.bm25_utils.ensure_bm25_index()`，后者强制数据库名以 `test_` 开头。业务库、Docker Volume 和已有业务索引未修改。

## F. 兼容性与安全

- 不传 mode 且未设置环境变量时仍执行 Vector；保留旧参数顺序。
- `SearchResponse.score` 不写入 BM25、RRF 或 Cross-Encoder 分数。BM25-only 文档即使余弦距离超过 Dense 阈值也可以进入结果。
- 两路均在用户/Agent、文件、日期及正负关键词过滤后截断；加载正文时再次限制范围。
- Agent 授权按数据库当前状态检查。内部 Agent-only 搜索使用 `authorization_user` 保留认证身份，同时 `user=None` 继续只允许 Agent 文档；不会因此加入用户个人文档。
- 无授权身份的 `execute_search(user=None, agent=private_agent)` 现在拒绝检索。合法内部调用须提供 `authorization_user`；不能沿用过去把空用户当作已授权的假设。
- 所有融合 chunk 先精排，随后才执行原有 corpus/hash 去重；RRF 阶段只用 Entry PK。
- 删除了 execute_search 的结果缓存读写。文档更新、删除、权限或模式切换不再复用旧检索响应；未引入新缓存系统。
- 空分支是正常结果；Hybrid 单分支后端故障记录明确告警后使用成功分支；两路失败报错。纯 BM25 不会伪装成 Vector。
- 查询 Embedding 不可用、候选无法计算有限余弦距离时明确报错。纯 BM25 虽不进行 Dense 召回，仍需查询 Embedding 来维持非空 float score 的历史契约。

## G. 风险与遗留问题

- P0：本次审查发现的权限问题均有失败复现与修复测试；最终回归结论见 E。
- P1：关闭结果缓存增加重复请求成本；两路采用线程敏感串行执行，避免共享连接/惰性 QuerySet 跨线程。尚未进行负载测试或质量对比，不能据此宣称 Hybrid 优于 Vector。
- P1：维持历史响应契约意味着纯 BM25 仍依赖 Embedding 可用性；缺少合法向量时不会生成伪造分数。
- P1：全量测试被 Wikipedia SSL 连接失败中断，不能宣称完整测试集全绿；在线服务不可达的问题未修改。
- P2：既有 Starlette/Pydantic 弃用警告仍存在；在线付费 LLM 质量测试不属于本次离线验证。
- 参数调优、Recall/MRR/nDCG 与 P50/P95 结果留给后续同数据集评估。

日志可观察 Dense/BM25/RRF 耗时、分支数量、重叠数、唯一候选数、Fusion 数量、Cross-Encoder 耗时、总耗时及最终数量；不记录文档正文。Cross-Encoder 未启用时无推理耗时事件。

实施判断与代价：

| 判断 | 原因 | 代价或局限 |
|---|---|---|
| 在原工作区增量修改，不新建独立分支/提交 | 前置 PR-1/2 尚未提交，且必须复用当前真实源码 | 交付仍是未提交 diff，不能独立从 HEAD 复现旧前置工作 |
| 保持 mandatory float cosine score | 避免历史字段语义或类型变化 | 纯 BM25 仍需 Embedding；不可计算时返回明确错误 |
| 禁用检索结果缓存 | 旧缓存无文档与授权版本失效机制 | 重复查询增加开销 |
| ORM 线程敏感串行执行 | 不在线程间传递惰性查询或共享连接 | 两路数据库召回耗时相加，需要后续负载评估 |
| 历史索引测试分库 | 冲突/生命周期测试与已有原生索引前提不兼容 | 测试命令需要区分数据库，不能随意合并 |

独立审查发现的 P1 私有 Agent 身份丢失和 P2 过滤查询绕过编排授权均已修复，测试记录见 `review-fixes.xml` 和最终核心 suite。没有延期处理的审查 minor。

## H. 使用与复现实验

沿用当前项目的启动参数，只增加环境变量：

```bash
KHOJ_RETRIEVAL_MODE=vector .venv/bin/khoj
KHOJ_RETRIEVAL_MODE=bm25 .venv/bin/khoj
KHOJ_RETRIEVAL_MODE=hybrid .venv/bin/khoj
```

例如：

```bash
curl -G 'http://localhost:42110/api/search' \
  -H "Authorization: Bearer $KHOJ_API_KEY" \
  --data-urlencode 'q=关键词 file:"*.pdf"' \
  --data 'mode=hybrid' --data 'n=5' --data 'r=true'
```

在 Hybrid 启动的服务器中正常聊天，无须传 mode。观察 `Retrieval mode=hybrid`（或明确降级日志），并检查 references.context。切回 `KHOJ_RETRIEVAL_MODE=vector` 后重启原服务即可回滚模式；无需修改数据库。

纯函数测试无需数据库：

```bash
PYTHONPATH=src .venv/bin/pytest --noconftest -c /dev/null tests/test_rank_fusion.py -q
```

其他测试沿用仓库 pytest 配置，必须指向隔离测试数据库；先按 `documentation/docs/contributing/bm25-index.md` 配置测试扩展，再设置 `KHOJ_TEST_BM25=1`。本次使用的本机 runner、测试 settings 与 JUnit 报告位于 `/tmp/khoj-pr3/`，具体命令与结果见 E。索引生命周期、同名索引冲突和原生检索测试应按上述数据库前提分开运行。

后续 frames_ir 对照实验无须修改评测框架。分别用三种服务器默认模式启动相同服务，固定文档快照、Embedding/Cross-Encoder、账号/Agent、抽样和随机种子，然后运行：

```bash
KHOJ_URL=http://localhost:42110 KHOJ_MODE=notes \
DATASET_SEED=42 SAMPLE_SIZE=100 RANDOMIZE=true \
.venv/bin/python tests/evals/eval.py --dataset frames_ir --output frames_ir_vector.csv
```

另两次保持参数相同，仅切服务器检索模式和输出文件名（bm25/hybrid）。评测框架会使用已有认证/LLM 配置，可能产生在线费用；本次没有执行该质量实验。用同一批请求的检索日志整理延迟分布，再结合标注评估召回与证据命中。
