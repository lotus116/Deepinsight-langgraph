# 重构记录与后续计划

本文记录本轮围绕 `FastAPI + LangGraph + Chroma` 的重构工作，以及下一阶段计划。

## 已完成的修改

### 1. 新增核心服务层

新增 `deepinsight_core/`，把原来集中在 Streamlit 和 `Text2SQLAgent` 里的核心逻辑拆出来：

- `config.py`：统一配置读取，支持 typed settings。
- `services/sql_service.py`：只读 SQL 校验和执行。
- `services/sql_generation_service.py`：封装 SQL 生成与修复。
- `services/rag_service.py`：封装旧 RAG 适配、Chroma 索引、Chroma 粗排和桥接。
- `services/text2sql_graph_service.py`：构建新的 Graph Text2SQL 服务。
- `services/graph_agent_adapter.py`：让 Graph 流程兼容旧的流式事件协议。

### 2. 新增 LangGraph 流程

新增 `deepinsight_core/graph/`，把 Text2SQL 拆成可测试节点：

- 检索上下文：schema、few-shot、术语。
- 生成 SQL。
- 校验 SQL 是否只读安全。
- 执行 SQL。
- 对可修复错误进行一次 healing 重试。

当前主流程：

```text
retrieve_context -> generate_sql -> validate_and_execute_sql
                                       |
                                       v
                                  heal_sql -> validate_and_execute_sql
```

### 3. 新增 Chroma 可选接入

已实现 Chroma 相关能力，但默认关闭：

- `ChromaKnowledgeStore`
- `ChromaRAGIndexer`
- `ChromaRAGRoughSearcher`
- `ChromaRAGBridge`

当前策略是保守接入：只在 `use_chroma_rag=true` 时启用，并优先替换旧 RAG 的粗排阶段，不破坏原来的精排、术语匹配和依赖补全逻辑。

### 4. 新增 FastAPI 后端入口

新增 `deepinsight_api/`：

- `/health`
- `/v1/sessions`：创建后端会话。
- `/v1/runs/{run_id}`：查询一次运行的状态。
- `/v1/query/stream`：保留 legacy agent 流式接口。
- `/v1/query/graph/stream`：实验性 Graph 流式接口。
- `/v1/sql/execute`：只读 SQL 执行接口。
- `/v1/index/rebuild`：重建 Chroma 索引。

Graph API 当前需要配置：

```json
{
  "enable_graph_query_path": true
}
```

### 5. Streamlit 支持后端切换

Streamlit 配置面板已新增：

- `Agent Backend`：`legacy` 或 `graph`。
- `Query Runtime`：`api`、`auto` 或 `local`。
- `FastAPI URL`：默认 `http://127.0.0.1:8000`。
- `Use Chroma RAG`：默认关闭。

`get_agent()` 已支持按配置返回 FastAPI client、API 优先的 fallback adapter、legacy agent 或 graph agent adapter。

### 6. eval_suite 支持 Graph 后端

评测命令新增参数：

```powershell
--agent-backend legacy
--agent-backend graph
```

推荐 AdventureWorks 对照命令：

```powershell
conda activate deepinsight
$env:PYTHONIOENCODING='utf-8'
python -m eval_suite.run_all --accuracy-only --groups G2 G3 --database adventureworks --agent-backend legacy
python -m eval_suite.run_all --accuracy-only --groups G2 G3 --database adventureworks --agent-backend graph
```

必须使用 `G2 G3` 顺序，因为你的实验设计中 G3 是在 G2 结果基础上继续修复失败样本。

### 7. 配置与密钥改进

`utils.py` 已支持环境变量覆盖。仓库配置中的模型 API key 已清空，真实密钥应放在本机 `.env`：

- `DEEPINSIGHT_API_KEY`
- `DEEPINSIGHT_API_BASE`
- `DEEPINSIGHT_MODEL_NAME`
- `DEEPINSIGHT_RECOMMENDATION_API_KEY`
- `DEEPINSIGHT_AGENT_BACKEND`
- `DEEPINSIGHT_QUERY_RUNTIME`
- `DEEPINSIGHT_API_SERVER_URL`
- `DEEPINSIGHT_USE_CHROMA_RAG`

已新增：

- `.env.example`
- `.gitignore` 对 `.env` 的忽略规则

## 当前验证结果

新增测试文件：

- `tests/test_refactor_foundation.py`

覆盖范围：

- 配置默认值和环境变量覆盖。
- SQL 只读安全校验。
- Chroma 索引和粗排。
- LangGraph 检索、生成、执行、修复流程。
- FastAPI 路由保护。
- GraphAgentAdapter 对旧事件协议的兼容。
- eval_suite Graph 后端接入。

最近一次本地验证：

```powershell
python -m unittest tests.test_refactor_foundation
```

结果：`34` 个测试通过。

你提供的 AdventureWorks 评测结果显示：

```text
改前 G3 Overall: 76.0%, Avg Tokens: 2792
改后 G3 Overall: 80.0%, Avg Tokens: 2267
```

当前结论：Graph 后端在 G3 上准确率提升约 `+4.0` 个百分点，平均 token 降低约 `18.8%`。

## 接下来计划

### 阶段 1：前后端解耦

状态：已完成第一版。

目标：让 Streamlit 优先通过 FastAPI 调用后端。

已完成：

1. 固化 `/v1/query/graph/stream` 的 SSE 事件格式。
2. 新增 `DeepInsightAPIClient`。
3. Streamlit 支持通过 HTTP SSE 调用 FastAPI。
4. 保留 `local` 和 `auto` 模式，便于安全回退。

### 阶段 2：FastAPI 服务化

状态：已完成第一版。

目标：让后端可以独立启动、部署和测试。

已完成：

1. 增加会话或 run 管理接口。
2. 增加运行状态查询接口。
3. 增加 Chroma 索引重建接口。
4. 统一错误响应结构。
5. 补充后端启动命令和最小 README。

### 阶段 3：Chroma 正式评测

目标：判断 Chroma 是否适合作为默认 RAG 能力。

计划：

1. 分别评测 `use_chroma_rag=false` 和 `true`。
2. 对比 G2/G3 准确率、token 和运行耗时。
3. 只有在不降低准确率且带来性能收益时，才考虑默认启用。

### 阶段 4：失败样本优化

目标：继续提升 AdventureWorks 的 Hard 和 Extra-Hard。

计划：

1. 分析失败 case。
2. 按表选择、字段选择、JOIN、聚合、时间条件、SQL 方言分类。
3. 针对失败类型优化 prompt、RAG schema、术语词典和 healing prompt。

### 阶段 5：配置安全收口

状态：已完成基础收口。

目标：避免 API key 明文保存在仓库配置里。

已完成：

1. 新增 `.env.example`。
2. 清空仓库配置中的模型 API key。
3. `data/config.json` 只保留非敏感默认配置。
