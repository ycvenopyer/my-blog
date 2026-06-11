# 基于 LangGraph 的销售智能体

B2B 销售日常涉及客户跟进、商机推进、沟通复盘、竞品应对与各类文案产出，信息分散在 CRM、沟通录音 ASR、评估报告与知识库中。一线销售难以在拜访间隙快速拉齐数据、生成材料或获得可执行建议。

本项目是面向一线销售代表的 AI 工作助手。基于 LangGraph 状态图与 Agentickit 企业级框架，实现 Skill 路由、Tool Calling、Text-to-SQL 与多层 Memory，通过 A2A 协议对接前端流式对话，并支持 Agent 间标准化协作通信。

---

## 1. 背景与目标

### 1.1 问题背景

慧销平台已沉淀大量销售过程数据：客户画像、商机阶段、沟通 ASR 转写、能力/任务评估、竞品提及、成交模型维度等。一线销售常见痛点包括：

- 查数难：“这周跟进了哪些高风险客户”、“某商机最近三次沟通的任务完成率”需要跨多表关联，销售不会写 SQL；
  
- 写材料慢：日报、客户邮件、方案摘要、会议纪要占用大量非销售时间；
  
- 上下文断裂：跨会话无法记住用户偏好与历史结论；
  
- 能力泛化：同一 Agent 需同时承担数据查询、文案写作、竞品情报、可视化报告等差异很大的任务，单一 Prompt 难以兼顾。

### 1.2 核心目标

- 搭建基于 LangGraph 的流式、可中断恢复的销售对话 Agent 后端；
  
- 采用 Skill + Tool Calling 架构，在单 ReAct 循环内按任务类型动态注入专业能力；
  
- 实现 Text-to-SQL 工具链，让销售用自然语言查询业务库（客户、商机、沟通、评估等）；
  
- 对接外部 Memory System，支持跨会话记忆检索、沟通原文与组织知识库 RAG；
  
- 参考 `data-analyst` 项目的 Multi-Agent 设计经验，完成向 Skill 架构的迁移与工具复用。

### 1.3 迭代概览

| 迭代 | 主题 | 主要产出 |
|------|------|----------|
| 需求与架构 | PRD 与 Prompt/Tool 设计 | System Prompt 架构、四类 Skill 文档、工具规格、数据库表说明 |
| 参考实现 | `data-analyst` 子 Agent 架构 | Text-to-SQL / 数据分析 / Artifact 三个 LangGraph 子 Agent |
| 工程落地 | `coach-askai` 后端 | Skill 版 ReAct Agent、A2A 流式服务、PostgreSQL Checkpoint、Memory System 集成 |
| 平台化 | Prompt 与 Skill 托管 | Langfuse 全链路监控，动态编译 System Prompt，工具消息配置 |

---

## 2. 系统架构

### 2.1 总体架构

系统采用前后端分离 + Agent 服务独立部署：前端通过 A2A 协议与 `coach-askai` 通信，Agent 内部为 LangGraph 编译的 ReAct 状态图。核心循环为 `before_model → model ↔ tools`，Middleware 负责工具链管理、状态注入、Skill Prompt 动态编译与任务取消检测。

```
┌─────────────────────────────────────────────────────────────────┐
│                     慧销前端（Vue / 移动端）                     │
│              对话 UI · 产物预览 · 流式 Tool 状态展示              │
└────────────────────────────┬────────────────────────────────────┘
                             │  A2A Protocol (SSE 流式)
┌────────────────────────────▼────────────────────────────────────┐
│              coach-askai（FastAPI + Agentickit）                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ AskSalescoachHandler → create_ask_salescoach_agent       │   │
│  │   ReAct Loop: before_model → model ↔ tools               │   │
│  │   Middleware: 任务取消 · 消息清理 · Skill Prompt · 选工具  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │              │                │              │        │
│         ▼              ▼                ▼              ▼        │
│   Langfuse        PostgreSQL         MySQL          Memory      │
│   全链路监控       Checkpoint         只读           System API  │
│   Prompt/Skill    + A2A Task Store   业务数据       记忆/知识库  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层次 | 技术选型 | 说明 |
|------|----------|------|
| Agent 框架 | LangGraph 状态图 + LangChain ReAct | 自定义 `AgentStateSchema` |
| 大模型 | Qwen3-max（内部 LLM Gateway） | 流式输出，`temperature=0` |
| 服务协议 | A2A + FastAPI | 流式对话、Agent 间协作通信、中断恢复 |
| 对话状态 | PostgreSQL | 多轮消息与业务状态持久化 |
| 业务数据 | MySQL | Text-to-SQL 查询目标库 |
| 长期记忆 | Memory System HTTP API | 相似记忆检索、沟通原文、组织知识库 RAG |
| 缓存 | Redis | 用户 Profile 缓存 |
| 可观测性 | Langfuse | Prompt/Skill 托管、LLM 与 Tool 全链路 Tracing |
| SQL 安全 | sqlglot | AST 级 SELECT-only、LIMIT、JOIN 规则校验 |

---

## 3. Skill 体系与 Multi-Agent 设计

### 3.1 Skill 路由机制

PRD 定义四类 Skill（托管于 Langfuse），通过 `select_skill` 支持多技能场景切换：

| Skill | 职责 | 典型场景 |
|-------|------|----------|
| `data-query` | 业务数据 Text-to-SQL、沟通内容分析 | 查客户/商机/业绩、统计高风险客户、客户需求洞察 |
| `sales-writing` | 销售文案生成 | 智能话术、日报周报、客户邮件、方案摘要 |
| `market-intelligence` | 竞品与行业情报 | 实时竞品分析、差异化定位、异议应对 |
| `visualization` | HTML 可视化产物 | 看板、图表、汇报页面 |

运行时，LLM 调用 `select_skill` 更新 `activated_skill`，Middleware 将对应 Skill 正文注入 System Prompt 并过滤工具集，以 Prompt 编排替代子 Agent 嵌套。

### 3.2 架构演进：从 data-analyst 到 coach-askai

`data-analyst` 是管理者侧 SalesCommander 的参考实现，采用主 Agent + 三个子 Agent：

| 子 Agent | 入口工具 | 内部工具链 |
|------|----------|------------|
| Text-to-SQL | `data_query` | `get_table_schema` → `generate_sql` → `modify_sql` → `execute_sql` → `return_result` |
| 数据分析 | `data_analysis` | `generate_script` → `execute_script` → `return_result` |
| Artifact | `create_artifact` | `generate_artifact_code` → `push_to_frontend` → `return_result` |

`coach-askai` 复用了 SQL 工具实现、状态注入模式（`StatefulToolNode` / `get_state_context()`）、SQL 校验工具与 Memory 客户端，但将子 Agent 扁平化为顶层 Tool，并用 Skill 替代子 Agent 的角色边界。核心操作工具（`create` / `view` / `str_replace` / `push_to_front`）对应原 Artifact 子 Agent 能力，供 `sales-writing` 与 `visualization` Skill 使用。

---

## 4. Tool Calling 与状态管理

### 4.1 工具分类

| 类别 | 工具 | 作用 |
|------|------|------|
| 路由 | `select_skill` | 切换激活 Skill，写回 `activated_skill` |
| 核心操作 | `create`, `view`, `str_replace`, `push_to_front` | 创建/编辑/预览/推送 Markdown/HTML/Python 产物至前端 |
| 数据查询 | `get_table_schema`, `generate_sql`, `modify_sql`, `execute_sql`, `get_conversation_content` | Text-to-SQL 全链路与单次沟通 ASR 原文 |
| 信息检索 | `search_memory`, `search_knowledge_base`, `web_search` | 跨会话记忆、组织知识库 RAG、联网搜索 |

### 4.2 状态管理机制

设计 `AgentStateSchema` 统一管理 Agent 运行时状态，核心包括：

- 上下文追踪：LangGraph Checkpoint 持久化多轮 messages 与 `activated_skill`；
  
- 对象管理：`objects` 存储 HTML/Markdown 等产物，`object_id` 引用避免上下文膨胀；
  
- SQL 查询结果缓存：`sql_queries` 按 `sql_id` 缓存 SQL 文本、行数据与预览摘要；
  
- 工具执行状态：`last_tool_execution` 供流式 UI 展示。

工具通过 Middleware 注入 `InjectedState` + ContextVar（`set_state_context`）访问状态，LLM 侧仅保留 ID 与摘要，完整内容通过 `view` 分页读取，提升响应速度与对话连贯性。

### 4.3 动态工具加载

`get_system_tools(state)` 根据用户 `org_id` 从 `tool_manager` 动态挂载 `search_knowledge_base`，实现按组织隔离的知识库 RAG，与 Memory System 的 `knowledge_space_list` / `search_knowledge_docs` 对接。

---

## 5. Text-to-SQL 工具链

标准工作流：

```
get_table_schema(table_names) → generate_sql → (modify_sql)* → execute_sql → view(sql_id)
```

- `get_table_schema`：动态拉取 MySQL 表结构、主键、索引与 3 行样例；结合 Langfuse `deprecated_columns` 过滤废弃字段；
  
- `generate_sql` / `modify_sql`：将 SELECT 语句存入 `state.sql_queries`，支持字符串级增量修改；
  
- `execute_sql`：只读连接池执行，结果写回 state，返回预览；
  
- `get_conversation_content`：通过 Memory System 拉取指定 `conference_id` 的完整 ASR 转写，用于单次沟通深度分析。

执行前由 `validate_sql_utils.py` 基于 sqlglot 做 AST 校验（SELECT-only、强制 LIMIT、JOIN 规则等），配合只读 Session 构成纵深防御。

---

## 6. Memory 体系

Agent 记忆分为三层：

| 层级 | 存储 | 职责 |
|------|------|------|
| 短期 | LangGraph Checkpoint（PostgreSQL） | 当前会话消息、Skill 状态、SQL/产物 ID 引用 |
| 任务 | A2A Task Store（PostgreSQL） | 任务生命周期、取消/中断恢复 |
| 长期 | Memory System API | 跨会话相似记忆、用户 Profile、沟通原文、组织知识库 |

Memory System 主要接口（`memory_system_client.py`）：

- `POST api/v1/memories/search-similar` — `search_memory` 关键词检索历史结论；
  
- `POST api/v1/memories/get-original-text` — 沟通 ASR 原文；
  
- `POST api/v1/knowledge/search_knowledge_docs` — 组织知识 RAG；
  
- `GET api/v1/profiles/{user_id}/{org_id}` — 用户画像（Redis 缓存）。

Middleware 在每次 LLM 调用前检测 A2A 任务是否已取消，并清理不完整的 tool_calls，避免 Checkpoint 脏数据导致后续调用失败。

---

## 7. Prompt 工程

### 7.1 System Prompt 结构

主导 Agent 产品需求分析，设计完整的 System Prompt 架构（PRD 采用 XML 分段）：

- Persona：智能销售搭档人设；
  
- Core Principles：销售优先、结论先行、数据说话、用户决策；
  
- Tool Usage：各工具的调用规范与禁止事项；
  
- Skills：动态编译的 Skill 目录与激活后注入的 `<skill.body>`。

### 7.2 Langfuse 集中托管

- System Prompt、开场白、Skill 正文、工具流式/静态 UI 文案均版本化管理（`production` label）；
  
- `tool_message_configs` 配置各 Tool 在前端的展示标题与进度文案；
  
- `deprecated_columns` 按表维护废弃字段，减少 LLM 生成无效 SQL；
  
- Langfuse Tracing 覆盖 LLM 与 Tool 调用，构建可观测的 Agent 基础设施。

---

## 8. 流式服务与 API

服务通过 Agentickit 启动，默认端口 10000，核心能力：

| 能力 | 说明 |
|------|------|
| A2A 标准路由 | 任务发送、流式响应、Agent 间协作通信 |
| `/serverstatus` | 健康检查 |
| `/resume` | 中断后恢复 LangGraph 执行 |
| `/edit_message` | 编辑历史消息后重新运行 |

`AskSalescoachHandler` 负责将 LangGraph `astream_events` 转为前端 SSE 事件：文本增量、Tool 执行状态、产物 `object` 推送等。

---

## 9. 个人工作与收获

### 9.1 主要负责

1. 主导 Agent 产品需求分析，设计完整 System Prompt 架构，开发 Tools 与 Skills，支持多技能场景切换；
   
2. 基于 LangGraph 状态图与 Agentickit 企业级框架构建销售 Agent，使用 A2A 协议实现 Agent 间协作通信；
   
3. 设计状态管理机制，实现上下文追踪、对象管理、SQL 查询结果缓存等核心能力；
   
4. 采用 Middleware 模式实现工具链管理与状态注入，集成 Langfuse 全链路监控；
   
5. 落地智能话术生成、实时竞品分析、客户需求洞察等能力，赋能一线销售日常作业。

### 9.2 技术收获

1. Agent 架构选型：对比 Multi-Agent 与 Skill 设计两种 Agent 架构；
   
2. Tool Calling 工程化：Stateful Tool、ContextVar 注入、大结果 ID 引用，避免上下文爆炸；
   
3. Text-to-SQL 落地：业务表 schema 文档化、sqlglot 安全校验、2B/2C 领域规则注入 Skill；
   
4. 可观测性：Langfuse 全链路 Tracing + Middleware 可靠性保障（任务取消、脏消息清理）。

---

## 10. 总结与展望

Salescoach Agent 将 LangGraph ReAct、Skill 路由、Tool Calling、Text-to-SQL 与多层 Memory 整合为面向一线销售的对话式 AI 助手，嵌入慧销平台的客户—商机—沟通—评估数据闭环。Agent 已支持智能话术生成、实时竞品分析、客户需求洞察等场景，Skill 版架构在保持专业能力边界的同时简化了运行时拓扑，更适合高频、低延迟的销售日常问答。

---

??? note "写入简历"
    > 项目：基于 LangGraph 的销售智能体

    > 角色：Agent 实习生

    > 技术栈：Python / LangGraph / Agentickit / LangChain / Qwen3-max / Text-to-SQL / A2A / PostgreSQL / MySQL / Langfuse / Redis

    S（情境）

    慧销沉淀了客户、商机、沟通 ASR、销售评估等大量业务数据，一线销售在拜访间隙难以快速查数、写材料和获取可执行建议；同时 Agent 需覆盖数据查询、文案写作、竞品情报、可视化等多种能力，单一 Prompt 难以兼顾。

    T（任务）

    主导后端 Agent 需求分析与架构设计：基于 LangGraph 状态图与 Agentickit 构建 Salescoach Agent，实现 Skill 路由、Tool Calling、Text-to-SQL、多层 Memory 与 A2A 协作通信。

    A（行动）

    - 设计 System Prompt 架构与四类 Skill，开发 Tools 并支持 `select_skill` 多技能场景切换；
  
    - 设计状态管理机制，实现上下文追踪、对象管理、SQL 查询结果缓存，采用 Middleware 完成工具链管理与状态注入；
  
    - 搭建 Text-to-SQL 工具链（sqlglot AST 校验 + 只读 MySQL），梳理业务表 schema 与 2B/2C 查询规则；
  
    - 集成 Langfuse 全链路监控与 Memory System，构建可观测 Agent 基础设施；
  
    - 对接 A2A 流式 Handler，支持产物推送、任务取消与中断恢复。

    R（结果）

    - 交付可流式交互的 Salescoach Agent 后端，支持自然语言查数、智能话术生成、实时竞品分析与客户需求洞察；
  
    - 形成 Skill + Tool Calling 可扩展架构，Skill 与 Prompt 可独立迭代，响应速度与对话连贯性明显提升；
  
    - 沉淀 Text-to-SQL 安全校验与业务表文档化规范，为 SalesCommander 及教练 Multi-Agent 产品线提供可复用工具层。

