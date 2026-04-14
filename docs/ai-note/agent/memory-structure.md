# Agent 记忆框架

传统 LLM 是无状态的，每次交互相互独立，无法跨会话学习和适应。Agent 记忆框架通过引入持久化的记忆层，使 Agent 能够记住用户偏好、历史经验和领域知识，从而在多次交互中不断进化和个性化。

一个完整的 Agent 记忆体系通常包含以下层次：

- **短期记忆（Short-term Memory）**：维护当前会话的上下文，如对话历史、消息缓冲区

- **长期记忆（Long-term Memory）**：跨会话持久化的知识，包括语义记忆（事实与偏好）、情景记忆（历史经验）和程序记忆（行为模式）

- **外部存储（External Storage）**：向量数据库、图数据库等，用于存储海量记忆并通过检索召回

## 1.Mem0

Mem0 是一个专为 AI Agent 设计的智能记忆层，其核心理念是"AI 记忆不应只是简单的存储，而应像人类记忆一样具备筛选、遗忘和整合能力"。

### 1.1 核心架构

Mem0 的记忆管理包含四个关键机制：

**智能过滤（Intelligent Filtering）**：并非所有信息都值得记住。Mem0 通过优先级评分和上下文标签决定哪些信息需要存储，避免记忆膨胀，让 Agent 专注于关键信息。

**动态遗忘（Dynamic Forgetting）**：Mem0 不会将记忆视为静态数据堆。低相关度的条目会随时间衰减，释放空间和注意力。遗忘不是缺陷，而是智能记忆系统的必要特性。

**记忆整合（Memory Consolidation）**：根据使用频率、最近性和重要程度，信息在短期记忆和长期记忆之间流动，优化召回速度和存储效率。这模仿了人类将知识内化的过程。

**跨会话连续性（Cross-Session Continuity）**：大多数 Agent 在会话结束时重置，而 Mem0 的记忆架构能够在不同会话、设备和时间段之间保持相关上下文的连续性。

### 1.2 记忆类型

| 类型 | 作用 | 示例 |
|------|------|------|
| 工作记忆 | 维持短期对话连贯性 | "上一个问题是什么？" |
| 事实记忆 | 记住用户偏好、沟通风格、领域上下文 | "你偏好 Markdown 输出和简短回答" |
| 情景记忆 | 记住具体的过往交互或结果 | "上次部署这个模型时延迟增加了" |
| 语义记忆 | 存储随时间积累的泛化知识 | "涉及 JSON 解析的任务通常让你头疼，需要模板吗？" |

### 1.3 使用方式

Mem0 提供简洁的 API，支持多种 LLM 后端和向量存储：

```python
from mem0 import Memory

# 初始化
m = Memory()

# 添加记忆
m.add("I prefer short, concise answers", user_id="alice")

# 检索记忆
memories = m.search("What are my preferences?", user_id="alice")

# 获取所有记忆
all_memories = m.get_all(user_id="alice")

# 更新记忆
m.update(memory_id="xxx", data="I prefer detailed answers")

# 删除记忆
m.delete(memory_id="xxx")
```

Mem0 支持多维度组织记忆，可通过 `user_id`、`agent_id`、`app_id` 等维度进行隔离和共享，适合多用户、多 Agent 的场景。

## 2.LangMem

LangMem 是 LangChain 推出的长期记忆 SDK，专注于让 Agent 通过记忆学习和改进行为。它与 LangGraph 的深度集成使其成为 LangChain 生态中构建自适应 Agent 的首选方案。

### 2.1 三种记忆类型

LangMem 将记忆分为三种类型，每种服务于不同的自适应需求：

| 记忆类型 | 目的 | 示例 | 人类类比 |
|----------|------|------|----------|
| 语义记忆（Semantic） | 事实与知识 | 用户偏好、知识三元组 | 知道 Python 是编程语言 |
| 情景记忆（Episodic） | 过往经验 | 对话摘要、Few-shot 示例 | 记得第一天上班的经历 |
| 程序记忆（Procedural） | 系统行为 | 核心性格和响应模式 | 知道如何骑自行车 |

### 2.2 语义记忆：事实提取

语义记忆存储关键事实及其关系，使 Agent 能够记住不会预训练在模型中、也无法通过搜索获取的信息。LangMem 使用 LLM 自动从对话中提取事实：

```python
from langmem import create_memory_manager

manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    instructions="Extract user preferences and facts",
    enable_inserts=True
)

# 从对话中提取事实
conversation = [
    {"role": "user", "content": "Alice manages the ML team and mentors Bob."}
]
memories = manager.invoke({"messages": conversation})
```

### 2.3 程序记忆：行为优化

程序记忆是 LangMem 最具特色的能力。它通过分析成功和失败的交互轨迹，自动更新 Agent 的系统 Prompt，使核心行为模式随经验进化：

```python
from langmem import create_prompt_optimizer

trajectories = [
    (
        [{"role": "user", "content": "Tell me about Mars"},
         {"role": "assistant", "content": "Mars is the fourth planet..."},
         {"role": "user", "content": "I wanted more about its moons"}],
        {"score": 0.5, "comment": "Missed key information about moons"}
    )
]

optimizer = create_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="metaprompt",
    config={"max_reflection_steps": 3}
)

improved_prompt = optimizer.invoke({
    "trajectories": trajectories,
    "prompt": "You are a planetary science expert"
})
```

LangMem 提供多种优化算法：

- **metaprompt**：通过反思和额外思考时间研究对话，用元 Prompt 提出更新建议

- **gradient**：将工作分为批评和 Prompt 提案两个独立步骤

- **prompt_memory**：在单步中完成上述操作

### 2.4 命名空间与隐私

LangMem 中所有记忆都有命名空间（Namespace），最常见的是包含 `user_id` 以防止用户间记忆交叉。记忆可以限定在特定应用路由、单个用户、团队共享，或跨所有用户学习核心程序。

## 3.Letta

Letta（前身为 MemGPT）是一个面向有状态 LLM Agent 的平台，其核心思想是将上下文窗口视为受限的记忆资源，实现类似操作系统的记忆层级管理。

### 3.1 操作系统式的记忆架构

Letta 借鉴操作系统的内存管理理念，将 Agent 记忆分为四个层次：

| 记忆层 | 类比 | 作用 |
|--------|------|------|
| 消息缓冲区（Message Buffer） | CPU 寄存器 | 存储最近的对话消息，提供即时上下文 |
| 核心记忆（Core Memory） | RAM | 可编辑的上下文内记忆块，如用户画像、Agent 人设、当前任务 |
| 回溯记忆（Recall Memory） | 磁盘 | 完整的对话历史，可搜索和检索 |
| 存档记忆（Archival Memory） | 外部存储 | 显式结构化的知识，可使用向量数据库或图数据库存储 |

### 3.2 核心记忆块

核心记忆是 Letta 最具特色的抽象。每个记忆块包含：

- **标签（Label）**：记忆块的名称

- **描述（Description）**：说明存储内容的含义

- **值（Value）**：实际放入上下文的 Token

- **字符限制（Character Limit）**：分配的上下文空间

Agent 可通过工具调用自主编辑记忆块，实现自我记忆管理。其他专门的 Agent（如"睡眠时 Agent"）也可异步优化这些记忆块。

### 3.3 消息驱逐与递归摘要

当上下文窗口达到容量时，Letta 采用智能驱逐策略：

1. 仅驱逐部分消息（如 70%），确保连续性

2. 被驱逐的消息经过递归摘要处理，与已有摘要合并

3. 较旧的消息对摘要的影响力逐渐降低

### 3.4 Sleep-Time Compute

Letta 引入异步记忆管理的范式：

- **非阻塞操作**：记忆管理在空闲时段异步执行，不影响对话响应速度

- **主动记忆精炼**：在 Agent 空闲时重组和优化记忆，而非在对话中进行增量更新

- **更高质量的记忆形成**：有更多时间进行深度思考和记忆整合

### 3.5 使用方式

Letta 提供 API 和 CLI 两种使用方式：

```python
from letta import create_client

client = create_client()

# 创建 Agent
agent = client.create_agent(
    name="my_agent",
    memory={
        "human": {"name": "Alice", "preferences": "concise answers"},
        "persona": {"role": "helpful assistant"}
    }
)

# 发送消息
response = client.send_message(
    agent_id=agent.id,
    message="What do you know about me?"
)
```

Letta 支持多种 LLM 后端，完全模型无关（Model-Agnostic），并提供 Agent File（.af）格式用于序列化有状态 Agent。

## 4.三者对比

| 维度 | Mem0 | LangMem | Letta |
|------|------|---------|-------|
| 定位 | 通用 AI 记忆层 | LangChain 生态记忆 SDK | 有状态 Agent 平台 |
| 核心特色 | 智能过滤 + 动态遗忘 | 程序记忆（Prompt 优化） | OS 式记忆层级管理 |
| 记忆类型 | 工作/事实/情景/语义 | 语义/情景/程序 | 消息缓冲/核心/回溯/存档 |
| 生态集成 | 独立，多框架兼容 | 深度集成 LangGraph | 自有平台 + API |
| 适用场景 | 多用户个性化记忆 | Agent 行为自适应优化 | 长期有状态 Agent |
| 部署方式 | SDK + 云服务 | SDK（自带或托管） | 本地 CLI + 云服务 |

## 5.Reference

[Mem0](https://mem0.ai/blog/memory-in-agents-what-why-and-how/)

[LangMem](https://blog.langchain.com/langmem-sdk-launch/)

[Letta](https://www.letta.com/blog/agent-memory)
