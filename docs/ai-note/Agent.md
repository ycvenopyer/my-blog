## Agent定义

**能够感知环境并采取行动以实现目标的智能体。**

`Agent` = `LLM`+`Planning skills`+`Memory`+`Tool use`

![agent组件](image/agent.png)

这种系统=**LLM（大脑）+Planning skills（规划）+Memory（记忆）+Tool use（工具使用）**。

**Agenticness**是一种**程度**，不是**标签**，用来评估一个AI系统agent化的程度，即系统在**复杂环境下、有限的直接监督下，灵活实现复杂目标的能力程度**。

在产业界，将以Autonomous Agent作为目标但表现强弱不一的所有系统统称为Agentic System，并根据Agenticness的程度加以区分。

## Agent vs Workflow

这是Agentic System内部的两种架构模式：**workflow和agent**。

`workflow`：通过**预定义**代码路径编排LLM和工具的系统,强调的是过程的**标准化和自动化**,比如软件开发过程中,代码提交->代码审查->测试->部署,每个环节都有前后依赖和触发条件。

`Agent`：通过LLM**动态**指导其自身流程和工具使用，保持对任务完成过程控制的系统,强调**适应性与自主性,**比如一个客服Agent能根据用户输入判断意图,自主选择回复内容,甚至调用工具解决问题,而无需严格遵守固定步骤。

`Agentic AI`:是Agent概念的另一种表达,Agentic AI指旨在独立作出决策并采取行动以实现特点目标的软件程序,可认为AI Agent = Agentic AI System。

Agentic AI的关键特征:**记忆,规划,环境感知,工具使用,自主达成目标**。

Agentic AI的常见特征:**从环境中学习,制定复杂计划,自主执行任务**。

## Agentic在现阶段的重要性

**提升企业员工生产效率、优化业务运营、赋能产品服务和商业模式创新等方面发挥着重要作用**。

强调Agentic，就是在探索和定义未来**人机协作的新范式**：人类负责设定目标、提供价值观和进行关键审核；智能体负责执行复杂的操作和探索解决方案。

**长期愿景：迈向通用人工智能（AGI）的过渡形态**。

智能体被视为迈向AGI的**关键中间形态**：通过赋予AI“自主行动”的能力，研究者可以探索更高级的认知架构（如目标驱动、多智能体协作），推动AI向更通用、更拟人化的方向演进。

## Agent技术栈

![agent技术栈](image/agent-tech-stack.png)

1.**模型层**

**LLM或其他模型**：OpenAI(GPT系列), Google(Gemini系列), Anthropic(Claude系列)

**推理与规划**：CoT, ToT, ReAct框架

2.**数据层**

**存储是对于有状态的 Agent 来说是一个基本构建块——Agent 由其对话历史、记忆以及用于检索增强生成（RAG）的外部数据源等持久化状态来定义。**

**向量数据库**：Chroma, Weaviate,Pinecone,Milvus,pgvector,Faiss, Qdrant

**RAG**：RAGFlow, GraphRAG, RAGAS（评估）

3.**逻辑编排层**

**MCP协议**

**Agent框架**：Agent 框架协调大语言模型调用并管理 Agent 状态,如AutoGen, crewAI, LangChain,LangGraph, AutoGPT, LlamaIndex

**可观测性**：langfuse, langwatch, openlit

**提示词工程**：TypeChat, DSPy, Promptify

4.**可视化编排层(低代码,无代码编排能力)**

xyflow, Langflow, Flowise

5.**应用层**

XUI, API, SDK, Application

## 单智能体 vs 多智能体

单智能体架构：一个LLM自己完成所有的推理、规划、工具执行。

多智能体架构：涉及两个或多个Agent，可以是同一个LLM或者一组不同的LLM。

多智能体架构可分为2个分类：垂直架构和水平架构。这两种架构是两个极端，大部分现有的架构处于两者之间。

垂直架构：在这种结构中，一个智能体充当领导者，其他智能体直接向其报告。根据架构的不同，报告智能体可能只与领导智能体通信。或者，领导者可以定义为所有智能体之间的共享对话。垂直架构的定义特征包括有一个领导智能体和清晰的分工。

水平架构：在这种结构中，所有智能体都被视为平等的，并且是关于任务的一个群组讨论的一部分。智能体之间的通信发生在一个共享的线程中，每个智能体都可以看到其他智能体的所有消息。智能体也可以自愿完成特定任务或调用工具，这意味着它们不需要由领导者智能体分配。水平架构通常用于协作、反馈和组织讨论对任务总体成功至关重要的任务。

## AI Coding工具

Trae(字节跳动)

MarsCode(豆包)

CodeGeeX(智谱AI)

通义灵码(阿里云)

Cursor

Claude Code
