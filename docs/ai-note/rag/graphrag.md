# GraphRAG

## 1.定义

GraphRAG是将知识图谱（Knowledge Graph）引入 RAG 架构的检索增强生成范式。传统 RAG 以文本块（Chunk）为粒度进行向量相似度检索，在面对需要跨文档综合、全局主题归纳、多跳推理等复杂查询时存在天然局限。GraphRAG 通过从非结构化文本中自动构建实体-关系网络（知识图谱），并基于图结构进行检索与生成，从而提供更具深度和上下文感知能力的答案。

### 1.1 传统 RAG 的局限性

| 查询类型 | 传统 RAG | GraphRAG |
|----------|----------|----------|
| 具体事实检索 | 表现良好 | 表现良好 |
| 全局主题归纳 | 失败 | 已解决 |
| 模式与趋势分析 | 失败 | 已解决 |
| 跨文档综合 | 能力有限 | 已解决 |
| 关系推理 | 无法实现 | 已解决 |

传统 RAG 的核心问题是"相似的不等于代表性的"。当用户问"这些文档的主要趋势是什么"时，向量检索只能返回与问题字面最相似的几个 Chunk，如同只见树木不见森林。

### 1.2 GraphRAG 的核心思想

> **在索引阶段生成摘要，而非在查询时生成。**

GraphRAG 通过社区检测（Community Detection）将知识图谱划分为多个语义簇，并在索引阶段为每个社区预生成层次化摘要。查询时利用这些预构建的结构化摘要，实现对全局信息的快速理解和回答。

## 2.工作原理

GraphRAG 的完整流程可分为四大阶段：

```
文档 → 实体与关系抽取 → 知识图谱构建 → 社区检测 → 层次化摘要生成 → 检索与生成
```

### 2.1 实体与关系抽取

利用 LLM 对每个文档 Chunk 进行信息抽取，识别出：

- **实体（Entities）**：人物、组织、地点、技术、产品等，附带名称、类型和描述

- **关系（Relationships）**：实体之间的语义关系，如"投资了"、"合作于"、"开发了"等，附带关系类型和权重

- **实体声明（Covariates）**：可选的附加信息，如事件、主张、时间等

这一步通常通过精心设计的 Prompt 让 LLM 输出结构化的 JSON 数据。

### 2.2 知识图谱构建

将抽取的实体作为节点、关系作为边，构建知识图谱。图结构支持：

- 节点属性：实体类型、描述、来源文档

- 边属性：关系类型、权重、来源文档

- 重复实体的合并与消歧

- 关系的去重与聚合

### 2.3 社区检测

使用 Leiden 算法（一种改进的 Louvain 社区检测算法）对知识图谱进行层次化社区划分。Leiden 算法通过优化模块度（Modularity），将连接紧密的节点聚集到同一社区中。

社区具有层次结构：底层社区粒度细（如"某公司的投资关系"），高层社区粒度粗（如"整个 AI 产业生态"）。这种层次化结构为后续的层次化摘要奠定了基础。

### 2.4 层次化摘要生成

对每个社区，利用 LLM 生成摘要报告，内容包括：

- 社区内的关键实体及其角色

- 核心关系与互动模式

- 社区的整体主题概括

摘要从底层社区逐层向上聚合，形成层次化的摘要树。这一步是离线完成的，是 GraphRAG 索引阶段最耗时的操作。

### 2.5 检索与生成

GraphRAG 提供两种搜索模式：

#### 局部搜索（Local Search）

适用于针对特定实体的精确查询，如"AlphaTech 与哪些公司有合作？"

流程：

1. 从查询中识别目标实体

2. 在知识图谱中探索该实体的邻居节点（1-hop、2-hop）

3. 收集相关实体、关系和社区摘要

4. 将上下文提供给 LLM 生成答案

#### 全局搜索（Global Search）

适用于针对整个数据集的宏观查询，如"这些文档的主要主题和趋势是什么？"

流程：

1. 收集所有社区的层次化摘要

2. 从各摘要中提取与查询相关的信息片段

3. 将多个局部答案综合为全局回答

全局搜索是 GraphRAG 区别于传统 RAG 的核心能力，它使系统能够"看到整片森林"。

## 3.GraphRAG vs 传统 RAG

### 3.1 架构对比

| 维度 | 传统 RAG | GraphRAG |
|------|----------|----------|
| 索引单元 | 文本 Chunk | 实体、关系、社区 |
| 索引方式 | 向量 Embedding | 知识图谱 + 社区摘要 |
| 检索方式 | 向量相似度（ANN） | 图遍历 + 社区摘要聚合 |
| 上下文构建 | Top-K 相似 Chunk | 实体邻居 + 社区报告 |
| 全局理解 | 不支持 | 原生支持 |
| 索引成本 | 低（Embedding API 调用） | 高（LLM 抽取 + 摘要生成） |

### 3.2 成本对比

| 阶段 | 传统 RAG | GraphRAG |
|------|----------|----------|
| 索引（每文档） | ~$0.001（Embedding） | ~$0.1-1.0（LLM 抽取 + 摘要） |
| 查询（每次） | ~2000 tokens 输入 | Local: ~3000 tokens / Global: ~10000+ tokens |

GraphRAG 的索引成本显著高于传统 RAG，因为需要大量 LLM 调用进行实体抽取和摘要生成。但在需要全局理解和关系推理的场景中，其回答质量远超传统 RAG。

### 3.3 适用场景

| 场景 | 推荐方案 |
|------|----------|
| 具体事实检索 | 传统 RAG |
| 全局主题归纳 | GraphRAG |
| 成本敏感 | 传统 RAG |
| 关系推理需求 | GraphRAG |
| 实时响应要求高 | 传统 RAG |
| 文档间关联分析 | GraphRAG |

## 4.实现框架与工具

### 4.1 Microsoft GraphRAG

微软官方开源实现，提供完整的 CLI 和 Python API：

```bash
# 安装
pip install graphrag

# 初始化项目
graphrag init --root ./my_graphrag

# 索引构建
graphrag index --root ./my_graphrag

# 全局搜索
graphrag query --root ./my_graphrag --method global \
  --query "What are the main themes in these documents?"

# 局部搜索
graphrag query --root ./my_graphrag --method local \
  --query "Tell me about AlphaTech"
```

### 4.2 其他实现方案

| 工具 | 特点 |
|------|------|
| Neo4j + LangChain | 利用 Neo4j 图数据库存储，LangChain 提供 GraphCypherQAChain 实现自然语言到 Cypher 的转换 |
| NetworkX + 自研 | 轻量级方案，适合原型验证和小规模数据 |
| NVIDIA cuGraph | GPU 加速的图分析库，适合大规模图计算 |
| Diffbot Knowledge Graph | 商业化的自动知识图谱构建服务 |

## 5.典型应用场景

### 5.1 法律尽职调查

分析大量法律文档，识别关键人物、公司、交易之间的关系网络，发现隐藏的利益关联和风险模式。

### 5.2 科研文献分析

对某一领域的海量论文进行主题聚类和趋势分析，帮助研究人员快速把握领域全貌和前沿方向。

### 5.3 情报分析

从多源情报文本中自动构建实体关系网络，支持分析师发现关键人物、组织和事件之间的关联。

### 5.4 企业知识管理

整合企业内部文档、报告、邮件等非结构化数据，构建企业知识图谱，支持全局知识发现。

### 5.5 新闻事件追踪

从新闻流中自动提取事件要素和参与方，追踪事件发展脉络和影响范围。

## 6.挑战与优化方向

### 6.1 索引成本

GraphRAG 的索引阶段需要大量 LLM 调用，成本较高。优化方向包括：

- 使用更小的模型进行实体抽取

- 增量索引，仅对新文档进行处理

- 缓存和复用中间结果

### 6.2 抽取质量

LLM 抽取的实体和关系可能存在噪声、重复和错误。优化方向包括：

- 实体消歧与归一化

- 定义领域特定的实体类型和关系模式

- Prompt 调优与少样本示例

### 6.3 动态更新

知识图谱需要随新数据的到来而更新。当前的 GraphRAG 主要面向静态数据集，增量更新机制仍在发展中。

### 6.4 混合检索

在实际生产中，GraphRAG 与传统 RAG 并非替代关系，而是互补关系。最佳实践是构建混合检索系统：

```python
def hybrid_search(query: str):
    if is_global_question(query):
        return graphrag.global_search(query)
    elif contains_entity(query):
        return graphrag.local_search(query)
    else:
        return traditional_rag.search(query)
```

## 7.总结

GraphRAG 通过引入知识图谱和社区检测，解决了传统 RAG 在全局理解和关系推理方面的核心短板。其"索引时生成摘要"的设计理念和"局部-全局"双搜索模式，为复杂文档分析提供了全新的范式。然而，高昂的索引成本和抽取质量的不确定性仍是实际落地中需要权衡的关键因素。

## 8.Reference

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [GraphRAG: From Local to Global at Scale (论文)](https://arxiv.org/abs/2404.16130)
- [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG 官方文档](https://microsoft.github.io/graphrag)
- [完整案例透视GraphRAG工作机制：从图构建到图检索](https://zhuanlan.zhihu.com/p/1922782829256868561)
