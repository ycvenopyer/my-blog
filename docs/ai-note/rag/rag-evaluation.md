# RAG 评估

## 1.为什么要做 RAG 评估

搭建好一个 RAG 系统后，一个很现实的问题是：这个系统到底好不好用？靠几次手动测试显然不靠谱，可能刚好问了简单的问题，也可能漏掉了潜在的漏洞。就像训练深度学习模型时需要 Loss 来量化优化方向，RAG 系统也需要一套清晰的评估方法来衡量性能、定位瓶颈、指导优化。

## 2.RAG 三元组

标准 RAG 流程中涉及三个核心元素，构成"RAG 三元组"：

- **Question（用户问题）**：用户提出的查询

- **Context（检索上下文）**：从知识库中检索到的相关文档片段

- **Answer（生成回答）**：LLM 基于上下文生成的最终回答

RAG 评估的本质就是检测这三者之间的关系质量：

| 评估维度 | 衡量关系 | 说明 |
|----------|----------|------|
| Context Relevance | Context ↔ Question | 召回的文档是否支持用户问题 |
| Groundedness | Answer ↔ Context | 回答是否忠实于检索到的上下文 |
| Answer Relevance | Answer ↔ Question | 回答是否直接解决了用户问题 |

## 3.评估数据构建

评估数据集通常包含四个部分：

- **question**：用户输入的问题

- **contexts**：检索到的相关文档片段

- **answer**：RAG 系统生成的回答

- **ground_truths**：人工标注的参考答案

数据集的构建方式有三种：

- **人工制作**：标注人员根据知识库设计问题和参考答案，最准确但成本最高

- **日志收集**：从系统实际运行日志中提取真实问答对，最贴近真实场景但需要清洗

- **大模型生成**：用 LLM 基于知识库自动生成问题和参考答案，效率最高，是目前最常用的方式

## 4.评估方式

| 方式 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| 人工评估 | 人工对照参考答案打分 | 能处理复杂场景 | 主观、耗时、一致性差 |
| 基于规则 | BLEU、ROUGE、F1 等传统指标 | 客观、高效 | 灵活性差，难以覆盖复杂情况 |
| 大模型评估 | 用 LLM 根据评分规则自动打分 | 智能、灵活、可扩展 | 大模型本身存在不稳定性 |

## 5.RAGAS 评估指标

RAGAS（RAG Assessment）是目前最主流的 RAG 评估框架，其指标按"检索"和"生成"两大阶段划分：

### 5.1 检索阶段指标

**上下文精确率（Context Precision）**

衡量检索结果中与参考答案相关的条目是否排名靠前。值域 [0, 1]，越高越好。如果相关文档排在后面，说明检索排序有问题。

**上下文召回率（Context Recall）**

衡量检索到的上下文是否覆盖了参考答案中的全部关键信息。值域 [0, 1]，越高越好。低分说明检索遗漏了重要内容。

**上下文相关性（Context Relevancy）**

衡量检索到的上下文中有多少内容真正与问题相关。理想情况下，召回的文档应只包含解答问题所需的信息，不应包含大量无关内容。值域 [0, 1]，越高越好。

### 5.2 生成阶段指标

**忠实度（Faithfulness）**

衡量生成的回答与给定上下文的事实一致性。从回答中提取所有 claims，逐一检查是否可从上下文中推断出来。值域 [0, 1]，越高越好。低分意味着模型出现了"幻觉"或偏离了检索内容。

**答案相关性（Answer Relevancy）**

衡量生成的回答与用户问题的匹配程度。不完整或包含冗余信息的答案得分较低。该指标通过让 LLM 从回答中反推问题，再计算反推问题与原问题的相似度来评估。值域 [0, 1]，越高越好。

**答案正确性（Answer Correctness）**

衡量生成答案与参考答案的相似程度，包括语义相似度和事实覆盖度。值域 [0, 1]，越高越好。

### 5.3 指标速查表

| 指标 | 阶段 | 输入 | 说明 |
|------|------|------|------|
| Faithfulness | 生成 | Answer + Context | 回答是否基于检索内容 |
| Answer Relevancy | 生成 | Answer + Question | 回答是否切题 |
| Answer Correctness | 生成 | Answer + Ground Truth | 回答是否正确 |
| Context Precision | 检索 | Contexts + Question | 相关文档是否排前面 |
| Context Recall | 检索 | Contexts + Ground Truth | 是否覆盖了全部关键信息 |
| Context Relevancy | 检索 | Contexts + Question | 召回内容是否都相关 |

## 6.RAGAS 实践

### 6.1 安装

```bash
pip install ragas
```

### 6.2 准备评估数据

```python
from datasets import Dataset

questions = [
    "客户经理被投诉了，投诉一次扣多少分？",
    "客户经理每年评聘申报时间是怎样的？",
]

ground_truths = [
    "每投诉一次扣2分",
    "每年一月份为客户经理评聘的申报时间",
]

answers = []
contexts = []

# 运行 RAG 系统获取 answers 和 contexts
for query in questions:
    answers.append(rag_chain.invoke({"question": query}))
    contexts.append([doc.page_content for doc in retriever.get_relevant_documents(query)])

data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths,
}

dataset = Dataset.from_dict(data)
```

### 6.3 执行评估

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
print(df)
```

### 6.4 结果分析

评估结果通常以箱线图等方式可视化，便于直观分析：

- **中位数**：反映整体水平

- **四分位范围**：反映稳定性

- **极端值**：反映系统的薄弱环节，是优化的重点方向

例如，如果 Faithfulness 中位数在 0.75 左右但有少数低至 0.25 的极端值，说明系统整体表现不错，但在某些问题上出现了严重幻觉，需要重点排查。

## 7.基于评估结果的优化方向

根据各指标的表现，可以精准定位问题并优化：

| 指标偏低 | 问题定位 | 优化方向 |
|----------|----------|----------|
| Context Recall | 检索遗漏关键信息 | 优化分块策略、扩展数据源、调整检索参数 |
| Context Precision | 相关文档排名靠后 | 引入 Rerank 重排序、调整向量检索权重 |
| Context Relevancy | 召回太多无关内容 | 缩小 Top-K、增加过滤条件 |
| Faithfulness | 回答偏离检索内容 | 优化 Prompt、增加事实校验、降低温度参数 |
| Answer Relevancy | 答非所问 | 优化 Prompt 约束输出格式、改进查询重写 |
| Answer Correctness | 回答不准确 | 综合以上所有方向优化 |

## 8.其他评估工具

| 工具 | 特点 |
|------|------|
| **RAGAS** | 开源、轻量、指标丰富，几行代码即可跑通，适合快速评估 |
| **LangSmith** | 全链路监控平台，支持链路追踪、模型监控、A/B 测试，适合长期维护的 RAG 服务 |
| **DeepEval** | 基于 pytest 的评估框架，支持单元测试风格的 RAG 评估 |
| **LlamaIndex 评估模块** | LlamaIndex 生态内置的评估工具，与 LlamaIndex 深度集成 |
| **ARES** | 斯坦福大学开源框架，支持自动合成训练数据和评估 |

## 9.总结

RAG 评估是 RAG 开发中不可或缺的一环。光靠手动提问无法看出系统的整体水平，也无法知道瓶颈在哪。有了量化指标，就能直观地看到优点和缺点，后续优化才有方向。

评估的核心思路可以概括为：

1. **构建数据集**：问题 + 检索内容 + 生成回答 + 参考答案

2. **选择指标**：根据关注点选择检索指标和生成指标

3. **执行评估**：使用 RAGAS 等框架自动打分

4. **分析结果**：通过可视化定位薄弱环节

5. **持续优化**：根据指标反馈迭代改进

## 10.Reference

- [RAG 模型效果评估](https://1shuangjiang1.github.io/p/rag-%E6%A8%A1%E5%9E%8B%E6%95%88%E6%9E%9C%E8%AF%84%E4%BC%B0/)
- [2025年必备的RAG评估框架与工具详解](https://zhuanlan.zhihu.com/p/1892529470419736435)
- [RAG质量评估和RAGAS评估指南](https://developer.volcengine.com/articles/7540557260657786899)
- [RAG应用评估全攻略：从RAGAS原理到可视化实战](https://developer.aliyun.com/article/1707047)
- [RAG评测完整指南：指标、测试和最佳实践](https://cloud.tencent.com/developer/article/2597186)
- [RAGAS 官方文档](https://docs.ragas.org.cn/en/stable/)

