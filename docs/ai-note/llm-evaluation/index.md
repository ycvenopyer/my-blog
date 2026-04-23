# LLM Evaluation 

## 1.核心评估框架：从传统指标到模型裁判

评估一个LLM，可以根据测试目标和可用资源，选择不同层次的评估方法。

- 传统自动指标 (Traditional Metrics)：包括 BLEU、ROUGE、BERTScore 等，适用于机器翻译、文本摘要等有标准答案的任务，通过计算n-gram重叠或语义相似度来打分。它们计算快，但无法衡量语义的细微差别和逻辑连贯性。
  
- 评测基准 (Benchmarks)：这是最主流的方法，通过让模型做一系列标准化“考题”来评分。常用基准按能力划分，例如：GLUE/SuperGLUE（基础语言理解）、MMLU（多任务知识）、HellaSwag（常识推理）、TruthfulQA（真实性）。这种方式结果直观、可复现，但存在数据污染和刷榜风险。

- 人类评估 (Human Evaluation)：由人工评估者直接评价模型输出的质量，通常是最可靠但也是成本最高、最耗时的评估方式。

- 模型作为裁判 (LLM-as-a-Judge)：让一个能力强大的LLM（如GPT-4）去评估其他模型的输出。它能处理复杂的、开放式的生成任务，但可能存在偏见和不一致。

- 系统评估 (System Evaluation)：与仅针对基座模型的模型评估不同，系统评估更关注LLM在真实应用场景（如RAG系统、Agent）中的综合表现，包括提示工程、检索逻辑、工具调用等整个管线的效果。

## 2.常用评测基准

- GLUE / SuperGLUE：评估通用的语言理解能力。GLUE包含9项自然语言理解任务，SuperGLUE是难度升级版。

- MMLU：衡量模型在57个学科（包含STEM、人文、社科等）上的知识广度。MMLU-Pro是难度升级版，GPQA的问题则更难，旨在达到“专家级”水平。
  
- BIG-bench：包含200多个超难任务，旨在探索LLM的极限能力。
  
- HELM (Holistic Evaluation of Language Models)：一个更全面的评估框架，除了准确性，还强调鲁棒性、公平性、偏见、效率等多个维度。
  
- C-Eval：一个全面的中文基础模型评估套件，覆盖了人文、社科、理工等多个领域。
  
- TruthfulQA：专门用于衡量模型回答的真实性，检测其是否倾向于模仿人类的常见错误认知。
  
- SafetyBench：专注于评估模型在安全、偏见、伦理等方面的表现。
  
- HumanEval：用于评估模型生成代码的正确性和功能性，是代码能力的经典基准。
  
- AGIEval：使用人类标准化考试（如高考、司法考试）的题目来评估模型的“人机对齐”水平。

## 3.评估面临的挑战

- 数据污染：评测数据集可能无意中混入模型的训练数据，导致分数虚高，无法反映真实能力。
  
- 模型敏感性：LLM对提示词的微小变化非常敏感，这会影响结果的稳定性和可复现性。
  
- 评估成本与偏见：人类评估成本高昂，而LLM作为裁判又可能引入新的偏见。
  
- 动态性与复杂度：静态基准测试难以衡量模型在动态、多步骤交互环境下的真实能力，为此研究人员正探索动态评估方法。
  
- 评估流程不统一：不同研究的评估设置差异巨大，导致结论不一致。
  
- 安全与价值观风险：LLM可能生成违背人类价值观或存在偏见的内容，甚至被恶意输入诱导。

## 4.评估工具与最佳实践

关键实践步骤：

1.  明确评估目标：先明确业务目标和成功标准。
   
2.  创建测试集：构建能代表真实应用场景的数据。
   
3.  选择合适的指标：为不同场景挑选合适的评估指标（如RAG系统可关注忠诚度、上下文相关性等）。
   
4.  使用评估工具：借助DeepEval、Ragas等框架，高效实现评估流程。
   
5.  持续监控：将评估集成到开发和部署流程中，进行持续监控和优化。

## 5.Reference

[evaluation-guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)

[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

[Build an LLM Evaluation Framework: Metrics, Methods & Tools](https://www.codecademy.com/article/build-an-llm-evaluation-framework)

[Enhancing Evaluation Practices for Large Language Models](https://opendatascience.com/enhancing-evaluation-practices-for-large-language-models/?utm_campaign=Newsletters&utm_medium=email&_hsenc=p2ANqtz-_AZVynXVVcLcDEv98PAxYDeSGgC0OY9J4Q4k8dwstgthI9hrQ3IBhfyly8exNinWXLhVZoYrhwtlDiy3fm5LlszJ9agw&_hsmi=2&utm_content=2&utm_source=hs_email)

