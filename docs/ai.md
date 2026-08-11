# AI Time

如果说从2000年开始互联网领域开始爆发，那么从2022年开始就是AI大模型领域的爆发，约莫20年的变革足以改变行业内的版图。计算机科学的历史其实并不长，但是发展的速度越来越快。

早期从机器学习到深度学习，训练方式还处于数据集的特征处理和信息抽取，未能发现隐藏机制和深层信息，自然语言处理还在传统的word2vec，计算机视觉仍是ResNet和YOLO的天下。直到2017年的一篇《Attention is All You Need》轰动全球，OpenAI推出的GPT系列更是改变了传统计算机与软件工程的发展方向。

如今行业迭代的速度肉眼可见地加快：

- **以年为单位的工程迭代**（2022-2024 Prompt Engineering → 2025 Context Engineering → 2026 Harness Engineering）；
- **以月为单位的模型迭代**（26年一季度各家公司已经推出若干旗舰模型，这一速度正在不断加快）；
- **以周为单位的产品迭代**（产品设计方案和产品原型层出不穷，个人智能体兴起，AGI加速演进）。

不少人直言"只要学的慢什么都不用学"。AI时代到底学什么？怎么学？我也正在逐步探索这个问题。对于入门，我觉得必须要看[llm-course](https://github.com/mlabonne/llm-course)！而下面这个板块，就是我以"从模型基座到AI应用"为主线搭建的 AI 知识库。

## 知识地图

整个板块按"从理论到落地"的主线组织，分为四个阶段：

### 一、基础理论

理解模型的来路，打好地基。

| 板块 | 简介 |
|------|------|
| [机器学习](ai-note/ml/index.md) | 经典机器学习基础：模型评估与选择、线性模型、决策树、SVM、聚类、集成学习等 |
| [深度学习](ai-note/dl/index.md) | 神经网络入门：CNN、ResNet、生成/判别模型、VAE、GAN |
| [自然语言处理](ai-note/nlp/index.md) | 文本预处理、词向量、RNN、预训练语言模型、机器翻译、情感分析 |

### 二、模型基座

大模型本体，理解模型本身。

| 板块 | 简介 |
|------|------|
| [大语言模型 LLM](ai-note/llm/index.md) | 从 Tokenization、Transformer 到 MoE/Mamba、预训练、微调、对齐、Scaling Law |
| [扩散语言模型 DLM](ai-note/dlm/index.md) | U-Net、DDPM、DDIM、Flow Matching、DiT、LDM、LLaDA |
| [多模态大模型 MLLM](ai-note/mllm/index.md) | ViT、CLIP、BLIP、LLaVA、离散 Tokenization |
| [生成式 AI AIGC](ai-note/aigc/index.md) | Stable Diffusion、Midjourney、DALL-E、Suno |

### 三、模型工程

模型如何被训练、优化与部署，走向生产。

| 板块 | 简介 |
|------|------|
| [LLM 强化学习](ai-note/llm-rl/index.md) | RLHF、RLVR、RLAIF、Agentic RL |
| [分布式训练与推理](ai-note/llm-distributed-training-and-inference/index.md) | KV-Cache、FlashAttention、DDP/ZeRO、Megatron、vLLM、推测解码 |
| [模型压缩](ai-note/llm-compression/index.md) | 剪枝、量化、稀疏注意力、上下文稀疏 |
| [部署](ai-note/llm-deploy/index.md) | LLM API 等上线相关 |
| [评测](ai-note/llm-evaluation/index.md) | LLM 评估方法与基准 |
| [安全](ai-note/llm-safety/index.md) | MIA、Agent 安全等 |

### 四、应用落地

如何让模型可靠地做事，工程化与产品化。

| 板块 | 简介 |
|------|------|
| [提示词工程](ai-note/prompt-engineering/index.md) | CoT、ChatGPT 策略、System Prompt、学术提示词 |
| [上下文工程](ai-note/context-engineering/index.md) | 上下文分类、Context-Rot、渐进式披露 |
| [驾驭工程](ai-note/harness-engineering/index.md) | 六层架构、一线公司（Anthropic/OpenAI）实践 |
| [智能体 Agent](ai-note/agent/index.md) | ReAct、MCP、A2A、工具、框架、Sandbox、记忆、Skill |
| [检索增强生成 RAG](ai-note/rag/index.md) | 向量数据库、GraphRAG、RAG 评估、高级 RAG |
| [深度研究](ai-note/deep-research/index.md) | Deep Research 产品、DeerFlow |
| [工作流](ai-note/workflow/index.md) | AI 工作流与工具使用：Claude Code、Codex、Dify、Coze、n8n 等 |
| [模型平台](ai-note/model/index.md) | models.dev、OpenRouter、ModelScope、HuggingFace |

