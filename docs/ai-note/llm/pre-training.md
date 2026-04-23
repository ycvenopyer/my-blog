# Pre-training

## 1.什么是预训练？

Pre-training是指在一个较小的、特定任务的数据集上进行微调之前，在一个大数据集上训练一个模型的过程。这个初始阶段允许模型从数据中学习一般的特征和表征，然后可以针对具体任务进行微调。

简单来说就是让模型具备通用能力。

数据：海量无标注/弱标注数据（如互联网文本、图像库）。

## 2.LLM的基本结构

### 2.1 Alpaca

### 2.2 Vicuna

### 2.3 前置归一化

pre-norm, post-norm, RMSNorm, LayerNorm

### 2.4 SwiGLU

### 2.5 Scaling Law

## 3.注意力机制的优化

### 3.1 Longformer Sparse Attention

### 3.2 MHA/MQA/GQA

## 4.位置编码策略

### 4.1 RoPE

### 4.2 ALiBi(Attention with Linear Biases)

## 5.长上下文处理策略

### 5.1 位置插值法

### 5.2 基于NTK的方法

### 5.3 LongLoRA

### 5.4 YaRN (Yet another RoPE extension methods)

## 6.Reference

[什么是预训练？](https://ai-bot.cn/what-is-pre-training/)
