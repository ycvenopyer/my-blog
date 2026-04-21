# Speculative Decoding

随着大规模语言模型（LLM）在各类自然语言处理任务中的广泛应用，模型推理效率成为制约其落地和推广的关键因素，推理同时对吞吐和时延具有较高要求。投机解码（Speculative Decoding）作为一种提升大模型推理性能的有效技术，近年来受到广泛关注。投机解码通过引入辅助模型（通常为小型语言模型）或结构来预测主模型的输出，从而减少主模型的计算负担，提高推理速度。

大模型推理性能优化-投机解码： Transformer架构的大模型推理的解码（decode phase）每次根据注意力窗口内的所有上文信息，生成一个新token。这个自回归的过程受到访存带宽的限制（memory-bound，访存密集型）。投机解码（Speculative Decoding）通过引入辅助的草稿模型（通常为小型语言模型，draft model）或结构来预测主模型的输出的连续若干个token，目标模型（target model）只需对草稿模型预测的token进行验证，从而在前向计算中并行生成多个token，减少目标模型的计算负担，提高推理速度。

投机解码的研究领域主要集中在token预测的优化上：研究如何设计高效的draft model或预测模块，包括模型结构、参数量、训练方法、预测token数量等，以在保证预测准确率的同时最大化推理速度提升。

## 1.方向1

通过缓存之前预测的有效信息来提升draft model的预测能力：通过设计高效的缓存机制，存储和利用之前预测的（包括没有被命中）token及其上下文信息，提升draft model在连续token预测中的准确率和效率。可以优化的方向包括但不限于： a. 缓存更有效的信息，以提升预测效率； b. 优化检索策略，如向量检索，文本匹配等的效率。

## 2.方向2：高效Draft model结构设计

Draft model结构设计：优化预测模块结构设计，draft model的设计目标通常是得到和target model近似的数据分布，从而提升在验证阶段target model接受的token数量，主流的draft model如eagle经常采用和target model一样的自回归设计，然而我们认为自回归的draft结构并不是通向模拟target model数据分布的唯一途径，我们希望draft model在保持住数据分布的同时能够具有其他的如并行采样的优秀性质。可以优化的方向包括但不限于：a. 设计出最小推理代价但最大保留原模型数据分布的新型网络架构 b. 探索高速高效的draft model采样机制来战胜自回归draft采样.

## 3.推荐论文和项目

SpecForge：SpecForge is an ecosystem project developed by the SGLang team. It is a framework for training speculative decoding models so that you can smoothly port them over to the SGLang serving framework to speed up your inference. Learn more: https://docs.sglang.ai/SpecForge/。

Accelerating Large Language Model Decoding with Speculative Sampling (2023)

Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads （SD方向的著名论文，与上述方向2相关）

EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty （还包括EAGLE-2, 3系列，EAGLE-3是当前业界落地的SOTA工作

Break the Sequential Dependency of LLM Inference Using Lookahead Decoding

Better & Faster Large Language Models via Multi-token Prediction

Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling （与上述方向1强相关）

此外，还包括Deepseek等的MTP等优化方案，可以扩展阅读。
