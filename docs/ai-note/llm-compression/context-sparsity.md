# Context Sparsity

上下文稀疏性（contextual sparsity）指：对给定输入，只需动用模型中一小部分、随输入而变的注意力头与 MLP 参数子集，即可在近似意义上复现稠密前向的输出；稀疏模式依赖整条上下文（token 及其相互作用），而非仅依赖单个 token 的静态嵌入。它与训后一次性固定掩码的静态掩码相对：动态剪枝在推理前向中按当前层输入、中间表示等实时决定保留哪些头、神经元或计算路径，从而在不改预训练权重的前提下省算力。

上下文稀疏性这一概念在[Deja Vu](https://arxiv.org/abs/2310.17157)中首次提出，论文是这样定义的：不显式重训大模型，在推理时对每一层、每一步生成，为当前输入选出结构化稀疏子集——具体是注意力头集合 \(S_A\) 与 MLP 神经元集合 \(S_M\)——使得稀疏注意力 \(\mathrm{MHA}_{S_A}\) 与稀疏 MLP \(\mathrm{MLP}_{S_M}\) 的输出与全量计算足够接近。这类稀疏是输入条件（input-conditioned）的：不同样本、不同上下文对应不同的 \(S_A, S_M\)，因此称为上下文稀疏，以区别于与输入无关的全局静态稀疏。

Deja Vu 的关键发现：

- **上下文稀疏性的存在性**：在 [OPT](https://arxiv.org/abs/2205.01068) 等预训练 LLM 上，用两遍前向的验证方式：第一遍记录对当前输入输出范数较大的头与 MLP 神经元，第二遍只算这些子集；在多种语言建模与 [in-context learning](https://arxiv.org/abs/2301.00234) 任务上，性能与稠密模型接近。实证上平均可对注意力头施加约 80% 稀疏、对 MLP 神经元约 95% 稀疏；结合 MLP 参数量更大，整体约 85% 量级的结构化上下文稀疏，理论上对应可观的计算/访存缩减空间。

- **可预测性依赖上下文，而非单 token**：仅用单 token 嵌入、不含足够上下文信息时，稀疏模式预测不准；需要携带前文混合信息的层间表示，才能较准地预测当前步需要的头与神经元。稀疏选择与层参数（头/MLP）和上一层输出之间的相似性相关，这一观察支撑了后面的预测算法设计。

- **注意力侧的直觉：并非均匀重要的头**：部分头呈现近似均匀混合（对所有 token 注意力较平），另一部分头在特定 token 上形成高注意力峰值；对预测而言，保留重命中式头、略去对当前步贡献小的头，往往仍能保持输出质量。论文还将单头动力学与均值漂移式聚类等视角联系，解释为何不同头在不同投影空间里刻画不同的 token 交互，从而自然出现按输入变化的稀疏激活。

- **MLP 侧与残差：动态预测可行的结构原因**：激活函数使 MLP 本身存在激活稀疏；同时残差连接使 token 表示在相邻层间变化相对缓慢，因此可以用跨层的 lookahead 预测下一子模块需要的稀疏子集，而不必在每一步都付出完全的串行预测开销。

近年来相关论文如下：

1.  [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://arxiv.org/abs/2310.17157)
2.  [Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity](https://arxiv.org/abs/2505.14884)
3.  [Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing](https://arxiv.org/abs/2502.15618)
4.  [DLP: Dynamic Layerwise Pruning in Large Language Models](https://arxiv.org/abs/2505.23807)
5.  [SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning](https://arxiv.org/abs/2508.06447)
6.  [DART-ing Through the Drift: Dynamic Tracing of Knowledge Neurons for Adaptive Inference-Time Pruning](https://arxiv.org/abs/2601.22632)
7.  [Runtime Adaptive Pruning for LLM Inference](https://arxiv.org/abs/2505.17138)
8.  [SkipGPT: Dynamic Layer Pruning Reinvented with Token Awareness and Module Decoupling](https://arxiv.org/abs/2506.04179)
9.  [Instruction-Following Pruning for Large Language Models](https://arxiv.org/abs/2501.02086)
10. [μ-MoE: Test-Time Pruning as Micro-Grained Mixture-of-Experts](https://arxiv.org/abs/2505.18451)
11. [MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](https://arxiv.org/abs/2511.15690)
    
## Reference

[Deja Vu: 利用上下文稀疏性提升大语言模型推理效率](https://mp.weixin.qq.com/s/CuZiOdzx0ltMdVqPlJz5zA)

[论文笔记：DejaVu、LLM in Flash、PowerInfer](https://zhuanlan.zhihu.com/p/675585887)
