# MoE Compression

SMoE（Sparse Mixture of Experts）的显存与部署压力主要来自专家总参数量与加载，与稠密 FFN 剪枝的设定不同。近年工作多围绕：专家级或神经元级删除/重分配、用路由/激活指引剪专家内部或整条专家、子空间/输出视角的专家合并等。

近年来相关论文如下：

1. [Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2402.14800)
2. [MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router](https://arxiv.org/abs/2410.12013)
3. [Delta Decompression for MoE-based LLMs Compression](https://arxiv.org/abs/2502.17298)
4. [Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2504.07807)
5. [Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266)
6. [Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2509.10377)
7. [Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](https://arxiv.org/abs/2510.02345)
8. [REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](https://arxiv.org/abs/2510.13999)
9. [MergeMoE: Efficient Compression of MoE Models via Expert Output Merging](https://arxiv.org/abs/2510.14436)
10. [Does a Global Perspective Help Prune Sparse MoEs Elegantly?](https://arxiv.org/abs/2604.06542)
11. [μ-MoE: Test-Time Pruning as Micro-Grained Mixture-of-Experts](https://arxiv.org/abs/2505.18451)
12. [MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](https://arxiv.org/abs/2511.15690)
13. [SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation](https://arxiv.org/abs/2506.18349)
14. [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334)
15. [Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270)
