# LLM Pruning

剪枝通过移除冗余权重或结构单元来压缩模型：与量化侧重降低数值精度不同，剪枝侧重减少参与计算的参数或改变结构。关于pruning的论文合集你可以看[awesome-pruning](https://github.com/hrcheng1066/awesome-pruning)。

## 1.剪枝的基础原理

### 1.1 为什么需要剪枝？

- 大规模语言模型往往过参数化，存在可移除的冗余权重。
  
- 在可控精度损失下，剪枝可减小参数量与理论计算量，并为后续量化、蒸馏等步骤腾出空间。
  
- 与量化不同：剪枝直接改变权重，而非仅改变数值表示。

### 1.2 分类

| 分类维度 | 类型 | 说明 | 硬件友好度 |
| :--- | :--- | :--- | :--- |
| 粒度 | 非结构化剪枝 | 任意位置移除单个权重，得到稀疏矩阵 | 低：依赖稀疏 GEMM（通用矩阵乘）、专用内核或半结构化模式 |
| | 结构化剪枝 | 移除整行/整列、注意力头、FFN 神经元等 | 高：矩阵维度变小，易在通用 GEMM 上加速 |
| 时机 | 训练前剪枝 | 基于初始化或随机掩码 | 较少单独用于 LLM |
| | 训练中剪枝 | 动态调整稀疏掩码、稀疏训练 | 灵活，训练与搜索成本高 |
| | 训练后剪枝 | 对已训练模型剪枝，可选校准与微调 | LLM 场景最常见 |

### 1.3 评估维度

- **重要性分数**：如何为每个权重或结构单元打分（幅度、梯度、Hessian 近似、激活相关项等）。
  
- **稀疏度**：被剪除权重的比例；结构化场景则对应移除头/神经元比例等。
  
- **精度与恢复**：剪枝后困惑度（PPL）、下游任务；是否需要校准或 LoRA/全参微调 以及可接受的性能回退。

!!! note "困惑度"
    在语言建模范式下，困惑度（perplexity, PPL）用来概括模型在一段无标签文本上预测下一个 token 的难易程度：PPL 越低，通常表示模型对语料的平均预测更自信、语言建模样本内拟合更好；是否与下游任务、泛化一致，仍需结合其他指标。

    设语料有 \(N\) 个被计入损失的 token，位置 \(t\) 上模型输出下一 token 的分布，真实出现的 token 为 \(w_t\)，条件概率为 \(p(w_t \mid w_1, \ldots, w_{t-1})\)（也简记为给定上文的 \(p(w_t\!\mid\!\text{context})\)）。平均负对数似然（NLL，按 token 平均）为：

    \[
    NLL = -\frac{1}{N} \sum_{t=1}^{N} \log p\bigl(w_t \mid w_1, \ldots, w_{t-1}\bigr)
    \]

    困惑度定义为：

    \[
    PPL = e^{NLL} = \exp\!\left(-\frac{1}{N} \sum_{t=1}^{N} \log p\bigl(w_t \mid w_1, \ldots, w_{t-1}\bigr)\right)
    \]


## 2.剪枝方法演进

### 2.1 传统剪枝

- **Deep Compression**：剪枝 → 量化 → 霍夫曼编码等流水线，强调剪枝后需训练恢复的经典思路；常与下面迭代式一起出现。

- **幅度剪枝（magnitude-based）**：剪掉绝对值小的权重，实现简单、代价低，大模型 one-shot 时常弱于利用层间敏感度、梯度或激活信息的方法。

- **梯度 / 泰勒式（gradient- / Taylor-based）**：用损失对权重的导数、或一阶/二阶泰勒项估计删掉某个权重/神经元对误差的扰动，与纯幅值法对照；是部分结构化剪枝的基础。

- **二阶 / 曲率（Hessian, OBD / OBS 等）**：用（近似）Hessian 或平方灵敏度衡量删权代价，必要时对剩余权重做补偿；经典代表为 OBD、OBS，在 LLM 上由 SparseGPT、LLM-Surgeon 等延续。

- **基于激活 / 通道（activation- / channel-based）**：用 BN 缩放因子、激活稀疏度、APoZ 等统计量定通道/滤波器/神经元重要度，易导向结构化、硬件友好的裁剪。

- **稀疏正则（regularization-based）**：L1、group sparsity、变分稀疏等，把稀疏性写进目标函数，与显式打分 + 硬掩码 + 再训练可对照理解，也常与量化、蒸馏同训。

- **信息论 / 最小描述长度（information-theoretic, MDL）**：从互信息、编码长度等角度选子网络或权值，与经验风险最小化并列为一条理论线，工程里不如幅值/梯度法常见。

- **迭代 / 渐进剪枝（iterative / gradual）**：多轮剪去一部分 → 短微调 → 再剪，和 one-shot（一次定掩码再校准）相对；传统 CNN 压缩里很常见，和 Deep Compression 的剪完再训叙述一致，部署周期通常长于单次剪枝。

### 2.2 LLM 剪枝的代表性方法

| 方法 | 类型 | 核心思想 | 是否需要微调 |
| :--- | :--- | :--- | :--- |
| [**OBD（Optimal Brain Damage）**](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) | 非结构化（经典） | 用对角近似 Hessian 估计删去单个权重对损失的影响 | 通常需要重训练或后续微调 |
| [**OBS（Optimal Brain Surgeon）**](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) | 非结构化（经典） | 在 OBD 上引入更完整的二阶信息，并在剪除时做权重补偿 | 常需与再训练/微调配合 |
| [**SparseGPT**](https://arxiv.org/abs/2301.00774) | 非结构化 | 借鉴 OBS 类思路，在单次剪枝中用近似二阶信息做权重更新，减轻全量重训练依赖 | 常配合少量校准数据，不依赖大规模梯度训练 |
| [**Wanda**](https://arxiv.org/abs/2306.11695) | 非结构化 | 用权重与输入激活（如行范数）构造重要性，无需 Hessian | 一般无需微调即可保持较好 PPL |
| [**LLM-Pruner**](https://arxiv.org/abs/2305.11627) | 结构化 | 基于梯度等信号识别可移除的耦合结构（头、神经元等） | 通常需要 LoRA 等快速微调 |
| [**LLM-Surgeon**](https://arxiv.org/abs/2312.17244) | 非结构化/半结构化/结构化 | 将 Kronecker 分解等曲率近似扩到 LLM 规模，联合决定删谁与对剩余权重的更新 | 剪枝中已含权重复原式更新；是否再训看目标与算力 |
| [**DISP-LLM**](https://arxiv.org/abs/2410.11988) | 结构化 | 面向维度的解耦式结构剪枝，用超网络等为各层/块学不同的保留模式，各块宽度可不一致 | 主模型常冻结，以剪枝/辅助模块训练为主 |
| [**Compresso**](https://arxiv.org/abs/2310.05015) | 结构化 | 指令微调阶段结合 LoRA 与 L0/协作式提示，在结构化剪枝下恢复指令跟随与通用能力 | 是（与指令微调/LoRA 同阶段） |
| [**FLAP**](https://arxiv.org/abs/2312.11983) | 结构化 | 基于输出特征图可恢复性等波动式重要性，加偏置补偿，追求免重训的硬件友好裁剪 | 一般无需全量重训 |
| [**LoRAPrune**](https://arxiv.org/abs/2305.18403) | 结构化 | 用 LoRA 权重/梯度作重要性，避免在冻结主干上存完整权重梯度，迭代剪通道/头 | 依赖 LoRA 式迭代与合并流程 |
| [**LoRAP**](https://arxiv.org/abs/2404.09695) | 结构化（差异化） | 对子层分治：MHA 侧用激活加权的低秩/谱分解，FFN 侧用无梯度通道剪枝 | 按论文设定有对应微调/压缩阶段 |
| [**SlimGPT**](https://arxiv.org/abs/2412.18110) | 结构化 | 在 OBS 思路上做层内批量贪心、动态度与增量剪枝率，控累积误差、偏快速近似最优 | 轻量重算或再训依实验设定 |
| [**SliceGPT**](https://arxiv.org/abs/2401.15024) | 结构化 | 利用 Transformer 的计算不变性，通过正交变换后裁掉矩阵行列以缩小隐藏维 | 常无需大规模数据再训（配合变换） |
| [**Týr-the-Pruner**](https://arxiv.org/abs/2503.09657) | 结构化 | 以逐层多稀疏率建超网、再搜全局层间稀疏分布，使总体稀疏满足目标时误差可接受 | 依管线可选继续训练 |
| [**ZipLM**](https://arxiv.org/abs/2302.04089) | 结构化 | 在目标环境下迭代剪掉损失–时延/算力最差的结构单元，与推理方式绑定 | 可一次性或逐步压缩，看设定 |
| [**Sheared-LLaMA**](https://arxiv.org/abs/2310.06694) | 结构化 | 在续训过程中按任务目标做结构化裁减，把大模型剪到小预算再继续预训练/对齐 | 是（续训/剪枝一体） |

### 2.3 重要性评分机制

- **幅度**：\( \text{score} \propto |W| \)（逐元素）。
  
- **SparseGPT**：在 OBS 框架下利用 Hessian 逆的近似（常是对角或块对角），使剪掉权重后的单层误差可控。
  
- **Wanda**（典型形式）：用权重与激活共同刻画敏感度，例如对输出通道维度有 \( \text{score}_j \propto |W_{:,j}| \odot \|X_{:,j}\|_2 \) 一类形式（\(X\) 为校准时的输入激活）。

### 2.4 与 Transformer 结构的关系

- 剪枝对象多为 Q/K/V/O 投影、FFN/SwiGLU 分支；结构化方法常剪MHA或FFN中间维度。
  
- 全参微调昂贵，故 post-training 剪枝常与少量校准方法或LoRA搭配。

## 3.其他剪枝论文

1. [Progressive Gradient Pruning](https://arxiv.org/abs/1906.08746)
2. [Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning](https://arxiv.org/abs/2602.04491)
3. [MoreauPruner: Robust Pruning of Large Language Models against Weight Perturbations](https://arxiv.org/abs/2406.07017)
4. [Beware of Calibration Data for Pruning Large Language Models](https://arxiv.org/abs/2410.17711)
5. [2SSP: A Two-Stage Framework for Structured Pruning of LLMs](https://arxiv.org/abs/2501.17771)
6. [Fast and Effective Weight Update for Pruned Large Language Models](https://arxiv.org/abs/2401.02938)
7. [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406)
8. [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://arxiv.org/abs/2403.07378)
9. [Lightweight and Post-Training Structured Pruning for On-Device Large Language Models](https://arxiv.org/abs/2501.15255)
10. [Efficient Post-Training Pruning of Large Language Models with Statistical Correction](https://arxiv.org/abs/2602.07375)
11. [FASP: Fast and Accurate Structured Pruning of Large Language Models](https://arxiv.org/abs/2501.09412)
12. [You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning](https://arxiv.org/abs/2501.15296)
13. [BlockPruner: Fine-grained Pruning for Large Language Models](https://arxiv.org/abs/2406.10594)
14. [Pruning Large Language Models with Semi-Structural Adaptive Sparse Training](https://arxiv.org/abs/2407.20584)
15. [MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models](https://arxiv.org/abs/2409.17481)
16. [Pruning Foundation Models for High Accuracy without Retraining](https://arxiv.org/abs/2410.15567)
17. [SlimLLM: Accurate Structured Pruning for Large Language Models](https://arxiv.org/abs/2505.22689)
18. [E³-Pruner: Towards Efficient, Economical, and Effective Layer Pruning for Large Language Models](https://arxiv.org/abs/2511.17205)

