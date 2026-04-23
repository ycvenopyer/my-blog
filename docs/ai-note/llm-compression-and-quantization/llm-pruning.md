# LLM Pruning

剪枝通过移除冗余权重或结构单元来压缩模型：与量化侧重降低数值精度不同，剪枝侧重减少参与计算的参数或改变结构。关于pruning的论文合集你可以看[awesome-pruning](https://github.com/hrcheng1066/awesome-pruning)。

## 1.剪枝的基础原理

### 1.1 为什么需要剪枝？

- 大规模语言模型往往过参数化，存在可移除的冗余权重。我们可以将大模型分为训练和推理两个阶段，训练阶段是根据数据学习模型中的参数（对MLP来说主要是网络中的权重）；推理阶段中将新数据给模型，经过计算得出结果。过参数化是指训练阶段我们需要大量的参数来捕捉数据中的微小信息，而到了推理阶段则并不需要那么多参数，因此就可以在部署前对模型进行简化。
  
- 在可控精度损失下，剪枝可减小参数量与理论计算量，使得计算实践更少，功耗更小。同时对设备要求变低，可以放在更低端的设备上跑。

- 也有pruning后精度提高的，说明原模型overfit了，pruning起到了regularization的作用。
  
### 1.2 分类

| 分类维度 | 类型 | 说明 | 硬件友好度 |
| :--- | :--- | :--- | :--- |
| 粒度 | 非结构化剪枝 | 任意位置移除单个权重，得到稀疏矩阵 | 低：依赖稀疏 GEMM（通用矩阵乘）、专用内核或半结构化模式 |
| | 结构化剪枝 | 移除整行/整列、注意力头、FFN 神经元等 | 高：矩阵维度变小，易在通用 GEMM 上加速 |
| 时机 | 训练前剪枝 | 基于初始化或随机掩码 | 较少单独用于 LLM |
| | 训练中剪枝 | 动态调整稀疏掩码、稀疏训练 | 灵活，训练与搜索成本高 |
| | 训练后剪枝 | 对已训练模型剪枝，可选校准与微调 | LLM 场景最常见 |

### 1.3 评估维度

- **重要性分数（importance score）**：如何为每个权重或结构单元打分（幅度、梯度、Hessian 近似、激活相关项等）。
  
- **稀疏度（sparsity ratio）**：被剪除权重的比例，可以定义为层中0参数所占比例；结构化场景则对应移除头/神经元比例等。稀疏度可以预先定义，也可以在剪枝过程中自动化，由pruning算法确定各模块的剪枝比例，而不是一开始固定一个值。
  
- **精度恢复（accuracy recovery）**：裁完后进行fine-tuning可以弥补pruning带来的精度损失，因此很多方法会在pruning后做fine-tuning。比较经典的是training，pruning，fine-tuning三段式。后面两个阶段交替进行，每次pruning后损失的精度可以由后面的fine-tuning来弥补，该过程也称为iterative pruning。评估指标有困惑度，是否需要calibration或LoRA微调等。

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

- **模型能力（capacity）**：主流pruning方法中，被裁剪的部分一般直接丢弃不会再拿回来了，即模型的capacity在iterative pruning的过程中不断减少。如此一旦有参数被不适当地裁剪掉便无法被恢复。近年来在模型压缩过程中保留被裁剪部分能力或者扩充能力的方法不断被提出。

## 2.剪枝方法演进

### 2.1 传统剪枝

- **幅度（magnitude-based）**：剪掉绝对值小的权重，实现简单、代价低，大模型 one-shot 时常弱于利用层间敏感度、梯度或激活信息的方法。《Comparing Biases for Minimal Network Construction with Back-Propagation》提出了magnitude-based的pruning方法，即对网络中每个hidden unit施加与其绝对值相关的weight decay来最小化hidden unit数量。

- **梯度（gradient-based）**：用损失对权重的导数、或一阶/二阶泰勒项估计删掉某个权重/神经元对误差的扰动，与纯幅值法对照；是部分结构化剪枝的基础。

- **二阶导数（Hessian）**：基于损失函数相对于权重的二阶导数（对权重向量来说即Hessian矩阵）来衡量网络中权重的重要程度，然后对其进行裁剪，必要时对剩余权重做补偿；经典代表为 OBD、OBS，在 LLM 上由 SparseGPT、LLM-Surgeon 等延续。
  
- **[Deep Compression](https://arxiv.org/abs/1510.00149)**：采用剪枝 → 量化 → 哈夫曼编码流水线，对当时经典网络AlexNet和VGG进行了压缩，其中对于pruning带来的精度损失，使用了iterative pruning方法进行补偿，可以让精度几乎没有损失。

- **正则化（regularization-based）**：L1、[group sparsity](https://blog.51cto.com/u_15837794/11420483)等，把稀疏性写进目标函数。

- **迭代（iterative pruning）**：多轮剪去一部分 → 短微调 → 再剪，和 one-shot（一次定掩码再校准）相对；传统 CNN 压缩里很常见，和 Deep Compression 的剪完再训叙述一致，部署周期通常长于单次剪枝。

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

## 4.Reference

[深度学习网络模型压缩剪枝详细分析](https://zhuanlan.zhihu.com/p/130645948)

