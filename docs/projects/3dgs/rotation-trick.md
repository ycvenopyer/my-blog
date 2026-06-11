# Restructuring Vector Quantization with the Rotation Trick

> 作者：Christopher Fifty、Ronald G. Junkins、Dennis Duan、Aniketh Iyengar、Jerry W. Liu、Ehsan Amid、Sebastian Thrun、Christopher Ré（斯坦福大学 / Google DeepMind）

> 论文链接：<https://arxiv.org/abs/2410.06424>（ICLR 2025）

> 论文代码：<https://github.com/cfifty/rotation_trick>

> 相关解读：[苏剑林《VQ 的旋转技巧：梯度直通估计的一般推广》](https://spaces.ac.cn/archives/10489)

---

## 1. 背景与动机

### 1.1 VQ 与 STE 的困境

向量量化（Vector Quantization, VQ）维护码本 \(\mathcal{C}=\{q_1,\ldots,q_K\}\)，将编码器输出 \(e\) 映射到最近码字 \(q=Q(e)\)。argmin 不可微，VQ-VAE 用直通估计器（STE）绕过量化层：

\[
\tilde{q} = e + \mathrm{sg}(q - e)
\]

前向 \(\tilde{q}=q\)，反向 \(\partial \tilde{q}/\partial e = I\)，等价于 \(\nabla_e L = \nabla_q L\)：梯度从解码器输入 \(q\) 原样复制到 \(e\)，量化层内的几何关系被完全忽略。

STE 的核心问题：同一 Voronoi 区域内所有 \(e\) 获得相同梯度，与 \(e\) 距 \(q\) 的远近、相对夹角无关。位置信息丢失，易导致码本利用率低、码本崩塌（大量码字范数趋零）以及量化误差 \(\|e-q\|_2^2\) 偏大。

### 1.2 核心目标

提出 rotation trick：前向仍输出码字 \(q\)，反向将 \(\nabla_q L\) 经旋转+缩放变换传到 \(e\)，使 \(q\) 与 \(\nabla_q L\) 的夹角在传到 \(e\) 时保持不变。期望在同一 Voronoi 区域内对不同位置的 \(e\) 给出差异化更新，从而提高码本利用率、降低量化误差，改善 VQ-VAE 训练。

---

## 2. 方法与框架

### 2.1 符号与 VQ 层

| 符号 | 含义 |
|------|------|
| \(x \in \mathcal{X}\) | 输入样本 |
| \(e\) | 编码器输出（待量化向量） |
| \(q = Q(e)\) | 码本 \(\mathcal{C}=\{q_1,\ldots,q_K\}\) 中与 \(e\) 最近的码字 |
| \(\tilde{q}\) | 送入解码器的量化后向量 |
| \(\tilde{x}\) | 解码器重建 |
| \(\mathrm{sg}[\cdot]\) | stop-gradient：前向正常计算，反向梯度视为 0 |

欧氏距离下的码本查找：

\[
Q(q = q_i \mid e) = \begin{cases} 1 & \text{若 } i = \arg\min_j \|e - q_j\|_2 \\ 0 & \text{否则} \end{cases}
\]

该 argmin 将连续空间划分为以各码字为中心的 Voronoi 区域（泰森多边形）：区域内任意 \(e\) 共享同一 \(q\)。

VQ-VAE 总损失（Van Den Oord et al., 2017）：

\[
L = \underbrace{\|x - \tilde{x}\|_2^2}_{L_{\text{ren}}} + \underbrace{\|\mathrm{sg}(e) - q\|_2^2}_{L_{\text{codebook}}} + \underbrace{\beta \|e - \mathrm{sg}(q)\|_2^2}_{L_{\text{commit}}}
\]

- \(L_{\text{ren}}\)：重建损失，梯度经解码器回传；  
  
- \(L_{\text{codebook}}\)：更新码字以跟踪编码器输出（实践中常用 EMA 替代显式梯度）；  
  
- \(L_{\text{commit}}\)：承诺损失，将 \(e\) 拉向被选中的 \(q\)，防止编码器在码字间频繁跳跃；\(\beta\) 通常取 \([0.25, 2]\)。

下文分析梯度穿过量化层时，主要关注 \(L_{\text{ren}}\) 一项（另两项不是解码器的函数）。

### 2.2 STE 与链式法则

反向传播对 \(x\) 的梯度可分解为：

\[
\frac{\partial L}{\partial x} = \underbrace{\frac{\partial L}{\partial q}}_{\text{解码器}}\quad \underbrace{\frac{\partial q}{\partial e}}_{\text{量化层}}\quad \underbrace{\frac{\partial e}{\partial x}}_{\text{编码器}}
\]

量化层 \(Q(\cdot)\) 非光滑，\(\partial q/\partial e\) 无良好定义，梯度无法经此项更新编码器。

直通估计器（STE） 在反向中令 \(\partial q/\partial e = I\)（单位阵），将 \(q\) 处梯度原样复制到 \(e\)：

\[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial q}\, I\, \frac{\partial e}{\partial x}, \qquad \nabla_e L = \nabla_q L
\]

前向写法等价于对量化做参数化：

\[
\tilde{q} = e + \underbrace{(q - e)}_{\text{detach}}
\]

下标 detach 表示截断梯度（stop-gradient，下文记作 \(\mathrm{sg}[\cdot]\)）：前向 \((q-e)\) 仍按原值参与计算，反向时该项视为常数、梯度不经过它。PyTorch 中即 `(q - e).detach()`。

前向 \(\tilde{q}=q\)；反向 \(\partial \tilde{q}/\partial e = I\)。因此 \(\nabla_e L\) 不依赖 \(e\) 在 Voronoi 区域内的位置——无论靠近 \(q\) 还是在区域边界，同区域内所有点获得相同梯度更新，量化操作的位置信息完全丢失。

### 2.3 旋转技巧：STE 的几何推广

设计问题：将 \(\nabla_q L\) 从 \(q\) 传到 \(e\) 时，应保留什么几何性质？

- STE 的答案：保留梯度的方向与幅度（平移梯度）；  
  
- 旋转技巧的答案：保留 \(\nabla_q L\) 与 \(q\) 的夹角（旋转梯度）。

旋转技巧将 STE 推广为一般线性变换 \(G\)：

\[
\tilde{q} = \mathrm{sg}[G]\, e + \mathrm{sg}[q - G e]
\]

前向 \(\tilde{q}=q\)，反向 \(\partial \tilde{q}/\partial e = G\)，即定义了 \(\partial q/\partial e = G\)。当 \(G e = q\) 时，第二项为零，简化为 \(\tilde{q} = \mathrm{sg}[G]\, e\)。

论文取 \(G = \dfrac{\|q\|}{\|e\|}\, R\)，其中 \(R\) 为将 \(e\) 旋转对齐到 \(q\) 的正交矩阵（\(R R^\top = I\)，\(R^{-1}=R^\top\)，\(\det(R)=1\)），\(\|q\|/\|e\|\) 将 \(e\) 缩放到与 \(q\) 相同范数：

\[
\tilde{q} = \underbrace{\frac{\|q\|}{\|e\|}\, R e}_{\text{detach}}
\]

前向：数值上 \(\tilde{q} = q\)，与 STE 输出完全一致。 

反向：\(\dfrac{\partial \tilde{q}}{\partial e} = \dfrac{\|q\|}{\|e\|}\, R\)，随 \(e\) 在 Voronoi 区域内的位置变化；\(\nabla_q L\) 传到 \(e\) 时，\(q\) 与 \(\nabla_q L\) 的夹角得以保持，且 \(\|\nabla_e L\| / \|\nabla_q L\| = \|e\| / \|q\|\)（模长比与 \(e\)、\(q\) 的范数比一致）。

构造 \(R\)：先将 \(e,q\) 归一化为 \(\hat{e}=e/\|e\|\)，\(\hat{q}=q/\|q\|\)。在 \(\hat{e}\) 与 \(\hat{q}\) 张成的平面内，从 \(\hat{e}\) 到 \(\hat{q}\) 的旋转矩阵为（\(\theta\) 为 \(e\) 与 \(q\) 夹角）：

\[
R = I + 2\hat{q}\hat{e}^\top - 2\left(\frac{\hat{q}+\hat{e}}{\|\hat{q}+\hat{e}\|}\right)\left(\frac{\hat{q}+\hat{e}}{\|\hat{q}+\hat{e}\|}\right)^\top
\]

于是 \(q = \dfrac{\|q\|}{\|e\|}\, R e\)。实现时须先对 \(\hat{e}\)、\(\hat{q}\)、\(\|q\|/\|e\|\) detach，再计算 \(G e\)。

为何 detach \(R\) 与 \(\|q\|/\|e\|\)：完整求导时 \(\tilde{q} = f(e)\, e\)，\(f(e)\) 内含不可微的 \(Q(e)\)，对 \(e\) 求导会出现 \(f'(e)\, e\) 项而无法计算。丢弃该项、近似 \(\partial \tilde{q}/\partial e \approx f(e) = G\)，比 STE 的 \(I\) 传递了更多量化几何信息。

算法流程：

1. \(e \leftarrow \mathrm{Encoder}(x)\)，\(q \leftarrow Q(e)\)；  
   
2. 计算将 \(e\) 对齐到 \(q\) 的 \(R\)；  
   
3. \(\tilde{q} \leftarrow \mathrm{sg}\bigl[\frac{\|q\|}{\|e\|} R e\bigr]\)，送入解码器；  
   
4. 反向时 \(R\)、\(\|q\|/\|e\|\) 视为常数。

STE 与旋转技巧可视为“兄弟”方法：二者都绕过不可微量化，但选择保持梯度的不同几何性质。

### 2.4 夹角保持

设 \(\|e\|=\|q\|=1\)，则 \(q = e R^\top\)，\(\partial q/\partial e = R\)。编码器处梯度：

\[
\nabla_e L = \nabla_q L \,\frac{\partial q}{\partial e} = \nabla_q L \, R
\]

设 \(\theta\) 为 \(q\) 与 \(\nabla_q L\) 的夹角，\(\varphi\) 为 \(e\) 与 \(\nabla_e L\) 的夹角。由内积：

\[
\|\nabla_q L\|\cos\theta = q[\nabla_q L]^\top = e R^\top [\nabla_q L]^\top = e[\nabla_e L]^\top = \|\nabla_q L\|\cos\varphi
\]

故 \(\theta = \varphi\)：梯度传到 \(e\) 时，与码字之间的夹角不变。

### 2.5 Voronoi 区域分析

设 \(\theta\) 为 \(e\) 与 \(q\) 的夹角，\(\varphi\) 为 \(q\) 与 \(\nabla_q L\) 的夹角。有损压缩中，VQ 希望在失真 \(\|e-q\|_2^2\) 低的同时信息容量（码本利用率）高。

| 条件 | 旋转技巧相对 STE 的行为 | 效果 |
|------|-------------------------|------|
| \(-\pi/2 < \varphi < \pi/2\)（\(\nabla_q L\) 与 \(q\) 同向） | 与 \(q\) 角距离大的 \(e\) 比 STE 被推得更远 | 边界点可跨入未使用码字的 Voronoi 区域，提高码本利用率 |
| \(\pi/2 < \varphi < 3\pi/2\)（\(\nabla_q L\) 与 \(q\) 反向） | 同区域内点间距离减小，被拉向码字 | 降低量化误差，使编码器输出逐步收敛并稳定贴近对应码字 |

STE 对同 Voronoi 区域内所有点施加相同更新，区域内点间相对距离不变。旋转技巧则形成“推—拉”效应：

- 推：角距离大的 \(e\) 被外向梯度推向其他（可能未使用的）码本区域；  
  
- 拉：指向中心的梯度将松散聚集在码字周围的 \(e\) 拉向目标码字。

二者同时服务于 VQ 的两个目标：提高信息容量、降低失真。

局限：

- 当 \(\|e\|\approx 0\) 或 \(\|q\|\approx 0\) 时，\(e\) 与 \(q\) 可能成钝角，旋转技巧会“过度旋转”，\(\nabla_e L\) 与 \(\nabla_q L\) 方向相反，性能可能劣于 STE；  
  
- 旋转以原点为中心，欧氏 VQ 类似无中心的 K-Means，存在平移不变性差异；当所有点远离原点时，夹角 \(\hat{\theta} \to 0\)，\(R \to I\)，\(\|\hat{q}\|/\|\hat{e}\| \to 1\)，旋转技巧平滑退化为 STE。

---

## 3. 实验与结果

### 3.1 设置

在 11 种 VQ-VAE 范式上对比 STE 与 rotation trick，除 \(\partial q/\partial e\) 外架构、超参、训练设置相同。涵盖：

- ImageNet VQ-VAE（Van Den Oord et al. 设置）；  
  
- VQGAN（自回归 / 潜扩散两种范式）；  
  
- ViT-VQGAN；  
  
- TimeSformer 视频 VQ-VAE。

评估：重建 FID（r-FID）、重建 IS（r-IS）、码本使用率、量化误差 \(\|e-q\|_2^2\)。

### 3.2 主要结果

VQ-VAE（ImageNet，潜扩散 VQGAN 设置，8192 码本）：

| 方法 | r-FID ↓ | r-IS ↑ | 码本使用率 | 量化误差 |
|------|---------|--------|------------|----------|
| STE | 5.0 | 141.5 | 2% | 基准 |
| Rotation Trick | 1.1 | 200.2 | 27% | ↓ 两个数量级 |

典型改善（跨 11 种设置）：

- 量化误差常降一个数量级；  
  
- 码本使用率显著提高；  
  
- r-FID、r-IS 一致改善；  

视频 VQ-VAE：STE 训练出现码本崩塌；rotation trick 稳定，重建 r-FVD 更优。

---

## 4. 讨论

### 4.1 理论视角

论文附录从微分几何将 STE 与 rotation trick 统一为平行传输（沿 \(q \to e\) 的曲线移动梯度）：

- STE：在笛卡尔坐标、单位度量、Levi-Civita 联络下传输，梯度场沿曲线为常数（保持方向与幅度）；  
  
- 旋转技巧：在归一化超球坐标下传输，梯度相对坐标基保持常分量，对应笛卡尔系中按对齐 \(q \to e\) 的旋转 \(R_{q\to e}\) 变换：\(\nabla_e L = R_{q\to e}\,\nabla_q L\)。

二者是同一平行传输在不同坐标系下的两种实现；差别在于 STE 保持梯度方向，旋转技巧保持梯度与码字的夹角。

### 4.2 实践注意点

苏剑林指出：

- 旋转技巧多出 \(\|q\|/\|e\|\) 缩放；若初始化时 \(\|q\|\ll\|e\|\)，重构损失梯度会被压低，承诺损失占主导，可能导致码表坍缩——需重新调 \(\beta\) 或用 K-means 初始化码本使范数量级匹配。  
  
- 旋转以原点为中心，与欧氏 VQ 的平移不变性存在张力；对余弦距离 VQ 更自然，但论文对欧氏距离同样有效。  
  
- 并非所有 VQ-VAE 代码无脑替换都能提升，需结合初始化与超参验证。
  
---

## 5. 总结

Rotation Trick 通过旋转+缩放将编码器输出平滑对齐到码字，前向不变、反向保持 \(q\) 与梯度的夹角，从而改变 Voronoi 区域内各点的更新方式。在 11 种 VQ-VAE 设置上显著改善 r-FID、码本利用率与量化误差。作为 STE 的即插即用替代，前向输出不变、计算开销可忽略，在码本利用率低或训练不稳定时值得尝试；实际效果仍依赖码本初始化与超参设置。
