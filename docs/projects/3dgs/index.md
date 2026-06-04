# 3D/4D GS 轻量化 — 码本向量量化

## 1.项目要解决什么

新视角合成（Novel View Synthesis, NVS）：从多视角图像重建三维场景，并在任意新相机位姿下渲染照片级真实感图像。

3D Gaussian Splatting（3DGS）在质量与速度上已接近或超过 NeRF，常见桌面 GPU 可达 100+ FPS。但高质量场景往往有百万级高斯、单场景数百 MB～1 GB+ 存储，在移动端、VR/AR、云渲染与流媒体部署中成为瓶颈。4D 动态扩展会进一步放大存储与带宽压力。

---

## 2.NVS（新视角合成）

NVS 不是对输入视角做简单插值，而是学习场景的辐射场（Radiance Field）表示：把整段场景建成一个可查询的光场模型，而不是只保存几张输入图。

辐射场表示是什么意思？可以把它理解成对空间中每一点、朝每一方向的光学性质的描述：

- 有没有物质或者叫不透明度（NeRF 里常写成体积密度 σ 或 不透明度 α；3DGS 里是每个高斯的不透明度）； 
   
- 朝该方向看起来是什么颜色（NeRF 的辐射度 \(c(\mathbf{x}, \mathbf{d})\) 可随视角 \(\mathbf{d}\) 变化；3DGS 用球谐 SH 编码视角相关颜色）。

有了这样一个场，渲染任意新视角时：从相机发出一条射线，沿射线采样多个 3D 点，向场查询各点的颜色与不透明度，再按体渲染 / α 混合合成该像素——新视角来自对场的查询，而不是在已有照片之间做像素插值。

NeRF 用 MLP 隐式场 \(f(\mathbf{x},\mathbf{d})\mapsto(\sigma,c)\) 存这个场；3DGS 用百万个显式高斯近似，再用光栅化求像素。二者目标都是 NVS，差别主要在怎么存这个场以及如何计算。

流程：

- 采集：多视角照片或视频 + 相机内外参 + 稀疏 3D 点。  

!!! note 多视角照片采集
    SfM（Structure-from-Motion，运动恢复结构）：从多张有重叠的图像反推每张图相机在哪、朝哪看，并恢复一批稀疏 3D 特征点（角点、纹理块等），是经典多视图几何流程。  

    COLMAP：常用的开源 SfM/MVS（Multi-View Stereo） 工具（[colmap.github.io](https://colmap.github.io/)）。典型用法是输入图像文件夹，自动做特征匹配、增量式重建与光束法平差（Bundle Adjustment），输出相机位姿、内参和稀疏点云（`.ply` / `cameras.bin` 等）。  
   

- 表示：隐式场（NeRF）或显式基元（高斯点云）。  
   
- 优化：可微渲染 + 图像重建损失（L1、SSIM 等）。  
   
- 评估：PSNR、SSIM、LPIPS；部署时还需 FPS 与 模型体积（MB）。

---

## 3.NeRF（神经辐射场）

NeRF（[Mildenhall et al., ECCV 2020](https://arxiv.org/abs/2003.08934)）用 MLP 将空间位置 x 与视角 d 映射为体积密度 σ 与辐射度 c，沿射线采样后按体渲染公式 α 混合得到像素颜色，端到端优化网络权重。

离散形式的像素颜色：

\[
C(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot (1 - e^{-\sigma_i \delta_i}) \cdot c_i,\quad
T_i = \prod_{j=1}^{i-1} e^{-\sigma_j \delta_j}
\]

特点：画质高，但每条射线需大量 MLP 查询，训练与推理都慢，难以满足实时渲染要求。

---

## 4.3DGS（三维高斯溅射）

Kerbl 等（[SIGGRAPH 2023](https://arxiv.org/abs/2308.04079)）提出 3D Gaussian Splatting for Real-Time Radiance Field Rendering，首次在完整无界场景、1080p 下实现高质量实时新视角合成。与 NeRF 在空域逐点采样 + MLP 查询不同，原论文把场景建成百万量级的显式三维高斯，再用可微光栅化合成图像。

### 4.1 单高斯参数

| 属性 | 维度 | 说明 | 公式表示 |
|------|------|------|----------|
| 位置 \(\mathbf{p}\) | 3 | 世界坐标 | \(\mathbf{p}_n \in \mathbb{R}^3\) |
| 缩放 \(\mathbf{s}\) | 3 | 对数空间优化 | \(S_n = \mathrm{diag}\bigl(e^{s_{n,1}}, e^{s_{n,2}}, e^{s_{n,3}}\bigr)\) |
| 旋转 \(\mathbf{q}\) | 4 | 单位四元数 | \(\mathbf{q}_n \in \mathbb{R}^4,\ \|\mathbf{q}_n\|=1;\ \Sigma_n = R(\mathbf{q}_n)\, S_n\, S_n^\top R(\mathbf{q}_n)^\top\) |
| 不透明度 | 1 | sigmoid 激活 | \(\alpha_n = \sigma(o_n) = 1/(1+e^{-o_n})\)，\(o_n\) 为可学习 logit |
| 颜色 SH | 48 | 3 阶球谐，视角相关 | \(c_n(\mathbf{d}) = \sum_{\ell=0}^{3}\sum_{m=-\ell}^{\ell} \mathbf{h}_n^{\ell,m}\, Y_\ell^m(\mathbf{d})\)，RGB 各 16 系数共 48 维；\(\mathbf{d}\) 为视线单位向量 |

投影到屏幕后与 NeRF 类似做 α 混合：

\[
C(x) = \sum_{k} c_k \alpha_k \prod_{j<k}(1-\alpha_j),\quad
\alpha_n(x) = o_n \exp\!\left(-\tfrac{1}{2}(x-p'_n)^\top \Sigma'^{-1}_n (x-p'_n)\right)
\]

### 4.2 主要缺点：存储与内存占用大

3DGS 渲染快、画质好，但表示是逐高斯显式存盘，不像 NeRF 主要存一份 MLP 权重（通常仅数 MB～数十 MB）。高质量静态场景导出为 `.ply` 时，常见体积为数百 MB 至 1 GB 以上，主要瓶颈如下。

1. 高斯数量多：自适应致密化为拟合细节会持续增点，场景可达 \(10^6\) 量级，许多点体积小、不透明度低，对画质贡献有限却仍占存储。  
   
2. 每点参数量大：每点约 \(3+3+4+1+48=59\) 个标量。  
   
3. SH 占大头：48 维颜色系数。  
   
4. 部署侧代价：大文件拉长加载时间、占用显存/内存，云渲染与移动端传输也受带宽限制。

---

## 5.4DGS（动态 / 时空高斯）

静态 3DGS 无法直接表达运动。Spacetime Gaussians（STG）（[Li et al., CVPR 2024](https://arxiv.org/abs/2312.16812)）是本项目动态基线：不逐帧存满属性，而用基线加上时间多项式或者叫 TRBF 在渲染时刻 \(t\) 在线计算属性，再用与 3DGS 相同的光栅化。

| 时变属性 | 典型表示 |
|----------|----------|
| 位置 p(t) | 基线 + \(\sum u_k (t-\mu)^k\) |
| 旋转 r(t) | 基线 + 多项式系数 |
| 颜色 | 低维特征 + MLP（空间 / 视角 / 时间分解） |

---

## 6.CodeBook 与 CodeWord

码本（CodeBook）与码字（CodeWord）在计算机视觉与信号压缩里反复出现：核心都是用少量代表向量（码字）概括大量相似向量，存储时只记属于哪个码字的编号，而不是存完整浮点参数。

### 6.1 基本定义

| 术语 | 含义 | 记号示例 |
|------|------|----------|
| 码本 CodeBook | 有限个代表向量的集合，相当于字典 | \(C = \{c_1, c_2, \ldots, c_K\}\)，\(c_k \in \mathbb{R}^D\) |
| 码字 CodeWord | 码本中的一个元素，即一个典型模式 | 第 \(k\) 个码字 \(c_k\) |
| 编码 / 量化 | 把输入向量 \(\mathbf{x}\) 映射到最近（或最匹配）的码字 | 索引 \(i = \arg\min_k \|\mathbf{x} - c_k\|\) |
| 解码 / 反量化 | 用码字近似原向量 | \(\hat{\mathbf{x}} = c_i\) |

若码本大小为 \(K\)，每个向量只需 \(\lceil \log_2 K \rceil\) bit 的索引。例如 \(K=64\) 时每个属性仅需 6 bit，远小于 FP32 的 32 bit×维度——3DGS 百万高斯上，这正是 Compact 3DGS 等对 scale、rotation 做 R-VQ 的动机。

### 6.2 码本如何得到

常见构造方式：

| 方法 | 做法 | 特点 |
|------|------|------|
| K-means | 在训练集所有高斯属性向量上聚类，中心作码字 | 简单；需配合 STE 等做端到端微调 |
| VQ-VAE | 编码器 + 可学习码本 + 解码重建 | 码本与渲染目标可联合优化 |
| R-VQ | 第一级量化残差，再对残差建下一级码本 | Compact 3DGS 几何属性采用；压缩率更高 |
| 在线 CodeBook | 来一个新样本，匹配或新建码字 | 适合时序像素；3DGS 更常用离线/周期性更新码本 |

---

## 7.VQ（向量量化）

向量量化将连续向量 \(\mathbf{x} \in \mathbb{R}^D\) 映射到码本 \(C=\{c_1,\ldots,c_K\}\) 中最近码字：

\[
q(\mathbf{x}) = \arg\min_{c_k \in C} \|\mathbf{x} - c_k\|_2
\]

存储时每个高斯只保留 \(\log_2 K\) bit 的索引，而非完整 float 向量。训练常用 VQ-VAE 风格损失 + 直通估计器（STE） 缓解 argmin 不可导：

\[
L_{VQ} = \|sg[\mathbf{x}] - c_k\|^2 + \beta \|\mathbf{x} - sg[c_k]\|^2
\]

残差向量量化（R-VQ）：多级码本级联，逐级量化残差。第 \(l\) 级：

\[
\hat{r}_n^l = \sum_{j=1}^{l} Z^j[i_n^j],\quad
i_n^l = \arg\min_k \left\| Z^l[k] - (r_n - \hat{r}_n^{l-1}) \right\|
\]
