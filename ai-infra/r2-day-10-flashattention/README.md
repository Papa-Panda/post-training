# r2-Day10 — FlashAttention：tiling + online softmax，把 O(N²) 中间量留在片上

## Connection to Prev

r2-Day09 把 GEMM 的 $A/B$ tile 留在 shared memory 里反复复用；r2-Day10 把同一思想推进一步：attention 的中间量 $S=QK^T$ 本是 $N\\times N$、似乎非留不可——online softmax 让 $Q/K/V$ tile 与 running 统计量 $(m,l)$ 在片上流动，$S$ 永不落地 HBM。**牺牲** SRAM 预算、block 尺寸约束、逐 block 的 rescale 开销与反向重算，**换取** HBM 访问从 $\\Theta(Nd+N^2)$ 降到 $\\Theta(N^2d^2/M)$、HBM 驻留从 $O(N^2)$ 降到 $O(N)$；当 $N$ 很小、head dim $d$ 相对 SRAM 很大、或 block bookkeeping 主导流量时，tiling 不赚。

## 1. 标准注意力的 HBM 账

单头、batch=1，$Q,K,V\\in\\mathbb{R}^{N\\times d}$（row-major），$\\tau=1/\\sqrt{d}$：

$$S=\\tau QK^T\\in\\mathbb{R}^{N\\times N},\\qquad P=\\mathrm{softmax}(S)\\ \\text{(row-wise)},\\qquad O=PV\\in\\mathbb{R}^{N\\times d}$$

标准实现把 $S$ 和 $P$ 都写回 HBM。按源码语义数 fp32 scalar payload（theoretical model，不是 DRAM 实测）：

- loads：$Q,K$ 读 $2Nd$（算 $S$）＋ 读 $S$（$N^2$，softmax）＋ 读 $P$（$N^2$）＋ 读 $V$（$Nd$）；
- stores：写 $S$（$N^2$）＋ 写 $P$（$N^2$）＋ 写 $O$（$Nd$）。

$$L_{std}=3Nd+2N^2,\\qquad S_{std}=2N^2+Nd,\\qquad \\text{payload}=4N^2+4Nd\\ \\text{elements}$$

主导项是 $4N^2$：两个 $N\\times N$ 中间矩阵各一写一读。FlashAttention 要消灭的正是这 $4N^2$。

## 2. Tiling：外层 K/V block，内层 Q block

把 $K,V$ 按列切成 $B_c$ 宽的 block（外层循环 $j$），$Q$ 按行切成 $B_r$ 高的 block（内层循环 $i$）。第 $(i,j)$ 步只把 $Q_i\\in\\mathbb{R}^{B_r\\times d}$、$K_j,V_j\\in\\mathbb{R}^{B_c\\times d}$ 搬进 SRAM，在片上算出 $S_{ij}=\\tau Q_iK_j^T\\in\\mathbb{R}^{B_r\\times B_c}$ 并就地消费。输出 $O_i$ 常驻 SRAM 累加，只在最后写回一次。

SRAM 容量约束（元素数，$M$ 为可用 SRAM）：

$$B_r d\\ (Q_i)\\ +\\ 2B_c d\\ (K_j,V_j)\\ +\\ B_rB_c\\ (S_{ij})\\ +\\ B_r d\\ (O_i)\\ \\le\\ M$$

$M$ 越大，$B_r,B_c$ 可取越大，$Q_i/O_i$ 被重复加载的次数 $T_c=\\lceil N/B_c\\rceil$ 越少——这就是论文标题里 "IO-aware" 的含义：block 尺寸按 SRAM 反推，而不是随便切。

## 3. Online softmax：分块算 softmax 还精确等价

Softmax 本来要看完整一行才定分母。Online softmax 维护两个 running 量——行最大值 $m$ 与归一化子和 $l$，新 block 到来时**重缩放**旧累加量：

$$\\tilde{P}_j=\\exp(S_{ij}-m_j),\\qquad \\tilde{\\ell}_j=\\mathrm{rowsum}(\\tilde{P}_j),\\qquad m_j=\\mathrm{rowmax}(S_{ij})$$

$$m' =\\max(m, m_j)$$

$$\\ell' = e^{m-m'}\\,\\ell\\ +\\ e^{m_j-m'}\\,\\tilde{\\ell}_j,\\qquad O' = \\mathrm{diag}(e^{m-m'})\\,O\\ +\\ \\mathrm{diag}(e^{m_j-m'})\\,\\tilde{P}_j V_j$$

全部 block 跑完后 $O=\\mathrm{diag}(\\ell')^{-1}O'$，顺手得到 $L=m'+\\log\\ell'$（logsumexp，反向重算要用，只 $O(N)$ 存储）。

为什么精确：$\\ell' e^{m'}=\\sum_{\\text{blocks}}\\sum_{k}e^{S_k}$ 恰为 softmax 分母，$O' e^{m'}=\\sum_k e^{S_k}v_k$ 恰为分子；相除即标准 $\\mathrm{softmax}(S)V$。第一次访问某行时 $m=-\\infty$，$e^{-\\infty}=0$ 自动丢弃旧累加量——代码里不需要特判。

## 4. 可手算例子

$N=4,\\ d=2,\\ B_r=B_c=2$，为手算清晰取 $\\tau=1$（生产取 $1/\\sqrt{d}$，两条代码路径都透传 `scale` 参数）：

$$Q=\\begin{bmatrix}1&0\\\\0&1\\\\1&1\\\\2&0\\end{bmatrix},\\quad K=\\begin{bmatrix}1&0\\\\0&1\\\\1&1\\\\0&2\\end{bmatrix},\\quad V=\\begin{bmatrix}1&2\\\\3&4\\\\5&6\\\\7&8\\end{bmatrix}$$

标准路径第 0 行：$S$ 第 0 行为 $[1,0,1,0]$，softmax 分子 $[e,1,e,1]$、分母 $2e+2$，

$$O_0=\\frac{e\\cdot(1,2)+1\\cdot(3,4)+e\\cdot(5,6)+1\\cdot(7,8)}{2e+2}\\approx(3.5379,\\ 4.5379)$$

Tiled 路径（只跟第 0 行）：

- $j=0$（$K/V$ 列 0–1）：$S_{00}$ 第 0 行 $[1,0]$，$m=1$，$\\tilde{P}=[1,e^{-1}]$，$\\tilde{\\ell}=1+e^{-1}$，$O_{acc}=(1,2)+e^{-1}(3,4)$；
- $j=1$（列 2–3）：$S_{01}$ 第 0 行 $[1,0]$，$m_{block}=1$，$m'=\\max(1,1)=1$，rescale $e^{1-1}=1$；新贡献 $(5,6)+e^{-1}(7,8)$；$\\ell'=2(1+e^{-1})$；
- 收尾：$O_0=[(6+10e^{-1}),\\ (8+12e^{-1})]\\,/\\,[2(1+e^{-1})]\\approx(3.5379,\\ 4.5379)$，与标准路径一致。

完整 $4\\times4$ 两条路径最大绝对差 $8.88\\times10^{-16}$（CPU 双精度，见测试）。

流量账（本课 theoretical payload model）：标准版 $4N^2+4Nd=96$ 个 fp32 元素；tiled 版按论文 Algorithm 1 的 HBM 访存（每步加载 $Q_i,O_i,m_i,\\ell_i,K_j,V_j$、写回 $O_i,m_i,\\ell_i$）共 $112$ 个元素。**玩具尺寸下 tiled 反而略高**——$B_r d=4$ 太小，bookkeeping 占比主导。这正是第 6 节的"何时不赚"：主导项口径下 tiled 的 HBM 量级约为 $2N^2d(1/B_c+1/B_r)$，只有当 $d(1/B_r+1/B_c)<2$（theoretical estimate）时才赚；$d$ 相对 SRAM 越大、block 越小，越不赚。

## 5. 可执行代码

### CPU：两条路径真正执行，计数来自被执行的数据流

`flash_models.py` 的 `standard_attention` 实际 materialize $S$；`flash_attention` 实际执行外层 $K/V$、内层 $Q$ 的分块循环与 online softmax，payload 计数随循环累加（edge tile 按实际尺寸），不是打印步骤。

```bash
python3 ai-infra/r2-day-10-flashattention/flash_models.py
python3 -m unittest discover -s ai-infra/r2-day-10-flashattention -p 'test_*.py' -v
```

### CUDA：两遍 baseline（S 落地版），fused kernel 未交付

`flash_attention.cu` 是**被 FlashAttention 取代的 baseline**：tiled 两遍实现（`S` 写回 HBM → row softmax → `O=PV`），tile 结构复用 r2-Day09 的 shared-memory GEMM 写法，host 侧有 CPU reference 做逐元素校验。它的 HBM 账正是第 1 节的 $4N^2+4Nd$。fused online-softmax CUDA kernel 在本环境无法编译验证，按质量门槛不交付半成品；算法本体以白板推导＋Python 循环模型（与论文 Algorithm 1 同构）交付。

```bash
nvcc -O3 -std=c++17 ai-infra/r2-day-10-flashattention/flash_attention.cu \
  -o /tmp/flash_attention
/tmp/flash_attention 8 4
```

## 6. Block 尺寸、反向与何时不赚

- **正向**：论文结论（引用，非本课实测）——HBM 访问 $\\Theta(Nd+N^2)\\to\\Theta(N^2d^2/M)$，HBM 驻留 $O(N^2)\\to O(N)$，数值精确等价。
- **反向**：前向只存 $O(N)$ 的 $L$（logsumexp）；反向用 $Q,K,V,O,dO,L$ 在 SRAM 里**重算** $S_{ij}$ 分块，既省显存又多花一次 HBM 遍历——又一次"牺牲换取"。
- **何时不赚**：$N$ 很小（tiling 开销主导，如本课玩具）；$d$ 相对 SRAM 很大时 $B_r,B_c$ 被迫取小、$Q_i/O_i$ 反复加载吃掉收益（这也是生产用 fp16/bf16、FA2/FA3 做 warp-specialization 与更大有效 $M$ 的动机之一）；非 attention-bound 的模型或 kernel。

这也直接连接 Day11 Triton：手写 fused kernel 的工程形态，以及 Day12 用 Nsight 验证"到底省在哪"。

## 状态

- 已验证：Python 语法；10 个 CPU 单元测试（手算行 $O_0$、$8.88\\times10^{-16}$ 精确等价、logsumexp 与直接稳定计算一致到 12 位、非整除/单 block/scale/$1\\times1$/非法输入）；标准 vs tiled 的 theoretical payload 模型 $96$ vs $112$ 元素。
- **execution not validated on CUDA/H100 / 待H100验证**：环境无 `nvcc`/CUDA GPU；`.cu` 未编译、kernel 正确性、真实 HBM 流量、相对标准实现的加速比均未验证。论文的 $\\Theta(N^2d^2/M)$ 是引用结论，不是本课测量。
- 本课状态保持 `blocked`；没有声称 loss、耗时、带宽、comm%、MFU、设备拓扑或 benchmark speedup。

## 原始 / 官方来源

- FlashAttention 论文（Dao et al., 2022），tiling + online softmax 与 IO 复杂度：<https://arxiv.org/abs/2205.14135>
- Online normalizer 计算（softmax 分块归一化的原始出处，论文引用）：<https://arxiv.org/abs/1805.02867>
- FlashAttention-2（并行度与 warp 分工改进）：<https://arxiv.org/abs/2307.08691>
