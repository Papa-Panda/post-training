# r2-Day15 NOTES — 术语、公式核对与上一课/下一课链接

## 术语（准确定义）

- **Tensor Parallelism (TP)**：把单个 GEMM 的权重矩阵按输入/输出维度切分到 $T$ 个 rank。
  列并行（split 输出维）后接行并行（split 输入维），中间只需要一次同步：
  列并行的输出是分片的，直接喂给行并行做局部乘加，最后一次 all-reduce 求和。
- **Pipeline Parallelism (PP)**：把 $L$ 层按深度切成 $P$ 个 stage，
  $m$ 个 microbatch 流水通过。stage 之间只传激活（forward）与激活梯度（backward）。
- **Data Parallelism (DP)**：r2-Day04 的 DDP / r2-Day14 的 ZeRO 维度——每个 `(t,p)`
  位置复制一份数据分片，梯度做 all-reduce（或 ZeRO 切分）。
- **Sequence Parallelism (SP)**：沿序列维 $S$ 切分 activation；活在 TP 域内部，
  把 attention 尾部的 all-reduce 换成 reduce-scatter（forward）/ all-gather（backward）。
- **Microbatch**：一个全局 batch 切成 $m$ 份，逐份过 pipeline； $m \gg P$ 是 PP 的生命线。
- **Bubble**：pipeline 排空/注水阶段的空闲，占比 $(p-1)/(m+p-1)$ （GPipe 调度）。
- **1F1B (one-forward-one-backward)**：每个 stage 交替做 1 个 forward 和 1 个 backward，
  在飞的 microbatch 数从 $m$ 降到约 $p$ ，激活内存大减，bubble 公式不变。
- **Interleaved 1F1B**：每个 rank 再虚拟切成 $v$ 个小 stage，bubble 近似降为
  $(p-1)/(m\cdot v + p - 1)$ ，代价是点对点通信次数 $\times v$ 。

## 公式核对（与 r2-Day03/Day14 一致性）

- ring all-reduce 单 rank 流量 $= 2(N-1)/N\cdot M$ ：与 r2-Day03 的
  $2(N-1)/N\times$ 数据量同一公式，本课代码 `allreduce_ring_bytes` 直接复用该口径。
- TP 每层每 microbatch 4 次 all-reduce：forward（attn out-proj、mlp fc2）
  2 次 + backward 2 次。注意 backward 的 reduce 对象是输入梯度 `dX` ，
  payload 同样是 $b\cdot S\cdot H$ 个元素——不是梯度 `dW` （ `dW` 是局部算好、
  只在 DP 维同步的）。
- PP 点对点量：每边界每 microbatch forward 传激活 $b\cdot S\cdot H$ 元素 + backward 传梯度
  $b\cdot S\cdot H$ 元素，共 $2\cdot b\cdot S\cdot H\cdot 2\ \text{B}$ 。
- DP 维的梯度同步量走 r2-Day14 的 ZeRO 口径（本课不重复）。

## DeepSpeed Ulysses vs Megatron SP（对比，鸟瞰级）

- Megatron SP：沿 $S$ 切，attention 内部用 reduce-scatter/all-gather 保持分片。
- Ulysses：沿 **head 维** 做 all-to-all，把"按 head 分片"转成"按 seq 分片"再算 attention。
  通信量都是 $O(b\cdot S\cdot H)$ 量级，实现位置不同；Ulysses 对 GQA/MQA 更友好
  （head 数少时 Megatron 式按 head 切会切不动——r2-Day13 的 GQA 在这里回响）。

## 与上一课的连接

r2-Day14 回答"model states 太大怎么办"（切 optimizer/grad/param，通信换显存）。
r2-Day15 回答"模型本体太大怎么办"（切张量/层/序列）。两者正交且常叠加：
64 卡例子中 DP=2 那一维内部，完全可以再套 ZeRO-1 切 optimizer states
（这就是 DeepSpeed / FSDP + TP/PP 混用的现实形态，r2-Day17 选型课展开）。

## 下一课预告

r2-Day16：混合精度（BF16 为什么指数位 8 位、尾数 7 位就够）/ 梯度累积 /
Activation Checkpointing——用**计算**换显存的第三条路，
与 Day14（通信换显存）、Day15（切模型换规模）形成完整的不可能三角。
