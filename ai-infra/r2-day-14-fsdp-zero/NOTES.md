# r2-Day14 NOTES — FSDP/ZeRO：用通信换显存

> 符号约定与 README §0 一致： $P$ 参数量， $N$ rank 数，
> $b_p/b_g/b_o$ 参数/梯度/优化器状态的每参数字节数，
> $M$ 一次 collective 的 payload 字节数。GB 为十进制。

## 1. 术语辨析（准确说法）

- **model states**：训练时必须存的三样——参数（params）、梯度（grads）、
  优化器状态（optimizer states，如 Adam 的 m/v）。ZeRO 切的就是这三样，
  **不切 activations**（那是重计算 r2-Day16 的活）。
- **ZeRO（Zero Redundancy Optimizer）**：DeepSpeed 的数据并行显存优化，
  三个阶段渐进切分。stage 数字越大，切得越多、通信越多。
- **ZeRO-1**：只切 optimizer states。梯度做 reduce-scatter（每卡只拿到
  自己那 $1/N$ 的求和梯度）→ 本地更新自己分片的优化器状态 → all-gather
  更新后的参数。通信量与 DDP 的 all-reduce 相等。
- **ZeRO-2**：在 ZeRO-1 基础上再切梯度（梯度 reduce-scatter 后**不再**
  all-gather 回全量，直接保留分片）。通信量仍与 DDP 相等。
- **ZeRO-3**：再切参数。forward/backward 前 all-gather 凑出全量参数，
  用完即释放（只保留 $1/N$ 分片）。通信量是 DDP 的 $1.5\times$ 。
- **FSDP（Fully Sharded Data Parallel）**：PyTorch 原生的 ZeRO-3 风格实现。
  `FULL_SHARD` ≈ ZeRO-3；`SHARD_GRAD_OP` ≈ ZeRO-2（切梯度+优化器状态，
  参数不切）；`HYBRID_SHARD` = 组内 FULL_SHARD、组间复制（机内切分、
  机间 DDP，省跨机 all-gather）。
- **flat_param**：FSDP 把一个 wrap 单元（如一个 transformer block）的所有参数
  展平成一维再切分——切分粒度是"单元"不是"张量"，减少小消息数量。
- **prefetch / limit_all_gathers**：下一个 block 的 all-gather 与当前 block
  的计算 overlap；同时限制飞行中的 all-gather 数量，防止 prefetch 的
  瞬时 buffer 吃掉省下来的显存。

## 2. $1.5\times$ 的推导（手算）

DDP 每步：all-reduce 梯度，ring 公式 $2(N-1)/N \times M_g$ 。

ZeRO-3 每步（ $M_p$ 参数字节， $M_g$ 梯度字节，通常 $M_p = M_g$ ）：
$$2 \times \frac{N-1}{N} M_p \;\;(\text{两次 all-gather}) \;+\; \frac{N-1}{N} M_g \;\;(\text{一次 reduce-scatter})$$

$M_p = M_g = M$ 时 $= 3(N-1)/N \times M = 1.5 \times [2(N-1)/N \times M]$ 。
这就是 ZeRO 论文的原话：ZeRO-3 的通信量是数据并行的 $1.5\times$ 。
toy 数值（README 例 B）：DDP $600\ \text{B}$ ，ZeRO-3 $900\ \text{B}$ 。

## 3. 为什么 ZeRO-1 的通信"不多花钱"

DDP 的 all-reduce = reduce-scatter + all-gather（ring 实现里本来就是两阶段）。
ZeRO-1 把"all-reduce 梯度"拆成"reduce-scatter 梯度（求和分片）+
all-gather 参数（更新后广播）"——总量都是 $2(N-1)/N \times M$ ，
只是把一次 all-gather 从"梯度"换成了"参数"。所以 ZeRO-1/2 相对 DDP
是**纯赚显存、不加通信**；贵的是 ZeRO-3 那两次参数 all-gather。

## 4. mixed precision 的 $16\ \text{B/param}$ 从哪来

$$2\ (\text{fp16 参数}) + 2\ (\text{fp16 梯度}) + 4\ (\text{fp32 master}) + 4\ (\text{fp32 m}) + 4\ (\text{fp32 v}) = 16$$

fp32 master 的存在是因为：fp16 的精度不够做 $lr \times grad$ 这种小更新累加，
更新必须在 fp32 里做，再 cast 回 fp16。这 $4\ \text{B}$ 是"精度税"，
roadmap 的 $84\ \text{GB}$ 口径把它折掉了（ $12\ \text{B/param}$ ）。
两种口径结论一致：都装不进单张 80GB。

## 5. 与前后课程的连接

- **承 Day13**：Day13 省的是推理侧 decode 的 KV cache；Day14 省的是训练侧的
  model states。两边都在做同一件事：把"不得不存的东西"切小/压小，
  代价分别是"换一套数学"和"多一次通信"。
- **启 Day15（TP/PP/SP）**：ZeRO/FSDP 是"数据并行内部"的切分；当 $N$ 继续放大、
  通信盖不住时，就要把模型本身按张量/流水线切开——那是 Day15 的 3D 并行。
- **启 Day16（混合精度/重计算）**：ZeRO 不碰 activations；长序列训练的
  activation 显存墙靠 activation checkpointing（用计算换显存，
  与今天"用通信换显存"对称）解决。

## 6. 来源

- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models", <https://arxiv.org/abs/1910.02054>（三阶段定义与
  $1.5\times$ 通信结论的原始出处）
- Rajbhandari et al., "ZeRO-Offload", <https://arxiv.org/abs/2101.06840>；
  "ZeRO-Infinity", <https://arxiv.org/abs/2104.07857>（把 states 进一步
  offload 到 CPU/NVMe，本课未展开）
- Zhao et al., "PyTorch FSDP", <https://arxiv.org/abs/2304.11277>
  （flat_param、per-block wrap、prefetch 的工程实现）
- PyTorch FSDP 文档：<https://pytorch.org/docs/stable/fsdp.html>
- DeepSpeed ZeRO 教程：<https://www.deepspeed.ai/tutorials/zero/>
