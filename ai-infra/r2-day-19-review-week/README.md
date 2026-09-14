# r2-Day19 — 复盘周：第二层（分布式训练）收官

## Connection to Prev

r2-Day18（挂了怎么恢复）是第二层的**最后一个生产问题**。
r2-Day19 把 Day14–18 串成一张端到端大账：

- Day14 给模型状态记账：7B = 84 GB（12 B/param），单卡 80 GB 装不下；
- Day15 把模型切开：TP=8 机内 / PP=4 跨机 / DP=2 收尾；
- Day16 压每 rank 峰值：BF16 / 梯度累积 / activation 重计算；
- Day17 选框架装机：Megatron / DeepSpeed / FSDP 一句选型；
- Day18 挂了恢复：DCP 分片写 + Young/Daly 最优间隔。

同时交接第三层：r2-Day20–21 从"**训得动**"转向"**跑得快**"
（TTFT/TPOT & KV Cache 32GB 手算）。

**牺牲**：通信（ZeRO-3 付 1.5× DDP、TP 每层 4 次 all-reduce、
PP bubble 空转）+ 实现复杂度 + checkpoint 写盘 I/O。
**换取**：把 84 GB（12 B/param）装进 $N$ 张卡，挂了能从快照恢复。
**何时不赚**：总账本来就装得下单卡时，ZeRO-3 的 1.5× 通信、
TP 的 4 次 all-reduce 全是纯 overhead； $T_{train} < \tau^*$
时 checkpoint 一次都轮不到，白写。

抄百科类比：第二层 = 合租房。Day14 算清"家具一共多少立方"
（84 GB）；Day15 决定"怎么分房间"（TP 切矩阵/PP 切楼层/DP 复印）；
Day16 "每人只留当季衣服在手边"（压峰值）；Day17 "选个管家"
（框架）；Day18 "贵重物品拍照存档"（checkpoint）；Day19
"搬完家回头看账本：哪笔钱花得值、哪笔是冤枉钱"。

## 0. 符号表（每个字母的含义）

- $P$ ：参数量； $N$ ：参与训练的 GPU 卡数；
  $T/P/D$ ：TP / PP / DP 并行度（ $N = T \times P \times D$ ）
- GB：十进制 $10^9$ （与 r2-Day14/15/16/17/18 口径一致）
- $\delta$ ：一次 checkpoint 写盘耗时（秒）； $M$ ：系统 MTBF（秒）；
  $\tau$ ：checkpoint 间隔（秒）； $W(\tau)$ ：浪费占比
- $b$ ：microbatch 大小； $S$ ：序列长； $H$ ：hidden 维； $L$ ：层数
- $M_{ar}$ ：一次 all-reduce 的 payload 字节数

## 1. 可手算例子 A：显存总账（Day14→16→17→18，同一口径 7B）

| 口径 | 公式 | 7B 总量 | 出处 |
|---|---|---|---|
| 模型状态（fp16+Adam） | $P \times 12$ B | 84 GB（14+14+56） | Day14 |
| AMP 混合精度 | $P \times 16$ B | 112 GB（14+14+28+56） | Day16 |
| checkpoint 快照 | $P \times 10$ B | 70 GB（14+56，**不含 grads**） | Day18 |
| FSDP 每 rank（ $N=8$ ） | $84/8$ | 10.5 GB | Day17 |
| FSDP 每 rank（ $N=64$ ） | $84/64$ | 1.3125 GB | 本课验收 |

三处"反直觉"（代码 `test_amp_equals_pure_fp32` /
`test_ckpt_diff_is_grads` 锁定）：

1. AMP 112 GB = 纯 fp32 训练 112 GB：BF16 省的是**数值范围**
   （8 位指数），不是字节——混合精度台账反而更贵，多出来的是
   fp32 master 权重（Day16）。
2. checkpoint 70 GB < 模型状态 84 GB：差的 14 GB 正是 grads，
   step 内瞬态，存了白存（Day18）。
3. FSDP 每 rank 随 $N$ 线性降，但通信账（例 B）随 $N$ 涨——
   显存和通信的跷跷板是第二层的整条主线。

## 2. 可手算例子 B：通信总账（Day03/15→17→18）

| 项 | 公式 | 手算锚点 | 出处 |
|---|---|---|---|
| ring all-reduce（单 rank） | $2(N-1)/N \times M_{ar}$ | $M_{ar}=67{,}108{,}864$ B, $N=8$ → 117.4 MB | Day03/15 |
| TP 每层每 microbatch | $4 \times 117.4$ MB | 469.8 MB（attn+mlp，fwd 2 + bwd 2） | Day15 |
| TP 80 层每步走线 | $80 \times 469.8$ MB | 37.6 GB | Day15 |
| NVLink vs IB | $37.6/900$ vs $37.6/50$ | 41.8 ms vs 0.75 s（18×，theoretical） | Day15 |
| PP bubble | $(p-1)/(m+p-1)$ | $p=4,m=8$ → 27.3%； $m=32$ → 8.6% | Day15 |
| ZeRO-3 每步通信 | $1.5 \times$ DDP | 7B / $N=8$ ：36.75 GB/step | Day17 |
| DCP sharded | 每 rank $70/64$ GB | 1.09375 GB，零额外通信 | Day18 |
| rank0 gather | $(N-1)/N \times 70$ GB | 68.9 GB（单向汇聚，无 ring 的 ×2） | Day18 |

何时不赚（通信侧）： $N$ 小时 gather 的简单 > DCP 的文件数；
模型+优化器单卡装得下时，TP 的 4 次 all-reduce/层是纯 overhead；
 $m \le p$ 时 bubble $\ge 50\%$ ，加机器反而更慢（Day15）。

## 3. 可手算例子 C：决策闭环（Day17 选型 + Day18 Young/Daly）

框架一句选型（Day17）：FSDP 原生轻量（ZeRO-3 生产形态），
DeepSpeed 功能全（ZeRO-Offload 等），Megatron 3D 切分最精细——
选型 = 并行语义 + 代码侵入度的打包。

Young/Daly： $\tau^* = \sqrt{2\delta M}$ ，
 $W(\tau) = \delta/\tau + \tau/(2M)$ ：

| 方案 | $\delta$ | $\tau^*$ | $W$ |
|---|---|---|---|
| toy | 10 s | 141 s | 14.1% |
| sync（sharded 写盘） | 60 s | 34.6 min | 5.8% |
| async（stall ≈ 55 ms） | 55 ms | 63 s | 0.17% |

（ $M=10$ h；代码 `test_young_daly_*` 三组断言锁定。）
async 改变的不是"要不要存"，而是"存"从半小时一次的慎重决定
变成每分钟一次的常规操作（Day18 讨论题结论）。

决策链一条线：总账（例 A）→ 切分（例 B）→ 压峰值 → 选框架 →
定存档间隔。任何一环的数字变了，下游全得重算——这就是复盘的意义。

## 4. 三词复盘法（Day14–18，每课：最顺 / 最卡 / 一句话沉淀）

| Day | 最顺的一块 | 最卡的一块 | 一句话沉淀 |
|---|---|---|---|
| 13 | KV 账本公式（16384/4096/1152/512 B） | MLA 的 RoPE 解耦复杂度税 | decode 省字节 = 省时间；prefill 算量原样不动 |
| 14 | 84 GB 台账（12 B/param） | per-block FSDP 峰值 14.4 GB | 12 B/param 是单卡生死线 |
| 15 | bubble 公式 $(p-1)/(m+p-1)$ | TP 跨机 18× 定量 | 通信最密处用最快的网 |
| 16 | BF16 8 位指数 ≈ fp32 范围 | AMP 112 GB 反直觉 | 低精度省范围不省字节 |
| 17 | 10.5 GB/rank 一句算清 | 框架侵入度对比 | 选型 = 并行语义 + 代码代价打包 |
| 18 | $\tau^* = \sqrt{2\delta M}$ | async 一致性（快照隔离） | 存档降 MTTR，不降 MTBF |

## 5. 何时不赚汇总（第二层总表）

| 场景 | 结论 | 出处 |
|---|---|---|
| 模型+优化器单卡装得下 | DDP + ZeRO 就够；TP/ZeRO-3 的额外通信是纯 overhead | Day15/17 |
| TP 跨机（IB 50 GB/s） | 42 ms → 0.75 s，step time 被通信主导 | Day15 |
| $m \le p$ | bubble ≥ 50%，加机器更慢 | Day15 |
| 只想省 activation | 先试 SP / activation checkpoint，别上 TP | Day15/16 |
| $T_{train} < \tau^*$ | checkpoint 一次轮不到，只存最终版 | Day18 |
| $N$ 小（如单机 8 卡） | rank0 gather 的简单 > DCP 的文件数 | Day18 |

## 术语与来源

- **ZeRO / FSDP**：Rajbhandari et al., *ZeRO: Memory Optimizations Toward
  Training Trillion Parameter Models*, arXiv:1910.02054；PyTorch FSDP 官方文档。
- **TP/PP/DP 三维切分**：Shoeybi et al., *Megatron-LM*, arXiv:1909.08053。
- **GPipe / bubble**：Huang et al., *GPipe*, arXiv:1811.06965。
- **混合精度 AMP**：Micikevicius et al., *Mixed Precision Training*,
  arXiv:1710.03740。
- **DCP**：PyTorch Distributed Checkpoint 官方文档，
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- **Young / Daly**：Young, CACM 1974；Daly, FGCS 2006。
- **MLA**：DeepSeek-AI, *DeepSeek-V2*, arXiv:2405.04434；
  KV 口径复用 r2-Day13 手算表。
- **ring all-reduce 流量公式**：r2-Day03（NCCL 环形算法）。

## 验证状态

- 已验证：Python 语法 + `unittest` 20/20 通过（纯 CPU，
  `review_ledger.py` + `test_review_ledger.py`；每个断言的期望值
  均直接复用 Day13/14/15/16/17/18 README 手算锚点）。
- 未验证：多卡 torch / NCCL / CUDA / H100（execution not validated，
  待 H100 验证）。带宽换算时间为 theoretical estimate，非实测。
  本课状态：blocked。
- 公开 repo 无雇主标识。
