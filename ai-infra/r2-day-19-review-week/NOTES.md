# r2-Day19 NOTES — 术语、公式核对与上一课/下一课链接

## 术语（准确定义）

- **复盘周**：路线图"19 复盘周"的落地。第二层（分布式训练 14–18 天）
  收官：不教新公式，只验证 Day14–18 的手算锚点在同一总账里自洽，
  并沉淀"何时不赚"总表。
- **三词复盘法**：每课三个词——最顺的一块 / 最卡的一块 / 一句话沉淀
  （r2-Day18 NOTES 预告的方法，本课 README §4 落地）。
- **模型状态（model states）**：params + grads + optimizer states，
  12 B/param（fp16+Adam 口径，Day14）。注意与 checkpoint（10 B/param，
  不含 grads，Day18）、与 AMP 台账（16 B/param，含 fp32 master，
  Day16）区分——三处口径差是本课例 A 的核心。
- **走线量**：单 rank 每步在集合通信上实际搬运的字节数
  （如 TP 80 层 37.6 GB，Day15）。注意它不是"通信时间"，
  除以带宽才是时间下限（theoretical estimate）。
- **MTTR vs MTBF**：checkpoint 降的是 MTTR（从 O(重训) 到 O(恢复)），
  不降 MTBF（Day18）。复盘时记住：所有"恢复"技术都不减少故障，
  只减少故障的代价。

## 公式核对（与 Day13/14/15/16/17/18 一致性）

- 84 GB（12 B/param）： $7\times 10^9 \times 12 = 84\times 10^9$ B，
  = 14（params fp16）+ 14（grads fp16）+ 56（Adam fp32 m+v）。
  与 Day14/Day17 口径一致； $84 > 80$ 故单卡装不下（Day14 结论）。
- 112 GB（16 B/param）：14 + 14 + 28（fp32 master）+ 56，
  = 纯 fp32 训练的 28 + 28 + 56。BF16 的 8 位指数只解决数值范围，
  不减少字节（Day16）。
- 70 GB（10 B/param）：14 + 56，不含 grads。
  $84 - 70 = 14$ 恰为 grads（代码 `test_ckpt_diff_is_grads` 锁定）。
- ring： $2(N-1)/N \times M_{ar}$ ， $N=8$ 时系数 $7/4=1.75$ ；
  $1.75 \times 67{,}108{,}864 = 117{,}440{,}512$ B ≈ 117.4 MB（Day15）。
- TP：每层 4 次 all-reduce（attn out 行并行 + mlp fc2 行并行，
  forward 2 + backward 2），payload 都是 $b\cdot S\cdot H$ 个元素
  （Day15）。 $4 \times 117.4 = 469.8$ MB/层，80 层 37.6 GB。
- bubble： $(p-1)/(m+p-1)$ ， $p=4,m=8$ → $3/11$ ≈ 27.3%；
  $m=32$ → $3/35$ ≈ 8.6%（Day15）。
- ZeRO-3 1.5× DDP：7B fp16 grads 14 GB，ring $N=8$ 得 DDP
  $1.75 \times 14 = 24.5$ GB/step，×1.5 = 36.75 GB/step（Day17）。
- DCP： $70/64 = 1.09375$ GB/rank；gather 额外
  $63/64 \times 70 = 68.90625$ GB，单向汇聚无 ×2（Day18）。
- Young/Daly： $\tau^* = \sqrt{2\delta M}$ ，
  $W = \delta/\tau + \tau/(2M)$ 。三组锚点（toy 141s/14.1%；
  sync 34.6min/5.8%；async 63s/0.17%）与 Day18 一致。
- KV（Day13 口径，fp16，每 token 每层）：MHA 16384 B，
  GQA-8 4096 B，MLA 1152 B，MQA 512 B；MLA/MHA = 14.2×。
  7B 风格 128k 上下文：64 GiB → 4.5 GiB。

## 与上一课的连接

r2-Day18 回答"跑起来之后挂了怎么恢复"（框架选定 → 训练启动 →
故障 → 快照恢复）。r2-Day19 回答"整层学完了，账算对了吗"：
把 Day14 的台账、Day15 的切分、Day16 的压峰值、Day17 的选型、
Day18 的存档放进同一张表，互相验算（84 vs 70 vs 112；
37.6 GB 走线 vs 41.8 ms； $\tau^*$ vs $T_{train}$ ）。

和 r2-Day02 的连接：Day02 的 `state_dict`（model+opt+epoch+rng）
是单机版四件套；Day18 是它的 N 卡版；Day19 是"从 Day02 到 Day18，
单机训练长成分布式训练，一共多付了哪些代价"的总账。

## 下一课预告

r2-Day20–21：TTFT/TPOT & KV Cache 32GB 手算。第三层（推理部署）
开篇：从"训得动"转向"跑得快"——compute-bound vs memory-bound，
7B 的 KV cache 手算（Day13 的 64 GiB 表格将在推理视角重算一遍：
这次按"每 token 每步读多少"算时间）。

## 讨论难点（供 ai chat）

综合验收（7B fp16+Adam， $N=64$ ，TP=8/PP=4/DP=2，
 $\text{rank}=(d\cdot P+p)\cdot T+t$ ， $\delta=60$ s sharded 写盘，
 $M=10$ h）：

1. 每 rank model-state（FSDP 口径）与 checkpoint per-rank 各多少 GB？
   （答案： $84/64=1.3125$ GB； $70/64=1.09375$ GB。）
2. 每 microbatch 80 层 TP 走线总量多少 GB？走 NVLink 900 GB/s
   与 IB 50 GB/s 各需多久（theoretical）？
   （答案：37.6 GB；41.8 ms vs 0.75 s——TP 不能跨机的定量版本。）
3. sync 方案的 $\tau^*$ 与 $W$ ；训练总长 2 h 时值得开 checkpoint 吗？
   （答案： $\tau^*\approx34.6$ min， $W\approx5.8\%$ ； $2\,\text{h} > \tau^*$
   → 值得，约存 3 次。反问： $T_{train}=20$ min 时呢？）
4. 若把 attention 换成 MLA（Day13 口径）：训练侧 Day14 的 84 GB
   台账变吗？推理侧哪笔账变、量级多少？
   （答案：训练侧不变——12 B/param 里参数还是那些参数，
   MLA 动的是 KV 表示；推理侧 KV cache：7B 风格 128k 上下文
   64 GiB → 4.5 GiB，14.2×。Day13 省的是 decode 的账，
   不是训练的账——两层账本不要串。）
