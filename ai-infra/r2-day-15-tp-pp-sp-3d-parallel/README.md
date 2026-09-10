# r2-Day15 — 3D 并行 TP/PP/SP：把模型本身切开

## Connection to Prev

r2-Day14 切的是 **model states**（参数/梯度/优化器状态），模型本体还在每张卡上完整放着——
7B 能这么玩，70B 不行：光是 fp16 参数就 $140\ \text{GB}$ ，
单卡连参数都装不下。r2-Day15 切的是**模型本身**：TP 把每一层的矩阵按张量维度切
（列并行 + 行并行），PP 把 $L$ 层按深度切成流水线，SP 把序列维度切开省 activation。

**牺牲**：TP 每层每个 microbatch 引入 4 次 all-reduce（forward 2 + backward 2），
通信发生在训练最热的路径上；PP 引入 pipeline bubble（ $p$ 个 stage、 $m$ 个 microbatch
时空闲占比 $(p-1)/(m+p-1)$ ）；SP 在 attention 处需要把 TP 域的 all-reduce 换成
reduce-scatter/all-gather。**换取**：单卡装不下时的唯一放大路径——
TP 让 $140\ \text{GB}$ 参数分散到 8 卡，PP 让 80 层分散到 4 个 stage，
SP 让 activation 省 $T$ 倍。**何时不赚**：模型装得下时 DDP + ZeRO 更简单；
TP 跨机（IB 只有约 $50\ \text{GB/s}$ ）会被通信淹没——TP 必须待在机内；
microbatch 数 $m$ 太小则 PP 的 bubble 吃掉并行收益。

抄百科类比：TP = 每页拆成左右两栏，两人各抄一栏，拼页时要对齐（all-reduce）；
PP = 整本百科按章节分给几组人流水抄，上一组抄完一章传给下一组（传的是纸=激活，
不是整本百科=梯度全局同步）；
SP = 一页太长，按行切段分给多人抄。

## 0. 符号表（每个字母的含义）

- $N$ ：总 rank 数； `T, P, D` ：TP / PP / DP 并行度， $N = T\cdot P\cdot D$
- `(d, p, t)` ：某 rank 在三个维度上的坐标；rank 编号约定
  $$\text{rank} = (d\cdot P + p)\cdot T + t$$
  与 Megatron-LM 的 `parallel_state.py` 一致：TP 的 $T$ 个 rank 连续编号。
- $b$ ：microbatch 大小； $S$ ：序列长； $L$ ：层数； $H$ ：hidden  dim
- $M$ ：一次 all-reduce 的 payload（字节）；ring all-reduce 单 rank 流量
  $= 2(T-1)/T\cdot M$ （r2-Day03 的 ring 公式）
- 带宽口径：NVLink（H100 SXM，单卡双向合计） $900\ \text{GB/s}$ ；
  IB NDR $400\ \text{Gb/s} = 50\ \text{GB/s}$ 每网卡（理论值）；
  PCIe Gen5 x16 $64\ \text{GB/s}$ 。本课 GB 均为十进制。

## 1. 可手算例子 A：TP 每层 4 次 all-reduce，通信量到底多大

Megatron 式 TP 把一个 Transformer 层的两个 GEMM 拆开：

- Attention：QKV 投影**列并行**（每 rank 算自己的 head 分片），
  output 投影**行并行** → 行并行输出需要 **all-reduce #1** 把 $T$ 个分片加起来；
- MLP：fc1 列并行（ $H\to 4H$ 按输出切），fc2 行并行 → **all-reduce #2** ；
- backward 对称地再来 2 次。**每层每 microbatch 共 4 次 all-reduce**，
  payload 都是行并行输出 $b\cdot S\cdot H$ 个元素。

手算（70B 量级： $H=8192$ ，bf16 占 $2\ \text{B}$ ， $b=1, S=4096, T=8$ ）：

$$M = 1 \times 4096 \times 8192 \times 2\ \text{B} = 67{,}108{,}864\ \text{B} \approx 67.1\ \text{MB}$$

单 rank 一次 all-reduce 流量（ring，r2-Day03 公式）：

$$2\cdot\frac{7}{8}\cdot 67.1\ \text{MB} \approx 117.4\ \text{MB}$$

每层每 microbatch： $4 \times 117.4\ \text{MB} \approx 469.8\ \text{MB}$ ；
$L=80$ 层： $80 \times 469.8\ \text{MB} \approx 37.6\ \text{GB}$ （单 rank 每步走线量）。

NVLink $900\ \text{GB/s}$ 下约 $41.8\ \text{ms}$ ；若把 TP 跨机放到 IB $50\ \text{GB/s}$ 上，
$37.6/50 \approx 0.75\ \text{s}$ ——**慢 18 倍**。这就是"TP 不能跨机"的定量版本：
TP 的通信发生在每个 microbatch 的最热路径上，带宽差直接乘到 step time 里。

代码 `test_tp_traffic_handcalc` 用 $M=67{,}108{,}864$ 断言了 $117.4\ \text{MB}$ 与 $469.8\ \text{MB}$ 。

## 2. 可手算例子 B：64 卡拓图（ $T=8, P=4, D=2$ ，每节点 8 卡）

$8\times 4\times 2 = 64$ 。按 $\text{rank}=(d\cdot P+p)\cdot T+t$ ：

| $d$ | $p$ | rank 区间 | 节点 |
|---|---|---|---|
| 0 | 0 | 0–7 | node0 |
| 0 | 1 | 8–15 | node1 |
| 0 | 2 | 16–23 | node2 |
| 0 | 3 | 24–31 | node3 |
| 1 | 0 | 32–39 | node4 |
| 1 | 1 | 40–47 | node5 |
| 1 | 2 | 48–55 | node6 |
| 1 | 3 | 56–63 | node7 |

关键结论（代码 `node_of_rank` + 测试全覆盖 64 个 rank）：

- **TP group 是 8 个连续 rank = 恰好一台机器**（如 `d=0,p=0` 的 rank 0–7 全在 node0）。
  例子 A 的 $37.6\ \text{GB}$ 全部走机内 NVLink。
- **PP 只跨节点传"纸"**：stage 边界的点对点激活/梯度，每 microbatch 每边界
  $2\cdot b\cdot S\cdot H\cdot 2\ \text{B} \approx 134.2\ \text{MB}$ （同上例），
  走 IB 只要 $2.7\ \text{ms}$ ——PP 天生容忍跨机。
- **DP group**：相同 `(t,p)` 的两个 `d` ，如 `(t=0,p=0)` → rank $\{0, 32\}$ ，
  做梯度 all-reduce（r2-Day04 的 DDP 逻辑，只是每组只有 2 个 rank）。

所以 64 卡的标准答案是：**TP=8 机内（通信最密处用最快的网），
PP=4 跨机（只传边界激活），DP=2 收尾**。

## 3. 可手算例子 C：PP 的 bubble

$p$ 个 stage， $m$ 个 microbatch，GPipe 调度下总空闲占比：

$$\text{bubble} = \frac{p-1}{m+p-1}$$

$p=4, m=8$ ： $3/11 \approx 27.3\%$ 的时间有机器在等；
$m=32$ ： $3/35 \approx 8.6\%$ 。**教训： $m \gg p$ ，microbatch 数是 PP 的生命线**。
1F1B 调度把 bubble 的内存代价减半（GPipe 要存全部 $m$ 个 microbatch 的激活，
1F1B 只需约 $p$ 个在飞），但公式里的 $(p-1)$ 空转躲不掉——除非用 interleaved
virtual stage 把每个 rank 再切 $v$ 份。

## 4. SP：序列维度的切分

LayerNorm / dropout 是沿 $S$ 逐元素的——按 $S$ 切分**零通信**。
麻烦在 attention：算 $QK^T$ 需要全局的 `K,V` 。
Megatron 的 SP 做法：TP 域内 attention 后的 all-reduce 换成
forward 做 **reduce-scatter**、backward 做 **all-gather**，
让 activation 在 TP 域里始终保持按 $S$ 分片。
效果：sharded 区域的 activation 内存降为 $1/T$ ——长上下文训练的刚需。
注意 SP 不省**权重**内存，只省 activation；且它活在 TP 域内部，
不能替代 TP 跨机（跨机问题例子 A 已经算过）。

## 5. 何时不赚（决策表）

| 场景 | 结论 |
|---|---|
| 模型+优化器状态单卡装得下 | DDP + ZeRO 就够了；TP 的 4 次 all-reduce/层是纯 overhead |
| TP 跨机（IB $50\ \text{GB/s}$ ） | 例子 A： $42\ \text{ms} \to 0.75\ \text{s}$ ，step time 被通信主导 |
| $m$ 太小（如 $m \le p$ ） | bubble $\ge 50\%$ ，加机器反而更慢 |
| 只想省 activation | 先试 SP / activation checkpoint（r2-Day14 思路），别上 TP |
| PP stage 层数不均 | 最慢的 stage 决定 throughput；首尾 stage 还背 embedding/head |

## 来源

- Megatron-LM（TP/PP/DP 三维切分）：Shoeybi et al., *Megatron-LM: Training Multi-Billion
  Parameter Language Models Using Model Parallelism*, arXiv:1909.08053
- GPipe（microbatch pipeline 与 bubble）：Huang et al., *GPipe: Efficient Training of
  Giant Neural Networks using Pipeline Parallelism*, arXiv:1811.06965
- 1F1B 调度：Narayanan et al., *Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM*, arXiv:2104.04473
- Sequence Parallelism：Korthikanti et al., *Reducing Activation Recomputation in Large
  Transformer Models*, arXiv:2205.05198
- DeepSpeed Ulysses（SP 的另一种实现：沿 head 维 all-to-all，本课 NOTES 有对比）：
  DeepSpeed 官方文档

## 验证状态

- 本地： `python3 -m unittest discover` ，20/20 CPU 测试通过
  （topology/坐标互逆/三组 group 成员/64 卡节点映射/ring 流量公式与 r2-Day03 一致/
  TP 手算 $117.4\ \text{MB}$ 与 $469.8\ \text{MB}$ /bubble $3/11$ /P2P $134.2\ \text{MB}$ ）。
- `tools/check_repo.py` 对新目录检查通过；新文件无雇主标识（grep 验证）。
- 未验证：无 PyTorch/NCCL/CUDA/H100， $37.6\ \text{GB}$ / $41.8\ \text{ms}$ /
  $0.75\ \text{s}$ 均为理论估计（theoretical estimate），非实测 benchmark；
  状态保持 blocked，execution not validated / 待H100验证。
