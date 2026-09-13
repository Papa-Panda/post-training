# r2-Day18 NOTES — 术语、公式核对与上一课/下一课链接

## 术语（准确定义）

- **checkpoint（训练状态快照）**：params + optimizer states + step/epoch +
  rng 的持久化。故障后从最近快照恢复，只重做快照之后的步数。
  **不是** r2-Day16 的 activation checkpointing（重计算省显存）。
- **DCP（torch.distributed.checkpoint）**：PyTorch 的分布式 checkpoint
  语义。核心是 planner + storage writer/reader 的分离：
  planner 决定"谁写哪片"，writer 决定"写成什么格式落到哪"。
  每 rank 只写自己手里的 shard → 零额外通信。
- **reshard**：换 world size 重启时，把旧分片拼回全局再按新卡数重切。
  DCP 的 load 天然支持（`load(world_size=...)`），代价是一次性全量搬运。
- **async_save**：先做 D2H 一致性快照（快照点冻结参数），再后台线程写盘。
  训练在快照后立即继续，stall 从"写盘时间"降到"快照时间"。
- **torn write**：crash 恰好发生在写盘中途产生的半截文件。
  兜底：两槽轮换（总有一个上一个好槽）+ manifest 原子发布
  （os.replace，半截 manifest 不可见）+ 每 shard sha256 校验。
- **Young / Daly 最优间隔**： $τ^* = \sqrt{2δM}$ ，
  $\delta$ = 写盘耗时， $M$ = 系统 MTBF。Young 1974 一阶近似，
  Daly 2006 高阶修正（本课用一阶式，鸟瞰够用）。
- **MTBF / MTTR**：平均故障间隔 / 平均恢复时间。
  checkpoint 不降 MTBF，降的是 MTTR（从 O(重训) 到 O(恢复)）。
- **rng state**：exact resume 需要 python / numpy / torch-cpu /
  每 rank torch-cuda 四套随机数状态；丢一个，恢复后采样顺序分叉，
  "续训"变"重开一条时间线"（代码反例测试锁定）。

## 公式核对（与 r2-Day02/Day14/Day17 一致性）

- 70 GB 台账： $7\times 10^9 \times 10$ B = 70 GB，
  其中 params 14 GB（fp16，r2-Day14 口径）+ opt 56 GB（fp32 m+v）。
  与 Day17 的 12 B/param（84 GB）区别：checkpoint **不含 grads**
  （2 B/param），Day17 的 model-state 台账含 grads。两处 README
  都已声明口径，不要混用。
- per-rank 1.09375 GB = 70/64：与 Day17 FSDP 每 rank 10.5 GB 对比——
  Day17 的 10.5 GB 是 model states（12 B/param），本课是 checkpoint
  （10 B/param），差的 2 B/param 正是 grads。
- rank0 gather 额外流量 $(N-1)/N \times$ 总量： $N=64$ 时 68.90625 GB。
  注意它**不是** ring all-reduce（Day03 公式 $2(N-1)/N$ ），
  gather 是 $(N-1)$ 份 shard 单向汇聚，没有"×2"。
- Young/Daly toy： $δ=10$ s， $M=1000$ s → $τ^*=\sqrt{20000}\approx141.4$ s，
  $W=14.14\%$ ，最优点两项各 7.07%（测试断言）。
- 现实口径为 theoretical estimate（假设已声明）：sharded 写盘
  $δ=60$ s、 $M=10$ h → $τ^*\approx34.6$ min、 $W\approx5.8\%$ ；
  async（stall≈55 ms）→ $τ^*\approx62.9$ s、 $W\approx0.17\%$ 。

## 与上一课的连接

r2-Day17 回答"训练装进哪个框架跑"（并行语义 + 代码侵入的打包）。
r2-Day18 回答"跑起来之后挂了怎么恢复"：框架选定 → 训练启动 →
第一个生产问题就是 fault tolerance。DCP 恰好是 FSDP/ZeRO-3 的
原生存档语义（shard 本来就在各 rank 手里，写本地即存档，零额外通信）——
Day14 算出台账（ZeRO 切分），Day17 选了框架（FSDP 即 ZeRO-3 生产形态），
Day18 把"ZeRO 切分后的状态"持久化。三课是一条线：
**切分 → 装框架 → 存档**。

和 r2-Day02 的连接：Day02 的 `state_dict`（model+opt+epoch+rng）
是单机版四件套；Day18 是它的 N 卡版——四件套不变，
变的是"分片写、异步写、换卡数重分片"三件事。

## 下一课预告

r2-Day19：复盘周。第二层（分布式 14–18 天）收官：
回头检查 MHA→FSDP→3D→AMP→框架选型→Checkpoint 哪块最顺、哪块最卡，
用三词复盘法沉淀，再进第三层推理。

## 讨论难点（供 ai chat）

同步 checkpoint（ $δ=60$ s）vs async（stall $\approx 55$ ms），
系统 MTBF $M=10$ h，7B / $N=64$ ：

1. 用 Young/Daly 分别算两种方案的 $τ^*$ 和 $W$ ，
   async 是否改变"多久存一次"的决策？（答案： $τ^*$ 从 34.6 min
   缩到 ~63 s——存档从"半小时一次的慎重决定"变成"每分钟一次的常规操作"。）
2. 如果这次训练总时长只有 2 h，同步方案还值得开 checkpoint 吗？
   若 $M=100$ h 呢？（提示：先算 $τ^*$ ，再问 $T$ 和 $τ^*$ 谁大。）
