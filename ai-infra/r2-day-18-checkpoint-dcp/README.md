# r2-Day18 — Checkpoint & Recovery（DCP）：挂了怎么恢复

## Connection to Prev

r2-Day02 的 `state_dict`（model + opt + epoch + rng）是**单机版答案**；
r2-Day17 把训练装进生产框架跑起来。跑起来之后，第一个生产问题就是
"挂了怎么恢复"——r2-Day18 是 Day02 state_dict 的 **N 卡分布式版**：
分片写、异步写、换 world size 重分片。

注意和 r2-Day16 区分：Day16 的 checkpoint（activation checkpointing /
重计算）省的是 **activation 显存**，是"用计算换显存"；
Day18 的 checkpoint（训练状态快照）防的是**故障**，是"用写盘换 MTTR"。
名字一样，对象完全不同。

**牺牲**：写盘 I/O + 存储 + 同步 stall（或 async 的一致性复杂度：
快照时刻冻结、torn write 处理）。**换取**：故障后 MTTR 从
O(从头重训) 降到 O(恢复时间)。**何时不赚**：训练总时长
 $T$  < τ\*（最优间隔一次都轮不到）时，只存最终版；
小模型/短训练的故障期望损失 < checkpoint 开销时，不赚。

抄百科类比：checkpoint = 游戏存档。full rank0 gather = 全队把装备
寄到一个人手里再存档（多付一次快递）；sharded DCP = 每人存自己的背包
（零快递，但换队伍人数要重新分装备）；async = 先截图存内存，
存档慢慢写（截图那一瞬间不能动）。

## 0. 符号表（每个字母的含义）

- $P$ ：参数量； $N$ ：参与训练的 GPU 卡数
- $B_p$ ：每参数的参数字节（fp16/bf16 取 2，fp32 取 4）
- $B_{opt}$ ：每参数的 optimizer 状态字节（Adam fp32 的 m+v 取 8）
- checkpoint 台账 = params + optimizer states，**不含 grads**
  （grads 是 step 内瞬态，恢复后第一个 step 重算，存了白存）
- GB：十进制 $10^9$ （与 r2-Day14/15/16/17 口径一致）
- $\delta$ ：一次 checkpoint 写盘耗时（秒）； $M$ ：系统 MTBF（秒）；
   $\tau$ ：checkpoint 间隔（秒）； $W(τ)$ ：浪费占比

## 1. 可手算例子 A：checkpoint 台账——存什么、不存什么

存四件套：**params + optimizer states + step/epoch + rng**
（rng 要存 python / numpy / torch-cpu / 每 rank torch-cuda 四套，
 exact resume 全要；本课代码用一整数 LCG 状态演示）。

先 toy 热身（ $P=100$ ）： $100 \times 10 = 1000$ B。

7B 实战口径（ $P=7\times 10^9$ ，fp16 参数 + Adam fp32 m/v）：

| 项 | 字节/参数 | 7B 总量 |
|---|---|---|
| params（fp16） | 2 | 14 GB |
| optimizer states（fp32 m+v） | 8 | 56 GB |
| grads | — | 0（不存） |
| **合计** | **10** | **70 GB** |

为什么 opt state 和 rng 必须存？代码里的反例测试
`test_cold_restart_without_opt_rng_diverges` 证明了：
只恢复 params（momentum 归零、rng 重开），10 步后轨迹与 uninterrupted
分叉——momentum 是"历史梯度的记忆"，rng 决定采样顺序，
丢一个，恢复就不是"续训"而是"重开一条分叉的时间线"。
crash 恢复测试 `test_crash_recovery_identical` 则证明：
四件套全存时，crash + 恢复后的 30 步与 uninterrupted 的 30 步
**逐 bit 一致**（确定性代码下）。

## 2. 可手算例子 B：full rank0 gather vs sharded DCP

| 方案 | 额外通信 | 单点负担 | 文件数 | 换 world size 重启 |
|---|---|---|---|---|
| full rank0 gather | $(N-1)/N \times$ 总量 | rank0 staging = 总量 | 1 | 天然支持 |
| sharded DCP | **0** | 每 rank = 总量 $/N$ | $N$ | 需重分片（一次性全量搬运量级） |

7B / $N=64$ 口径（代码 `test_rank0_gather_extra_7b_64` 断言）：

- sharded DCP：每 rank 写 1.09375 GB，**零额外通信**
  （FSDP/ZeRO-3 下 shard 本来就在各 rank 手里，写本地即存档）。
- full rank0 gather：额外搬运 $63/64 \times 70 = 68.90625$ GB，
  rank0 还要 70 GB staging 内存——这次搬运训练本身完全不需要，
  纯粹是"为了写成一个文件"付的税。

$N$ 越大，gather 的税越接近"搬运整个 checkpoint"
（`test_rank0_gather_grows_with_n` 锁定单调性），
而 sharded 恒为 0——**DCP 是 $N \to \infty$ 时的答案**。
代价是另一面： $N$ 个文件（万卡即万文件，元数据/小文件开销），
以及换 world size（弹性重启）时要做一次重分片：
每字节最多搬运一次，总量级 ~70 GB，一次性。

## 3. 可手算例子 C：Young / Daly 最优间隔

浪费占比 $W(τ) = δ/τ + τ/(2M)$ ：第一项是写盘开销占比，
第二项是故障期望返工（故障均匀落在间隔内，平均丢 $τ/2$ 进度，
单位时间故障率 $1/M$ ）。求导：

$$-δ/τ^2 + 1/(2M) = 0 \quad\Rightarrow\quad τ^* = \sqrt{2δM}$$

最优点处两项相等（代码 `test_waste_at_optimum_equal_split` 断言）。

先 toy（ $δ=10$ s， $M=1000$ s）： $τ^* = \sqrt{20000} \approx 141.4$ s，
 $W = 14.14%$ （两项各 7.07%）。

现实口径（**theoretical estimate**，假设 sharded 写盘 $δ=60$ s，
系统 MTBF $M=10$ h）： $τ^* \approx 2078$ s $\approx 34.6$ min，
 $W \approx 5.8%$ 。

async 的算法： $\delta$ 换成 **stall 时间**（HBM→CPU memcpy，
1.09 GB @ ~20 GB/s $\approx 55$ ms，后台写盘不 stall）：
 $τ^* \approx 62.9$ s， $W \approx 0.17%$ （theoretical estimate，
假设成立）。async 没有消灭写盘，只是把 $\delta$ 从"写盘时间"
换成了"快照时间"——但 Young/Daly 公式照用， $τ^*$ 缩短约 33 倍。

**何时不赚**：若训练总时长 $T < τ^*$ ，按最优间隔一次都轮不到——
只存最终版（或不存）。 $M \gg T$ 时故障期望损失 $T^2/(2M)$
量级 < checkpoint 开销，同样不赚。

## 4. 何时不赚（反例，必读）

- **训练时长 < τ\***：最优间隔比训练还长，存了就是纯开销。
- ** $M \gg T_{train}$ **：训练期内故障概率 $T/M$ 很小，
  期望返工 < checkpoint 开销。
- **间隔 << τ\***： $δ/τ$ 项主导，存得越勤亏得越多。
- **大 $N$ 下用 rank0 gather**：额外流量 $\approx$ 整个 checkpoint，
  不如 sharded。
- **torn write 没兜底**：crash 恰好发生在写盘中途，唯一的 checkpoint
  是半截文件——还不如不存。生产做法：两槽轮换 + manifest 原子发布
  + 每 shard 校验（代码 `ShardedCheckpointer` 全实现并测试）。

## 术语与来源

- **DCP（torch.distributed.checkpoint）**：PyTorch 分布式 checkpoint
  语义——planner（谁写哪片）+ storage writer/reader（落盘格式），
  per-rank 分片写零额外通信，load 时可换 world size 重分片。
  来源：PyTorch 官方文档 Distributed Checkpoint，
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- **async_save**：先做 D2H 一致性快照（快照点冻结），再后台线程写盘；
  快照之后训练继续，写盘看到的仍是快照时刻的值
  （代码 `test_async_save_snapshot_isolation` 断言）。
  来源：同上，`torch.distributed.checkpoint.async_save`。
- **Young / Daly 最优间隔**： $τ^* = \sqrt{2δM}$ 。
  来源：Young, "A First Order Approximation to the Optimum Checkpoint
  Interval", CACM 1974；Daly, "A higher order estimate of the optimal
  checkpoint interval for restart dumps", FGCS 2006。
- **MTBF / MTTR**：平均故障间隔 / 平均恢复时间；checkpoint 把 MTTR
  从 O(重训) 降到 O(恢复)。
- **torn write / 两槽轮换**：写盘中途 crash 产生半截文件；
  两槽轮换 + manifest 原子发布保证总有一个好槽。
- **activation checkpointing**：r2-Day16 的"重计算"，省 activation
  显存；与本课"训练状态快照"名字一样、对象完全不同。

## 验证状态

- 已验证：Python 语法 + `unittest` 19/19 通过（纯 CPU，
  `checkpoint_dcp.py` + `test_checkpoint_dcp.py`）。
- 未验证：多卡 torch DCP 真实落盘 / NCCL / CUDA / H100
  （execution not validated，待 H100 验证）。本课状态：blocked。
- 公开 repo 无雇主标识。
