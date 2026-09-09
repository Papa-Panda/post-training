# r2-Day14 — FSDP/ZeRO 显存账：用通信换显存

## Connection to Prev

r2-Day13 在**架构层面**省 decode 的 KV cache（推理侧：decode 是 memory-bound，
省字节 = 省时间）。r2-Day14 回到**训练侧**：不管 attention 怎么压缩，
训练一个 7B 模型要同时放下参数 $14\ \text{GB}$ （fp16）+ 梯度 $14\ \text{GB}$ （fp16）
+ Adam 状态 $56\ \text{GB}$ （fp32 的 m+v） $= 84\ \text{GB} > 80\ \text{GB}$ ——
单张 H100 根本装不下。FSDP/ZeRO 把 model states（参数/梯度/优化器状态）
切到 $N$ 张卡上，每张只存 $1/N$ 。

**牺牲**：通信。ZeRO-3 每步通信量是 DDP 的 $1.5\times$ ：forward 和 backward
各一次 all-gather 参数凑全 + backward 后一次 reduce-scatter 梯度；
DDP 只有一次 all-reduce 梯度。**换取**：被切分的 states 显存降为 $1/N$ ——
7B 在 8 卡上每卡 model states 只要 $14\ \text{GB}$ （mixed 口径）/
 $10.5\ \text{GB}$ （roadmap 口径，不计 fp32 master）。
**何时不赚**：模型本来就装得下——DDP 更简单、通信只有 ZeRO-3 的 $2/3$ ；
ZeRO-3 的全量 all-gather 在带宽一般、 $N$ 大时最痛；
optimizer states 已经是主要矛盾时，ZeRO-1/2 往往就够了。

## 0. 符号表（每个字母的含义）

- $P$ ：参数量（元素个数）； $N$ ：rank 数（数据并行度）
- $b_p, b_g, b_o$ ：参数 / 梯度 / 优化器状态的每参数字节数
- $M$ ：一次 collective 的 payload 字节数
- stage 0 = DDP（全复制），1/2/3 = ZeRO 的三个阶段
- 本课 GB 均为十进制（ $10^9$ 字节），与 H100 80GB 等厂商规格一致

## 1. 可手算例子 A：7B 显存账（为什么单卡装不下）

mixed precision Adam 的每参数字节：
fp16 参数 $2\ \text{B}$ + fp16 梯度 $2\ \text{B}$ + fp32 master 参数 $4\ \text{B}$
+ fp32 Adam 一阶矩 $4\ \text{B}$ + fp32 二阶矩 $4\ \text{B}$ $= 16\ \text{B/param}$ ，
$$7\times10^9 \times 16\ \text{B} = 112\ \text{GB}$$

roadmap 口径（把 fp32 master 折掉，只算"训起来必须有的"）：
$2 + 2 + 8 = 12\ \text{B/param}$ ，
$$14\ \text{GB} + 14\ \text{GB} + 56\ \text{GB} = 84\ \text{GB} > 80\ \text{GB}$$
→ 单卡放不下。代码 `test_7b_roadmap_ledger` 断言了 $14/14/56\ \text{GB}$ 精确整数。

关键洞察：**optimizer states 才是大头**（ $56\ \text{GB}$ > 参数+梯度 $28\ \text{GB}$ ）——
这就是 ZeRO-1 只切 optimizer states 就能把显存砍掉近一半的原因。

## 2. 可手算例子 B：ZeRO 三阶段（ $P=100$ ，fp32 Adam， $N=4$ ）

每参数 $16\ \text{B}$ ，model states 总量 $1600\ \text{B}$ 。
记参数/梯度各 $400\ \text{B}$ ，优化器状态 $800\ \text{B}$ ：

| 方案 | 每卡字节 | 算式 |
|---|---|---|
| DDP | $1600\ \text{B}$ | 全复制 |
| ZeRO-1（切 opt states） | $1000\ \text{B}$ | `400 + 400 + 800/4` |
| ZeRO-2（+ 切 grads） | $700\ \text{B}$ | `400 + 400/4 + 800/4` |
| ZeRO-3（+ 切 params） | $400\ \text{B}$ | `400/4 + 400/4 + 800/4` |

通信（ring 公式与 r2-Day03 一致：all-reduce $= 2(N-1)/N \times M$ ，
all-gather/reduce-scatter $= (N-1)/N \times M$ ）：
- DDP：all-reduce 梯度 $= 2 \times 3/4 \times 400 = 600\ \text{B}$ ；
- ZeRO-1/2：reduce-scatter + all-gather 梯度，总量与 all-reduce 相等 $= 600\ \text{B}$ ；
- ZeRO-3：all-gather 参数（forward 前）+ all-gather 参数（backward 前）
  + reduce-scatter 梯度 $= 3 \times 3/4 \times 400 = 900\ \text{B} = 1.5\times$ DDP。

代码断言了 $1600/1000/700/400$ 与 $900/600 = 1.5$ 。
检验标准一句话：**ZeRO-2 只 all-reduce 梯度；ZeRO-3 连参数也切，
forward/backward 都要 all-gather 凑全参数，通信是 DDP 的 $1.5\times$ ，
显存降到 $1/N$ 。**

## 3. 可手算例子 C：FSDP per-block 峰值

FSDP 把每个 wrap 单元（通常一个 transformer block）的参数展平成一维
flat_param 再切分。steady state 每卡只持有 $1/N$ ，但每个 block 的 forward 前
要 all-gather 出**完整**的 block 参数——瞬时峰值：

$$\text{peak} = \text{sharded 总量} + \text{一个完整 block（fp16）}$$

7B，32 个 block， $N=8$ ： $14.0 + 0.4 = 14.4\ \text{GB}$ /卡。
这是 r2-Day02 的 `(P-b)/G + b` 峰值公式的 FSDP 版本（Day02 是 DDP 版）。

为什么按 block wrap 而不是整模型 wrap：整模型 all-gather 的瞬时峰值是整份参数；
per-block 把峰值压到一个 block，还能让**下一个 block 的 all-gather 与当前 block
的计算 overlap**（prefetch； `limit_all_gathers` 控制飞行中的 all-gather 数量，
避免 prefetch 吃掉省下来的显存）。

## 4. 可执行代码

- `zero_memory.py`：`model_state_bytes_per_param`（四种精度口径）、
  `sharded_state_bytes`（stage 0–3 账本）、`collective_bytes`
  （ring 公式）、`ddp_step_comm_bytes` / `zero3_step_comm_bytes`
  （ $1.5\times$ ）、`shard_vector` / `all_gather_sim` / `reduce_scatter_sim`
  （纯 Python 的分片语义演示，字节精确）、`fsdp_peak_bytes`。

```bash
python3 ai-infra/r2-day-14-fsdp-zero/zero_memory.py
python3 -m unittest discover -s ai-infra/r2-day-14-fsdp-zero -p 'test_*.py' -v
```

## 5. 何时不赚

1. **模型装得下就别上 ZeRO-3**：DDP 每步通信只有 ZeRO-3 的 $2/3$ ，
   代码路径也简单得多。
2. **ZeRO-1/2 往往够用**：Adam 下 optimizer states 占 $8/16$ ，
   只切 states 就去掉了主要矛盾；ZeRO-3 的参数 all-gather 在跨机或低带宽时最痛。
3. **ZeRO 不切 activations**：长序列的 activation 显存靠重计算解决
   （r2-Day16），不是 ZeRO 的活——两者正交，可以叠加。
4. **小 $N$ 时切分收益小**： $N=2$ 时 ZeRO-3 的 all-gather 每卡搬运
   $(N-1)/N = 1/2$ 的参数，通信占比相对更高；永远看 step time，
   不只看显存数字。

## 状态

- 已验证：Python 语法；13 个 CPU 单元测试（7B 的 $14/14/56\ \text{GB}$ 精确整数、
  $84 > 80$ 装不下；toy 四阶段 $1600/1000/700/400$ ；通信 $600\ \text{B} / 900\ \text{B} = 1.5\times$ ；
  shard/all-gather/reduce-scatter roundtrip；7B 8 卡 per-block 峰值 $14.4\ \text{GB}$ ）。
- **execution not validated / 待H100验证**：本环境无 torch/CUDA GPU；
  真实 FSDP 分片训练、all-gather/reduce-scatter 实测带宽、step time、
  MFU、comm%、设备拓扑均未测量； $14.4\ \text{GB}$ 是理论账本
  （未含 activations 与通信 buffer）。
- 本课状态保持 `blocked`；没有声称 loss、耗时实测、带宽实测。

## 原始 / 官方来源

- ZeRO：Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training
  Trillion Parameter Models", <https://arxiv.org/abs/1910.02054>
- FSDP：Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded
  Data Parallel", <https://arxiv.org/abs/2304.11277>
- PyTorch FSDP 文档：<https://pytorch.org/docs/stable/fsdp.html>
- DeepSpeed ZeRO 教程：<https://www.deepspeed.ai/tutorials/zero/>
- FairScale（FSDP 的前身实现）：<https://github.com/facebookresearch/fairscale>
