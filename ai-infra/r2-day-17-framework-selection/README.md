# r2-Day17 — Megatron / DeepSpeed / FSDP 选型：一句话区分三个框架

## Connection to Prev

r2-Day14 切的是 **model states**（ZeRO：通信换显存），r2-Day15 切的是**模型本身**
（TP/PP/SP：切张量、层、序列），r2-Day16 压的是**每卡手里的张量本身**
（BF16/累积/重计算，不换并行结构）。r2-Day17 回答的是工程问题：
**Day01–16 手算过的公式，应该装进哪个生产框架里跑？**

**牺牲**：选定框架 = 选定并行语义 + 代码侵入形态。Megatron-LM 要求按它的
3D 切分假设组织模型与训练脚本；DeepSpeed 要求 `deepspeed.initialize`
接管 optimizer/scheduler；FSDP 要求模型能被 `fully_shard` 包裹
（submodule 结构要友好）。**换框架 ≈ 重写训练入口**，迁移成本真实存在，
不是换个 import 那么简单。**换取**：all-reduce 公式、ZeRO 台账、TP/PP 拓扑、
AMP/累积/checkpoint 全部变成框架开关或一行 config，不用手写通信代码。
**何时不赚**：单卡或 <1B 小模型时三者都 overkill——单卡 + AMP /
DDP 即够，框架迁移成本 > 收益；团队已深度绑定某生态时切换成本 > 收益；
推理侧（vLLM/SGLang）是第三层的事，训练框架选型管不着。

抄百科类比：选框架 = 选装修队。Megatron 是"大平层整装队"（3D 并行样板房，
照着装最省心，但户型得按它的来）；DeepSpeed 是"收纳改造队"
（ZeRO 专治显存不够，旧房子也能住）；FSDP 是"原厂精装"
（PyTorch 亲儿子，拎包入住，和全屋智能最配）。小单间就别请整装队了。

## 0. 符号表（每个字母的含义）

- $P$ ：参数量； $N$ ：参与训练的 GPU 卡数
- $B_p$ ：每参数的参数字节（fp16/bf16 取 2，fp32 取 4）
- $B_{opt}$ ：每参数的 optimizer 状态字节（Adam fp32 的 m+v 取 8）
- model-state 台账 = params + grads + opt states，**不含 activations**
  （activations 是 r2-Day16 checkpoint 的活，两课正交）
- GB：十进制 $10^9$ （与 r2-Day14/15/16 口径一致）
- $t$ ：Megatron 的 TP 切分数；每层 all-reduce 次数固定为 4
  （forward 2 + backward 2，r2-Day15 口径）

## 1. 可手算例子 A：一句话区分三框架

| 框架 | 一句话 | 通信模式 | 代码侵入 | 跨机 |
|---|---|---|---|---|
| Megatron-LM | 模型并行（TP/PP）的生产样板 | 每层固定 4 次 all-reduce，payload $ \propto b\cdot s\cdot h $ | 高（按 3D 切分组织模型/脚本） | TP 限机内，PP 跨机 |
| DeepSpeed | 以 ZeRO 为核心的"显存优化大礼包" | ZeRO-3 ≈ 1.5× DDP | 低（一行 config） | 可跨机 |
| FSDP | PyTorch 原生的 ZeRO-3 | ZeRO-3 ≈ 1.5× DDP | 低（`fully_shard` 包裹） | 可跨机 |

三句话的数学落点都在例 B 的台账里：Megatron 的代价是"每层 4 次"，
FSDP/DeepSpeed 的代价是"每 step 1.5× DDP"。选型本质是在这两种
通信形态之间选——而 r2-Day15 已经算过，TP 的 4 次只认机内带宽
（NVLink 41.8ms vs IB 0.75s，18×）。

代码 `comparison_table_7b_n8` 输出本表 7B/N=8 口径的数字版。

## 2. 可手算例子 B：7B / N=8 台账对比

先 toy 热身（ $P=100$ ， $N=4$ ， $B_p=2$ ， $B_{opt}=8$ ）：
每参数 12B → DDP 每 rank $100 \times 12 = 1200$ B；
FSDP 切分 $1200/4 = 300$ B。
grad 字节 200B，ring all-reduce $2\times 3/4 \times 200 = 300$ B，
FSDP ≈ $1.5\times = 450$ B。
Megatron TP=4：params 200B/4 = 50B 每 rank（常驻分片）。

7B 实战口径（ $P=7\times 10^9$ ， $N=8$ ）：

| 方案 | 每 rank model states | 每 step 通信 |
|---|---|---|
| DDP（基线） | 14+14+56 = 84GB（>80GB，装不下） | grad all-reduce $2\times 7/8 \times 14 = 24.5$ GB |
| FSDP / DeepSpeed ZeRO-3 | 84/8 = 10.5GB | ≈1.5× = 36.75GB |
| Megatron TP=8 | params 14/8 = 1.75GB（常驻分片） | 每层 4 次 all-reduce（r2-Day15：117.4MB/次，80 层 = 37.6GB/step） |

三个数字互相咬合：FSDP 的 10.5GB 是 DDP 84GB 的 $1/N$ ；
36.75GB 是 24.5GB 的 1.5 倍（forward all-gather + backward all-gather +
reduce-scatter，r2-Day14 已验证系数）；Megatron 的 37.6GB 看似与 FSDP
同量级，但**形态完全不同**——它是 80 层 × 每层 4 次小消息，只认机内带宽，
跨机 IB 下会被放大 18×（r2-Day15 实测口径）。

代码 `test_7b_*` 断言了本节全部数字。

## 3. 可手算例子 C：选型三问 → 决策表

选型三问（代码 `select_framework` 的可执行版本，每个分支都被测试覆盖）：

1. **单卡吗 / <1B 吗？** → 三者都 overkill，`none`。单卡 + AMP / DDP 即够。
2. **跨机 + 20B+ 吗？** → `megatron`。TP（机内）+ PP（跨机）+ DP 的 3D 样板
   最成熟，通信模式固定可预期。
3. **已在 HF Trainer 生态吗？** → `deepspeed`。一行 config 开 ZeRO-3，
   代码侵入最小，还能顺手开 ZeRO-Offload / 1-bit Adam。
4. **要求最小代码侵入 + PyTorch 原生吗？** → `fsdp`。`fully_shard` 包裹即得
   ZeRO-3 语义（用完即 reshard），与 `torch.compile` / DTensor 生态最顺。
5. 都不是 → `fsdp` 兜底（原生 ZeRO-3，跨机单机都 work）。

注意分支顺序即优先级：跨机大模型优先于生态分支。
测试 `test_branch_priority_cross_node_over_hf` 锁定了这一点
（64 卡 + 70B + HF Trainer → 仍然 `megatron`）。

## 4. 何时不赚（反例，必读）

- **小模型单卡**：7B 以下单卡能装时，上 FSDP 的分片/通信机器纯属 overhead，
  收益为 0，迁移成本 > 0。
- **生态已绑定**：团队训练脚本深度基于 DeepSpeed config，硬切 FSDP 重写
  数据/优化器管线，省下的那点通信不值回票价。
- **把训练框架当推理框架用**：FSDP/Megatron 管的是训练，推理吞吐是
  vLLM/SGLang（第三层，r2-Day20+）的事，选型管不着。

## 术语与来源

- **FSDP（Fully Sharded Data Parallel）**：PyTorch 原生 ZeRO-3 实现，
  flat parameter + `reshard_after_forward`（用完即释放，显存峰值低，
  代价是每层多一次 all-gather）。来源：Zhao et al., "PyTorch FSDP:
  Experiences on Scaling Fully Sharded Data Parallel", arXiv:2304.11277。
- **ZeRO（Zero Redundancy Optimizer）**：按 stage 切分 optimizer states /
  grads / params，ZeRO-3 三者全切，通信 ≈ 1.5× DDP。
  来源：Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training
  Trillion Parameter Models", SC 2020, arXiv:1910.02054。
- **Megatron-LM**：NVIDIA 的 3D 并行（TP/PP/DP）生产实现，
  transformer engine 做算子融合。来源：Shoeybi et al.,
  "Megatron-LM: Training Multi-Billion Parameter Language Models Using
  Model Parallelism", arXiv:1909.08053。
- **DeepSpeed**：微软的训练优化库，ZeRO + ZeRO-Offload + 1-bit Adam 等；
  对 HF Trainer 一行 config 接入。来源：Rasley et al., "DeepSpeed: System
  Optimizations Enable Training Deep Learning Models with Over 100 Billion
  Parameters", KDD 2020；DeepSpeed 官方文档。

## 验证状态

- 已验证：Python 语法 + `unittest` 18/18 通过（纯 CPU，
  `framework_selection.py` + `test_framework_selection.py`）。
- 未验证：多卡 torch / NCCL / CUDA / H100 上的真实通信与显存
  （execution not validated，待 H100 验证）。本课状态：blocked。
- 公开 repo 无雇主标识。
