# r2-Day16 — BF16 混合精度 / 梯度累积 / Activation Checkpoint：把每卡的张量再压薄

## Connection to Prev

r2-Day14 切的是 **model states**（通信换显存），r2-Day15 切的是**模型本身**
（TP/PP/SP 切张量、层、序列）。r2-Day16 不切分、不加卡，动的是**每个 rank 手里
剩下的张量本身**：把精度压薄（BF16/FP16 每个数只占 2 字节，Tensor Core 跑得更快）、
把 batch 拆小再累积（显存只按 microbatch 算）、把 activation 存一部分、
用的时候重算（计算换显存）。

**牺牲**：BF16 的 7 位尾数带来舍入误差界 `u ≈ 2^-8`——比更新量还小的权重增量会被
直接舍入掉；FP16 的 5 位指数让绝对值小于 `6.10e-5` 的梯度下溢成 0，必须靠
loss scaling 抢救；activation checkpoint 用多一次 forward（约 +33% 计算）换显存，
实现上还要处理 RNG 状态与 in-place 操作。**换取**：同 batch 下 activation 字节减半、
同显存下 batch 放大；梯度累积让有效 batch 放大 `m` 倍而显存只按 microbatch 算，
DDP 每样本分摊的通信量 ÷m；checkpoint 把 activation 内存从 `O(L·A)` 压到
`O(√L·A)`；低精度 GEMM 跑在 Tensor Core 上更快。**何时不赚**：小模型单卡随便装时，
AMP 的 fp32 master copy 把 params+grads 省下的 28GB（7B 口径）又加了回来，
model-state 总账与纯 fp32 完全相同（112GB），7B 照样装不进 80GB 卡；
计算已是瓶颈时 recompute 的 +33% 不划算；batch 本来就够大时梯度累积只是拖慢
optimizer step。

抄百科类比：混合精度 = 草稿用小字抄（bf16），定稿用大字描一遍存档（fp32 master）；
梯度累积 = 分四批抄完再统一交卷，交卷只交一次（all-reduce 一次）；
checkpoint = 只在每章开头折角，中间页看完就撕，反查时照着折角重抄一页。

## 0. 符号表（每个字母的含义）

- $L$ ：层数； $A$ ：每层 activation 的字节数（bytes/layer）
- $b$ ：microbatch 大小（每个 microbatch 的样本数）；
  $m$ ：梯度累积步数（一次 optimizer step 之前累积的 microbatch 数）；
  注意本课的 $m$ 与 r2-Day15 的 PP microbatch 数 $m$ 不是同一个 $m$
- $B_{eff} = b\cdot m$ ：有效 batch 大小（optimizer 实际看到的 batch）
- $s$ ：loss scale 因子（把 loss / 梯度整体放大 $s$ 倍再反向）
- $p$ ：显式尾数位数（不含隐藏的 leading 1）； $e$ ：指数位数
- $\text{eps} = 2^{-p}$ ：machine epsilon， `1.0` 处相邻可表示数的间距
- $u = \text{eps}/2$ ：unit roundoff，舍入到最近时的最大相对误差界
- $k$ ：checkpoint 间隔（每 $k$ 层存一个 checkpoint；反向重算段长度也是 $k$ ）
- 本课 GB 均为十进制（与 r2-Day15 一致）

## 1. 可手算例子 A：三种浮点格式的位布局

| 格式 | 符号/指数/尾数 | eps $= 2^{-p}$ | $u$ | 最小正规格数 | 最大值 |
|---|---|---|---|---|---|
| FP32 | 1+8+23 | `1.19e-7` | `5.96e-8` | `1.18e-38` | `3.40e38` |
| BF16 | 1+8+7 | `0.0078` | `0.0039` | `1.18e-38` | `3.39e38` |
| FP16 | 1+5+10 | `9.77e-4` | `4.88e-4` | `6.10e-5` | `65504` |

两个关键观察。第一，**BF16 就是 FP32 砍掉低 16 位尾数**：指数 8 位完全相同，
动态范围（`1.18e-38` 到 `3.39e38`）与 FP32 一致——这就是 BF16 时代 loss scaling
基本退役的原因。第二，FP16 的指数只有 5 位，最小正规格数 `6.10e-5`，
最大值 `65504`：动态范围窄是它一切数值麻烦的根因。

为什么 7 位尾数"够"？神经网络对**指数**（动态范围）敏感、对**尾数**（精度）
不敏感，这是 bfloat16 设计时的经验结论（见来源 Wang & Kanwar）。
但尾数不是没有代价：代码 `round_to_bf16` 模拟了真实舍入——权重 `1.0` 加上更新
`1e-4`（小于 `u/2`），舍入回 `1.0`，**更新被吞掉**；更新 `0.01` 则落到
`1.0078125`（ $1 + 2^{-7}$ ），保留。这就是 optimizer 状态与 master weights
必须留在 fp32 的原因：更新量常常比 bf16 在权重尺度上的 spacing 还小。

代码 `test_dtype_bit_layout` / `test_machine_epsilon` / `test_unit_roundoff` /
`test_min_normal` / `test_dtype_max` / `test_bf16_rounding_*` 断言了本节全部数字。

## 2. 可手算例子 B：FP16 下溢与 loss scaling

梯度 `g = 3.0e-5`：小于 FP16 最小正规格数 `6.10e-5`，在 fp16 里**直接冲成 0**，
梯度信号消失。loss scale `s = 2^15 = 32768`：

$$g\cdot s = 0.98304$$

落在 `[6.10e-5, 65504]` 区间内，可表示。完整链条：`loss·s` → 反向
（梯度自动带上 ×s）→ 在缩放域内做 all-reduce → 检查 inf/nan
（上溢则丢弃本步、缩小 $s$ ）→ 除以 $s$ → 还给 optimizer 干干净净的 fp32 梯度。
动态 loss scaling：连续多步无上溢则 $s$ 翻倍，省得人工调——代码
`dynamic_loss_scale_update` 模拟了"放大"与"上溢回退"两个分支
（`3.0 × 32768 = 98304 > 65504` 触发回退）。

同一梯度在 BF16 下：最小正规格数 `2^-126 ≈ 1.18e-38`，`3e-5` 毫无压力，
无需 loss scaling。**选 BF16 还是 FP16，本质是在选"要不要为下溢操心"**。

代码 `test_fp16_underflow_and_loss_scale` / `test_dynamic_loss_scale_*` 断言了
`32768`、`0.98304` 与回退逻辑。

## 3. 可手算例子 C：梯度累积

`b=2, m=4` → $B_{eff} = 8$ 。标量 toy：8 个样本的梯度
`[0.1, 0.2, ..., 0.8]`，全 batch 求和 `= 3.6`；
分成 4 个 microbatch，每 micro 求和 `[0.3, 0.7, 1.1, 1.5]`，
累加 `= 3.6`——数学上精确相等（Python 浮点实现误差 < 1e-12，
测试用 `assertAlmostEqual` 锁定）。

mean-reduction 的坑：每个 micro 内部先平均得 `[0.15, 0.35, 0.55, 0.75]`，
直接相加得 `1.8`，是全 batch 均值 `0.45` 的 `m = 4` 倍——
**累积后必须再除以 $m$ **。DDP 下每 optimizer step 只做一次 all-reduce，
每样本分摊的通信量是 `b = 2` 时的 `1/4`。

注意：梯度累积 ≠ 大 batch 的全部语义。BN 的 batch 统计、dropout 的随机性
仍按 microbatch 走；且 optimizer step 变慢 $m$ 倍——它省的是显存，
不是时间。

代码 `test_grad_accum_*` 断言了 `[0.3, 0.7, 1.1, 1.5]`、`3.6`、`0.45` 与通信 ÷m。

## 4. 可手算例子 D：activation checkpoint

$L = 64$ ， $A = 128$ MB/layer。朴素反向存下全部 activation：

$$64 \times 128\text{MB} = 8192\text{MB} = 8.192\text{GB}$$

每 $k = 8$ 层存一个 checkpoint：存 8 个 checkpoint，反向时从最近的 checkpoint
重算段内 8 层，内存 `= (8+8)×128MB = 2048MB = 2.048GB`，
**省 4 倍**。一般公式：内存 $= (\lceil L/k\rceil + k)\cdot A$ ，
对 $k$ 求极小得 $k = \sqrt{L}$ ，最小值 $2\sqrt{L}\cdot A$ ——
代码对 $k = 1, 2, 4, 8, 16, 32, 64$ 实测扫描，最小值确实落在 $k = 8$ 。

代价：反向时每个段多做一次 forward；一次训练迭代 forward:backward $= 1:2$ ，
额外计算占比 $= 1/3 \approx 33\%$ 。Chen et al. 的论文里，
1000 层 ResNet 从 48GB 压到 7GB，只多花 30% 时间——量级与我们的 `1/3` 一致。
这就是"用计算换显存"的定价：显存从 $O(L)$ 降到 $O(\sqrt{L})$ ，
计算多付三分之一。

代码 `test_checkpoint_*` 断言了 `8192` MB、`2048` MB、4 倍、 $k = 8$ 最优、
额外计算 `1/3`。

## 5. 可手算例子 E：7B AMP 显存台账（诚实版）

7e9 参数，十进制 GB：

| 项 | AMP（bf16） | 纯 fp32 |
|---|---|---|
| 参数 | 14GB | 28GB |
| 梯度 | 14GB | 28GB |
| fp32 master | 28GB | —（参数本身即 fp32） |
| Adam fp32 m, v | 56GB | 56GB |
| **合计** | **112GB** | **112GB** |

`112GB > 80GB`：AMP 独自救不了 7B 上单卡。更诚实的一行：
**AMP 与纯 fp32 的 model-state 字节数完全相同**——params+grads 省下的 28GB，
被 fp32 master copy 的 28GB 精确抵消。AMP 真正给的是两样东西：
activation 字节减半（同 batch 更省 / 同显存 batch 更大）与 Tensor Core 低精度加速。
想上 80GB 卡，必须叠加 r2-Day14 的 ZeRO（把 112GB 切到 $N$ 卡）、
r2-Day15 的 TP/PP，或本课的 checkpoint——三条路正交，可叠加。

代码 `test_amp_ledger_7b` / `test_pure_fp32_ledger_7b` 断言了
`14/14/28/56/112` 与两份台账相等。

## 何时不赚（决策表）

| 场景 | 结论 |
|---|---|
| 模型+优化器状态单卡装得下，batch 已够 | 纯 fp32 或简单 AMP 即可；checkpoint 的 +33% 计算是纯 overhead |
| FP16 梯度大面积下溢 | 先换 BF16（指数 8 位与 fp32 同范围，一般无需 loss scaling）；FP16 才需要动态 loss scaling |
| 计算已是瓶颈 | recompute 把 step time 拉长约 1/3；先从别处压 activation（SP、更小的 b） |
| 只想放大有效 batch | 梯度累积： $B_{eff} = b\cdot m$ ，显存只按 b 算；代价是 optimizer step 慢 m 倍，且 BN/dropout 语义仍按 microbatch |
| 7B 单卡 80GB | AMP 独自 112GB 装不下——必须叠加 ZeRO / checkpoint / TP（Day14/Day15/本课 D） |
| 只做推理（前向） | 不需要 master copy 与 loss scaling；低精度权重直接跑（那是量化部署，不是 AMP 训练） |

## 来源

- Micikevicius et al., *Mixed Precision Training*, arXiv:1710.03740（ICLR 2018）：
  fp32 master weights + loss scaling。https://arxiv.org/abs/1710.03740
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174：
  O(√L) checkpointing，用一次额外 forward 把 activation 内存压到次线性。
  https://arxiv.org/abs/1604.06174
- Wang & Kanwar, *BFloat16: The secret to high performance on Cloud TPUs*,
  Google Cloud Blog, 2018：bfloat16 语义（1+8+7）与"神经网络对指数敏感、
  对尾数不敏感"的设计依据。
  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus

## 验证状态

- 本地： `python3 -m unittest discover` ，21/21 CPU 测试通过
  （位布局/eps 与 u/上下溢阈值/最大值/loss scaling 放大与收缩/
  动态 scale 增减/bf16 舍入三组/梯度累积求和与均值/每样本通信 ÷m/
  checkpoint 手算数与 k 扫描最优/重算 1/3/AMP 与 fp32 台账）。
- `tools/check_repo.py` 对新目录检查通过；新文件无雇主标识（grep 验证）。
- 未验证：无 PyTorch/CUDA/H100，所有字节数/加速比/ `1/3` 均为理论估计
  （theoretical estimate），非实测 benchmark；
  状态保持 blocked，execution not validated / 待H100验证。
