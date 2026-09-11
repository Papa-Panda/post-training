# r2-Day16 NOTES — 术语、公式核对与上一课/下一课链接

## 术语（准确定义）

- **bfloat16**：1 符号位 + 8 指数位 + 7 显式尾数位。就是 FP32 砍掉低 16 位尾数，
  指数范围与 FP32 完全相同（`1.18e-38` 到 `3.39e38`）。名字来自 Google Brain。
- **IEEE fp16（half）**：1 + 5 + 10。指数只有 5 位，动态范围窄
  （`6.10e-5` 到 `65504`），是下溢/上溢麻烦的根因；尾数 10 位比 BF16 精细。
- **Master weights（fp32 主权重）**：AMP 在 fp32 里维护的一份全精度参数 copy。
  前向/反向用低精度跑，optimizer 的 `lr × grad` 更新在 fp32 里做完再 cast 回
  低精度——因为更新量常常比低精度在权重尺度上的 spacing 还小（本课例子 A
  的 `1.0 + 1e-4 → 1.0`）。
- **Loss scaling**：把 loss 放大 $s$ 倍再反向，让小梯度在 fp16 下不下溢；
  all-reduce 后在 fp32 里除以 $s$ 还原。静态 $s$ 需手调；
  **动态 loss scaling**：无上溢则 $s$ 翻倍，上溢则 $s$ 减半并跳过本步。
- **Machine epsilon（eps）**： `2^-p` ， `p` 为显式尾数位，
  `1.0` 处相邻可表示数的间距。**Unit roundoff（ $u$ ）**： $\text{eps}/2$ ，
  舍入到最近时的最大相对误差界。两者差 2 倍，别混用。
- **Gradient accumulation（梯度累积）**： $m$ 个 microbatch 的梯度累加后
  再做一次 optimizer step， $B_{eff} = b\cdot m$ 。省显存不省时间；
  BN/dropout 的 batch 语义仍按 microbatch 走。
- **Activation checkpointing / rematerialization（重计算）**：
  前向只存每 $k$ 层一个 checkpoint，反向时从最近 checkpoint 重算段内 activation。
  内存 $(\lceil L/k\rceil + k)\cdot A$ ， $k = \sqrt{L}$ 最优 → $2\sqrt{L}\cdot A$ 。
- **Sublinear memory（次线性内存）**：activation 内存从 $O(L\cdot A)$
  降到 $O(\sqrt{L}\cdot A)$ ，代价是一次额外 forward（约 +1/3 计算）。

## 公式核对（与 r2-Day03/Day14/Day15 一致性）

- AMP 每参数字节 $2 + 2 + 4 + 4 + 4 = 16$ B/param（fp16/bf16 参数 + 梯度 +
  fp32 master + fp32 m + fp32 v），7B → 112GB：与 r2-Day14 第 4 节
  "mixed precision 的 16B/param" 同一台账，本课 `amp_memory_ledger_gb`
  复算一致。纯 fp32 口径 $4 + 4 + 4 + 4 = 16$ B/param（master 即参数本身），
  7B → 112GB——Day14 没给纯 fp32 台账，本课补上。
- ZeRO 切分的对象正是这 16B/param 的 model states；AMP 不改变总量，
  只改变 params+grads 的 2B/4B 分布——与 Day14 的分母一致，可以直接套
  ZeRO-1/2/3 的分片公式。Day14 NOTES 明说"ZeRO 不切 activations
  （那是重计算 r2-Day16 的活）"：两课互补，Day16 的 checkpoint 负责
  activations，Day14 的 ZeRO 负责 model states。
- 本课的 $m$ （梯度累积步数，时间方向）与 r2-Day15 的 $m$
  （PP microbatch 数，pipeline 深度方向）**符号复用、含义不同**，
  README 符号表已声明。交叉引用时注意区分。
- r2-Day03 的 ring all-reduce 公式在本课依然成立：梯度累积下单次 all-reduce
  的 payload 不变（还是全部梯度字节），变化的是**频率**——每 $m$ 个 microbatch
  只做一次，每样本分摊通信量 ÷m（代码 `ddp_comm_per_sample`）。
- FP16 下溢阈值 $2^{-14} = 6.10\text{e-}5$ 、loss scale $2^{15} = 32768$ 、
  $g\cdot s = 0.98304$ ：阈值与缩放量的 2 的幂次形式互相咬合，
  不是凑出来的巧合—— $s$ 取 2 的幂是为了缩放本身不引入舍入误差。

## 与上一课的连接

r2-Day14 回答"model states 太大怎么办"（ZeRO：通信换显存，不碰 activations）。
r2-Day15 回答"模型本体太大怎么办"（TP/PP/SP：切张量、层、序列）。
r2-Day16 回答"每张卡手里的张量还能怎么压"：精度压薄（BF16/FP16）、
batch 拆小再累积（时间换空间）、activation 存一部分重算一部分（计算换显存）。
三课正交且常叠加：7B 上 80GB 卡的现实解法就是 AMP + ZeRO-1/2 +
activation checkpoint 三件套——Day14 切 model states，Day16 压 activations，
谁也不替代谁。

## 下一课预告

r2-Day17：选型课。把 Day14（ZeRO）、Day15（TP/PP/SP）、Day16
（AMP/累积/checkpoint）拼成一张决策表：什么规模用什么组合、
通信/计算/显存三者的预算怎么分、DeepSpeed / FSDP + TP/PP 混用的现实形态。
