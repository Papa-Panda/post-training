# r2-Day17 NOTES — 术语、公式核对与上一课/下一课链接

## 术语（准确定义）

- **FSDP（Fully Sharded Data Parallel）**：PyTorch 对 ZeRO-3 的原生实现。
  核心机制：flat parameter（把多层参数拍平成一维大 tensor 再切分，减少
  all-gather 碎片）+ `reshard_after_forward`（forward 用完参数分片立即释放，
  backward 再 gather）。效果：显存峰值低；代价：每层 forward/backward
  各一次 all-gather，通信 ≈ 1.5× DDP。
- **ZeRO stage**：ZeRO-1 只切 optimizer states；ZeRO-2 加切 grads；
  ZeRO-3 连 params 也切。FSDP 默认即 ZeRO-3 语义；DeepSpeed 允许按 stage
  选，config 里一行切换。
- **reshard_after_forward**：FSDP 的标志性 trade——"用完即还"。
  对比 Megatron TP 的"常驻分片"：TP 分片一直在显存里（通信模式固定），
  FSDP 分片用完就扔（显存峰值低，但 gather 更频繁）。这是讨论难点的题眼。
- **3D 并行**：TP（张量内切，通信密集限机内）× PP（层间切，有 bubble
  $ (p-1)/(m+p-1) $ ）× DP（数据并行）。Megatron-LM 是 3D 的生产样板；
  FSDP/DeepSpeed 本质是 DP 的显存优化版（ZeRO），不是 3D。
- **1-bit Adam / ZeRO-Offload**：DeepSpeed 大礼包里的两件：
  前者把 optimizer 通信压缩（误差补偿），后者把 optimizer states / params
  卸载到 CPU/NVMe（PCIe 带宽换显存）。本课只记名字，数学在选型时知道
  "有这个选项"即可。
- **代码侵入（code intrusion）**：从现有训练脚本切到某框架要改多少行。
  排序（低→高）：DeepSpeed config（一行）≈ FSDP `fully_shard` 包裹
  < Megatron（按 3D 重组模型与脚本）。

## 公式核对（与 r2-Day03/Day14/Day15/Day16 一致性）

- ring all-reduce $ 2(N-1)/N \times $ payload：r2-Day03 公式，本课
  `ring_allreduce_bytes` 原样复用。DDP grad 通信 24.5GB
  （ $2\times 7/8 \times 14$ ）与 Day14 的 DDP 口径一致。
- FSDP 1.5× 系数：r2-Day14 的 toy（P=100, N=4）已验证
  ZeRO-3 通信 = 1.5 × DDP。本课 toy 用同一 P/N，450B = 1.5 × 300B，
  系数咬合。
- 7B 台账 14/14/56 = 84GB：r2-Day14 的 ledger 原样引用；
  r2-Day16 的 AMP 台账 112GB 是另一口径（fp32 master + fp16 param），
  本课用 fp16 param + fp32 opt 的 12B/param 口径，README 已声明，
  不要与 Day16 的 16B/param 混用。
- Megatron 每层 4 次 all-reduce：r2-Day15 口径（forward 2 + backward 2），
  117.4MB/次、80 层 37.6GB/step 直接引用 Day15，不在本课重算。
- TP 的"只认机内带宽"：r2-Day15 的 NVLink 41.8ms vs IB 0.75s（18×），
  是本课讨论难点的定价依据。

## 与上一课的连接

r2-Day16 回答"每卡手里的张量还能怎么压"（精度/累积/重计算，不换并行结构）。
r2-Day17 回答"压完之后跑在哪个框架上"：框架 = 并行语义 + 代码侵入形态
的打包。FSDP 恰好是 Day14 ZeRO-3 的生产形态；Megatron 恰好是 Day15
3D 并行的生产形态；DeepSpeed 是"不想改代码也想吃 ZeRO 红利"的形态。
三课的台账互相咬合：Day14 算出台账，Day16 压出台账，Day17 把台账
装进框架——选型三问本质是在问"你愿意为什么付迁移成本"。

## 下一课预告

r2-Day18：Checkpoint & Recovery（DCP）。框架选好、训起来之后，
下一个工程问题是"挂了怎么恢复"：full rank0 gather vs sharded DCP
per-rank，含 optimizer + epoch + rng。r2-Day16 的 checkpoint（存 activation）
和 Day18 的 checkpoint（存训练状态）名字一样、对象完全不同，注意区分。

## 讨论难点（供 ai chat）

FSDP 的 reshard_after_forward（用完即释放、显存峰值低、通信 1.5×）
vs Megatron TP 分片常驻（通信模式固定 4 次/layer、要求机内 NVLink）：
如果只有跨机 IB（无 NVLink）的 8 卡环境跑 7B，你选 FSDP 还是 TP=8？
为什么？提示：用 r2-Day15 的 18× 带宽比给两种方案的通信定价。
