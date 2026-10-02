# EP52 — Sparse VideoGen：无需重新训练的 DiTs 推理加速框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP52-sparse-videogen.html

> 「它能够使得 kernel 的加速比和理论上基于 sparsity 算出来的加速比非常接近。」——讲者谈 layout transformation 对 temporal head 计算效率的修复

## 元信息

- 期号：52
- 标题：Sparse VideoGen：无需重新训练的 DiTs 推理加速框架
- BV：BV1xw4jzgENr
- 时长：00:53:17（字幕覆盖约 53 分钟，主持等待开场约前 5 分钟；讲授从 11 分钟起，之后为 Q&A）
- 提炼日期：2026-10-02
- 分享嘉宾：徐浩成（加州大学伯克利分校博士生；字幕中自述与主持人介绍一致，正式头衔以论文作者页为准）
- 相关论文：Sparse VideoGen——无需重新训练的视频扩散 Transformer（DiTs）推理加速框架；讲者称已被 SysML 2025 接收（字幕口径）；后续工作 Sparse VideoGen 2 提于 Q&A
- 相关代码：字幕中未给出
- B站链接：https://www.bilibili.com/video/BV1xw4jzgENr/
- 官网期号：第 52 期（官网预告链接省略）
- 字幕原文存档：本地 `transcripts/EP52.txt`（929 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP52，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如「spatial head」字幕作「special hand」、temporal head 作「TEMPORHEAD」、SysML 作「SML」、sparse video gen 作「sparse video g」，均以讲者口径与公开资料为准；讲者口头介绍会议时字幕写作「SML2025」，疑为 SysML；主持人开场介绍作「ICLO22025」，显系识别错误）。论文未在字幕中给出 arXiv 编号，故数字全部为讲授字幕口径。

## 一句话总结

讲者以「video DiT 的 attention 计算量占比随序列总长度快速膨胀」为起点，把视频生成的推理瓶颈定位到注意力；据此发现 attention head 天然分成两类——主管空间相关性的 spatial head（attention 集中在对角线附近的本地 tile 与首帧列）和主管时间一致性的 temporal head（attention 沿 3D 序列的同位置跨帧斜线规则分布）；Sparse VideoGen 的贡献，是给每个 head 在线、近乎零成本地选对该用哪类 mask，并把 temporal head 不规则的稀疏分布经 layout transformation 重排成 GPU 友好的紧凑小块，最终实现接近理论上限的 attention kernel 加速与特征上几乎无损的视频生成质量。

## 核心

### 背景：视频 DiT 中 attention 是计算大头，长序列时更夸张

视频扩散模型的骨架是 3D attention：sequence 同时沿空间 token（像素块）和帧（frame）两个维度展开。一个 30 帧、每帧 3000 token 的典型设置，sequence 长度直接到 90K。attention 计算量关于 sequence 长度平方增长，其他算子线性增长，所以序列越长 attention 占比越高——讲者给的比例是 17K → 51%、40K → 73%、120K → 82%。

在这种占比下，仅优化其余算子（MLP、normalization、采样 scheduling）改善的只是次要部分；要真正加速视频生成，必须直接压低 attention 峰值——Sparse VideoGen 因此把问题限定为——**不重训练、不改模型、不改扩散时间表，只靠跳过本就不重要的 attention 计算**。

### 发现：head 天然分 spatial 与 temporal 两类

讲者走的路子不是从通用稀疏 API 出发，而是先看 attention map 里固有的稀疏结构。他把全部 head 分成两类。

**Spatial head**：attention score 集中在对角线附近（同一帧内的本地 token + 邻近帧的对应 tile）、以及第一列（所有 token 都会关注首帧 token，相当于 LLM 的 attention sink 现象在时序上的对应物）。讲者的解释是这类 head 在维护局部空间信息——每个 token 不可避免地要像看它周围邻居那样看附近的 token。

**Temporal head**：attention score 分布是稀疏的对角线条纹——同一个空间位置在不同帧的 token 之间互相 attend，在由帧×空间构成的 3D attention map 上呈现斜线规律。讲者认为这类 head 在维护时间一致性（同一个物体在各帧间连续出现、避免闪烁跳动）。

讲者强调这条分法不是人为划分的稀疏 mask，而是模型内部真实存在的行为分化；知道 head 是空间型还是时间型，才能给每个 head 施以恰当的稀疏化方案。

### 方法：在线识别 head 类型 + 稀疏 mask + layout transformation

一旦能正确分类 head，后续处理拆成两块。

**在线识别 head 类型（online profiling）**。讲者先给出离线方案：跑一遍 dense attention、算两种 sparse map 的输出、按 MSE 选更近的；效果几乎无损（PSNR ≈ 30），但它**需要先跑 dense attention**，把前向传播的大头算完了所谓加速也就无从谈起。线上策略则把「先跑一遍」降到「只抽少量 query token 试探」：在每个 head 内部从全部 query 里随机取约 32 token（讲者举例 20K 里取 32，约 1%）计算两种稀疏 map 下的结果并比较，从而判定 head 类型。代价仅约 1% 量级。

**Layout transformation（解决 temporal head 的计算不效率）**。讲者点出 dense 稀疏 map 可以「识别出应该算的块」并不等于高效：temporal head 的斜线分布在内存上不连续——即便 density 降到 20%，理论加速 5 倍，实际往往只有 2–3 倍。其根源是斜线型布局与 GPU 上连续内存访问的不匹配。

讲者给的解法是 orthogonal 重排：按“token 位置”而不是帧序组织序列——先排帧 1–F 中“位置 1”的所有 token，再排“位置 2”的所有 token……这样同一个位置在不同帧的 token 变成一排连续的小块，temporal attention 的斜线在新排布里变成对角线附近的紧凑小块，GPU 内核即可按规则顺序装载。讲者同时强调这步重排是可逆的：算完再施加逆变换恢复到原来的 attention output 排布，工程上还与 text embedding/token 兼容——只对 video token 用 transformation，text token 保持不动。

**系统侧配套**。两条零星但重要的工程措施：把稀疏 attention 集成到现有高效内核之上（Q&A 澄清是基于 FlashAttention 做 sparse attention，并非重新写 kernel）；同时讲者还发现很多 video 模型里 QK norm（字幕作 SuperNorm）与 RoPE（字幕作「肉」）这些次要算子耗时不低，用 kernel fusion 合成一个卷一起的算子包能拿到 8–16 倍该级别算子的提速。

这套组合搭配的具体评估数字在下一节列出。

### Q&A 要点

- **和 XAttention 的区别**：讲者认同两者思路相近、甚至交流过；XAttention 可看作一种让 sparse 识别「该算哪些块」的通用方法，但识别完仍面临不连续内存访问的风险；Sparse VideoGen 不是靠稀疏计数规则，而是通过 layout transformation 把 temporal head 的计算真正落地，并且是为 video diffusion 的固有冗余结构设计。
- **跳过 denoising steps（如 TeaCache）正交**：与 TeaCache 完全兼容——TeaCache 从 denoising step 数量上做跳步，Sparse VideoGen 在单步 attention 内部做稀疏，互不冲突。
- **是否一定能二分类**：讲者承认有些 head 既不像纯 spatial 也不像纯 temporal——这也是他们做后续工作 **Sparse VideoGen 2** 的起因之一——新版对这类混合型 head 的近似更好。
- **mask 的时间-空间统一**：讲者解释 transformation 把 temporal head 拉回对角线附近后、规则化后的 temporal 与 spatial 的 mask 形式基本一致，实践时对 temporal head 施加 transformation、spatial head 不施加。
- **attention mask 的存储**：理论存储是 ， $N \times N$ ， 讲者用「大规模时靠 query token 索引压缩（如 1000–5000 条上限）」把 profiling 用的 mask 锁到 1GB 以内点到即止——承认真实 query 数量 20K 级别时要做约束。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 30 帧 × 3000 token 的 sequence 长度 | — | 约 90K | 字幕（讲授部分） |
| attention 占比（17K 序列） | — | 约 51% | 字幕（讲授部分） |
| attention 占比（40K 序列） | — | 约 73% | 字幕（讲授部分） |
| attention 占比（120K 序列） | — | 约 82% | 字幕（讲授部分） |
| offline profiling 识别 head 类型的质量 | PSNR | ≈ 30（几乎无损） | 字幕（讲授部分） |
| online profiling 抽样比例 | 20K query 中抽取 | 32 个 query token（约 1%） | 字幕（讲授部分） |
| 稀疏 density 20% 时的期望加速 | naive sparse kernel 2–3 倍 | 期望 5 倍 | 字幕（讲授部分） |
| attention kernel 整体加速区间 | — | 2.5–3.6 倍 | 字幕（讲授部分） |
| 端到端视频生成加速 | dense baseline | 约 2.33 倍（字幕口径） | 字幕（讲授部分） |
| PSNR（Sparse VideoGen） | 现有 baseline 约 22–23 | ≈ 29 | 字幕（讲授部分） |
| SSIM（Sparse VideoGen） | 现有 baseline 约 0.7–0.8 | > 0.9 | 字幕（讲授部分） |
| QK norm / RoPE 的 kernel fusion 加速 | 旧 PyTorch 实现 | 8–16 倍 | 字幕（讲授部分） |
| 附加 profiling 开销 | 占 sparse attention 总耗时 | 约 10–20% | 字幕（Q&A） |
| profiling 用的 attention mask 存储 | ， $N \times N$ ， | 约 1GB 以内（query 索引上限 1000–5000） | 字幕（Q&A） |

## 可迁移

- **选 head 前先 profile、别拍脑袋**：对于任何 transformer-based 的推理加速工作，「假设所有 head 都适合同一种稀疏结构」都可能过粗；讲者用极小算力（≈1% query token）两秒式地把每个 head 按它真正的行为定型，可以考虑把这种小额 online sampling 套用到 LLM 推理加速和 MoE 专家视角的 profile 里。
- **layout transformation 是个正交技巧**：稀疏结构的有效性与计算效率是两回事——只要访问模式与内存排布不匹配，FLOPS 优势就会被加载开销抵消。把「不连续块」重排成「连续行」先算、算完再恢复的模式，比尝试写更聪明的 kernel 更先验有效，与 EP50 中 MoLE「先重参数化、再查表」是同构的工作模式。

## 疑问 / 下一步

- 既然 Q&A 已经承认存在第三类 head（既非纯 spatial 也非纯 temporal），Sparse VideoGen 2 为此做了哪些具体修改暂时没有完整材料，建议后续补录。
- PSNR ≈ 29（字幕略带「PSNR 是不是该 28.5 左右」的口头摇晃）是与 baseline 相对而非绝对保真上限；若要实际用于端到端评测，还要再看 VBench/LIPS 等更细维的指标。
- layout transformation 在超长视频（比如 1000 帧）下是否仍稳定、profiling mask 的 1GB 上限在 MoE 化的新 video 模型上是否需要新的压缩策略，未在本次 Q&A 中展开。

## 原文金句（1-2句）

> 「它能够使得 kernel 的加速比和理论上基于 sparsity 算出来的加速比非常接近。」——讲者谈 layout transformation 对 temporal head 计算效率的修复

> 「如果这些 head 的计算不够好，生成出来的视频它的时空一致性就不会很好。」——讲者把 temporal head 与视频质量的契约摊开解释

> 「有一些 head 既不能太归类到 spatial head，也不太能归类到 temporal head。」——讲者在 Q&A 中自述后续把这类 head 当作第二版工作的独立议题
