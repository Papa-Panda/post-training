# EP57 — Fast-dLLM：无需重训的扩散大语言模型推理加速
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP57-fast-dllm.html

> "Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6× throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs." —— 论文 Abstract 对整体结果的总结

## 元信息

- 期号：57
- 标题：Fast-dLLM：无需重训的扩散大语言模型推理加速
- BV：BV1qs4uz3EXL（B站有视频，但全站无任何字幕轨：浏览器内 4 次检查字幕列表均为空，已确认字幕不可得）
- 时长：00:48:50（2930 秒）
- 直播时间：2025-06-24 20:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：吴成岳（香港大学 MMLab 博士生，导师罗平、王文平；论文一作；官网预告嘉宾介绍）
- 相关论文：Chengyue Wu, Hao Zhang, Shuchen Xue 等（香港大学 / NVIDIA / MIT），*Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding*，https://arxiv.org/abs/2505.22618（v3, 2025-07-03）
- 相关代码：https://github.com/NVlabs/Fast-dLLM
- 官网预告：https://qingkeai.online/blog/tFYMAGAQ

> ⚠️ 提炼方式说明：本纪要**非逐字稿**。本期 B站视频无字幕轨（已 4 次复核确认），本纪要据该期对应的公开材料还原——官网预告（讲者与三段提纲：扩散大语言模型推理难点 / 分块 KV 缓存与置信度感知并行解码 / LLaDA 与 Dream 上的性能验证）+ Fast-dLLM 论文原文（arXiv:2505.22618）。talk 现场发挥、演示与 AMA 环节未覆盖，若现场口径与论文冲突无法核知；所有实验数字均为论文口径，逐项标注出处。

## 一句话总结

扩散大语言模型（dLLM）推理慢在两处：双向注意力让标准 KV Cache 用不了、一步并行解码多个 token 又掉质量。Fast-dLLM 全程不训练：一是块级近似 KV Cache（块内复用、块边界刷新，另有 prefix+suffix 的 DualCache 版），二是置信度阈值并行解码——根因定位为「条件独立假设破坏 token 间依赖」，解法是只解置信度过阈值的 token。两者叠加在 LLaDA / Dream 上达到最高 27.6 倍端到端加速，准确率损失在 1–2 个点内。

## 核心

### 背景/问题：dLLM 的并行潜力被推理层锁住

扩散 LLM（LLaDA、Dream 一类 masked diffusion 模型）理论上能一步生成多个 token，但开源实现实际推理速度反而落后自回归模型（论文 §1）。两个结构性原因：

1. **没有 KV Cache**：双向注意力下，每一步去噪所有位置互相 attend，KV 随上下文变化，没法像 AR 模型那样增量缓存，前缀计算每步重算。
2. **并行解码掉质量**：MDM 反向过程默认每步只改一个 token（ $\tau$ -leaping 近似才允许多 token 并行）；一旦一步采样多个 token，条件独立假设会破坏 token 间依赖——论文 §2.2 的例子：两个空位可以填 "full house"，独立采样却可能拼出 "high house"。并行的 token 越多，问题越严重（论文称之为 "Curse of Parallel Decoding"）。

### 方法/设计：块级近似缓存 + 置信度门控并行

**（1）块级 KV Cache（论文 §3.2）**：按 block 生成。生成当前 block 前，先算好并缓存其余 block 的 KV；block 内多步复用同一份缓存；block 完成后统一重算全部缓存。依据是 KV 激活在相邻步之间高度相似，所以「近似缓存」在实践中几乎无损。DualCache 进一步把 prefix 与（masked 的）suffix 都缓存，复用更彻底。

**（2）置信度感知并行解码（论文 §3.3）**：不用 LLaDA 式的固定 top-k 个 token/步，改成动态规则：只解置信度超过全局阈值 $\tau$ 的 token。置信度低的 token 恰是最可能互相依赖、需要上下文先定下来的位置，先跳过、等周围信息落实后再解——直接对症「条件独立破坏依赖」这个根因。论文后续版本还补了 factor-based 变体（按因子控制每步 token 数），吞吐比阈值版再高 1.4–1.5 倍（论文 Table 11）。

关键超参很省：缓存 block size 常用 32，置信度阈值常用 0.9（论文 §4.1）；全部 training-free，不动模型权重。

### 实验/实战（论文 §4）

设置：NVIDIA A100 80GB，模型为 LLaDA-Instruct、LLaDA-1.5、Dream-Base（多模态另测 LLaDA-V），基准 GSM8K、MATH、HumanEval、MBPP，生成长度 256/512（消融至 1024）。

- **单项收益**：只加 KV Cache 普遍 2–3.6 倍；只加并行解码常见 4–6 倍区间随长度增长（论文 §4.2）。
- **叠加收益（LLaDA）**：GSM8K 长度 512 达 11.0 倍吞吐（3.2 → 35.3 tok/s），准确率 77.5 → 77.2；GSM8K 长度 256 为 8.1 倍（6.7 → 54.4 tok/s)，准确率 79.3 → 78.5；MBPP 长度 512 为 9.2 倍（论文 Table 1）。
- **长生成上限**：8-shot、生成长度 1024 下 DualCache 达 27.6 倍端到端加速（0.7 → 19.3 tok/s)，且加速随长度单调放大（256 时 9.4 倍、512 时 15.8 倍）（论文 Table 5）。
- **多模态**：LLaDA-V 在 MathVista 上 9.9 倍、MathVerse 上 8.5 倍，准确率基本持平甚至略升（论文 Table 3）。
- **代价面**：论文自己点明高 batch 下 dLLM 仍追不上 AR 模型的吞吐，因为解码是 full attention、更吃算力（论文 §4.3 末段）。

### 结论/观点（论文口径，非 talk 现场原话）

- dLLM 落地的瓶颈主要在推理系统不在模型：缓存近似 + 依赖感知的并行解码能把与 AR 的速度差距大体抹平，且零训练成本。
- 并行解码的正确打开方式是「只解有把握的 token」，而不是固定步幅；条件独立假设是质量退化的根因诊断，后续同类加速工作基本围绕这一诊断展开。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| 端到端最高加速 | vanilla LLaDA | 27.6 倍（8-shot、长度 1024、DualCache） | 论文 Abstract、Table 5 |
| GSM8K 长度 512（LLaDA） | 3.2 tok/s、准确率 77.5 | 35.3 tok/s（11.0 倍）、准确率 77.2 | 论文 Table 1 |
| GSM8K 长度 256（LLaDA） | 6.7 tok/s、准确率 79.3 | 54.4 tok/s（8.1 倍）、准确率 78.5 | 论文 Table 1 |
| MBPP 长度 512（LLaDA） | 4.3 tok/s | 39.5 tok/s（9.2 倍） | 论文 Table 1 |
| 单项：KV Cache | 无缓存 | 普遍 2–3.6 倍 | 论文 §4.2 |
| 单项：并行解码 | 1 token/步 | 常见 4–6 倍，随长度增长 | 论文 §4.2 |
| LLaDA-V 多模态 | Full Steps | MathVista 9.9 倍、MathVerse 8.5 倍 | 论文 Table 3 |
| 准确率损失 | 全部设置 | 1–2 个点内，部分设置略升 | 论文 §4.2 |
| 常用超参 | — | block size 32、置信度阈值 0.9 | 论文 §4.1 |

注：以上均为论文口径；talk 现场演示的具体数字无法核知。官网预告摘要称「1024 token 长文本 27.6 倍、266 秒压至 12 秒、准确率损失 2% 以内」，与论文 Table 5 的 27.6 倍结论一致。

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **「只解高置信 token」是可移植的解码调度原则**：任何并行/投机解码场景，先想清楚质量退化的根因是否「独立假设破坏依赖」，再用置信度门控而非固定步幅；这与 RL rollout 中 speculative decoding、block-wise 生成的调参是同一类问题。
  2. **近似缓存的关键是定位「相邻步高相似」的张量**：Fast-dLLM 在块内冻结 KV 是近似推理能落地的形态，做 inference 优化时可以先量化 KV 跨步漂移再决定缓存粒度；
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. training-free 的系统优化在部署侧最友好（不碰权重、不重走训练验收）；评估新模型时应先把「推理层未优化」从「模型不行」里剥离。
  2. 加速比随长度/prefill 增长（9.4 → 27.6 倍）提示评测必须按长度分层报，短样本调通的结论不能外推长生成。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：DualCache 冻结 suffix 的 KV 在生成极长、全局依赖强的文本（如跨 block 的指代一致性）上误差如何累积，论文只给了基准分数，没有 failure case 分析；现场若演示过长文本生成，值得回看。
- 讲者现场内容不可还原：若 B站后续补挂字幕轨，应以字幕为准修订本纪要。后续同系工作（Fast-dLLM v2，arXiv:2509.26328，改为轻量微调的 block-diffusion 模型路线）与本期的 training-free 路线是可对照的两条演进线。

## 原文金句（1-2句）

> "we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption." —— 论文 Abstract，本期方法论的出发点诊断

> "The issue is more problematic when a large number of tokens are unmasked simultaneously in a single step." —— 论文 §2.2（Curse of Parallel Decoding），并行解码的边界判断
