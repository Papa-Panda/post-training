# EP106 — Intern-S1：科学多模态基础模型（B站实际视频）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP106-intern-s1.html

## 元信息

- 期号：106（官网编号；本期存在编号冲突，见下文专节）
- 标题：Intern-S1: A Scientific Multimodal Foundation Model（B站视频实际标题口径）
- BV：BV1FY6sBuECe
- 时长：未取得（B站接口不可达）
- 提炼日期：2026-10-02
- 分享嘉宾：未确认（官网 106 期预告对应的是另一场 talk，见编号冲突专节；本场讲者待补）
- 相关论文：Intern-S1 Team, Shanghai AI Laboratory，https://arxiv.org/abs/2508.15763
- 相关代码/模型：https://huggingface.co/internlm/Intern-S1
- B站链接：https://www.bilibili.com/video/BV1FY6sBuECe/
- 官网预告（编号对应另一场）：https://qingkeai.online/blog/GDPO-talk

> ⚠️ 提炼方式说明：本期无字幕（B站字幕接口在本环境不可达）。本纪要以 B站该 BV 的实际视频内容（Intern-S1）为准，根据论文原文（arXiv:2508.15763）还原，非逐字稿；Q&A 即兴内容未覆盖。信息来源：论文还原。

## 编号冲突说明（106 期）

官网枚举的第 106 期是《GDPO：解决 GRPO 在多奖励 RL 训练中的"优势崩溃"问题》（讲者：香港科技大学博士候选人、NVIDIA 研究实习生刘诗扬；论文 arXiv:2601.05242，代码 https://github.com/NVlabs/GDPO ）。但合集中该期号对应的 B站视频（BV1FY6sBuECe）实际内容是 **Intern-S1**，GDPO 一期的视频未在合集出现。按「以视频实际内容为准」的原则，本纪要写 Intern-S1；GDPO 的要点附记于此：GRPO 在多奖励下先求和再 group-wise 归一化，会把不同奖励组合压成相同 advantage（advantage collapse），训练信号分辨率下降；GDPO 改为逐奖励解耦归一化再聚合、并做 batch 归一化稳定尺度，在工具调用、数学推理、代码推理三类任务上一致优于 GRPO。

## 一句话总结

Intern-S1 是上海 AI Lab 的「专才型通才」（specialized generalist）：241B 总参 / 28B 激活的多模态 MoE，在 5T token（其中 2.5T+ 科学数据）上持续预训练，后训练用 InternBootCamp 做离线 + 在线 RL，并以 Mixture-of-Rewards（MoR）在 1000+ 任务上协同训练——目标是把开源模型在科学专业领域与闭源 SOTA 的差距补上。

## 核心

### 背景/问题

通用基础模型在热门领域已接近闭源，但在高价值科学专业领域（分子合成规划、反应条件预测、晶体热力学稳定性等）要么依赖专家模型，要么通用模型明显落后。Intern-S1 的问题设定：能不能用一个模型同时保持通用理解/推理，又吃下多模态科学数据（文本、图像、分子、晶体等）？

### 方法/设计

- **底座**：多模态 MoE，241B 总参数、28B 激活参数；持续预训练 5T tokens，科学域数据占 2.5T 以上。
- **后训练两段**：先 offline RL 再 online RL，统一在 InternBootCamp 环境中进行。
- **Mixture-of-Rewards（MoR）**：1000+ 任务同时 RL 时，用奖励混合机制协同多任务信号，避免逐任务单独训练的系统开销与信号冲突——这是和多奖励归一化问题（对照 GDPO 的 advantage collapse）同源的工程关切：在多奖励/多任务下保住训练信号的分辨率。
- **系统**：算法、数据、训练系统三侧协同，论文称其在线 RL 训练性能达到 top-tier。

### 实验/实战

论文口径：通用推理基准上在开源模型中具竞争力；科学域显著超过开源模型，并在分子合成规划、反应条件预测、晶体热力学稳定性预测等专业任务上超过闭源 SOTA（具体分数以论文表格为准，本纪要未逐项抄录，避免转引误差）。

## 关键数字

| 指标 | 数值 |
|---|---|
| 总参数 / 激活参数 | 241B / 28B |
| 持续预训练 token | 5T（科学域 >2.5T） |
| MoR 协同 RL 任务数 | 1000+ |

## 可迁移

- 多任务 RL 的奖励侧设计先行：任务数上千时，奖励的归一化与混合方式（而不是单个奖励函数本身）决定信号分辨率——与 GDPO 的逐奖励解耦归一化是同一课。
- Infra 视角：科学多模态数据的模态对齐与 token 记账（分子/晶体等结构化模态如何进同一条序列）是数据管道的真成本，评测要按专业任务单独建集，通用基准测不出科学域差距。

## 疑问 / 下一步

- 本场 talk 的讲者与具体分享提纲未确认（官网 106 期预告对应 GDPO 场）；Intern-S1 的 MoR 细节（奖励如何加权、任务采样分布）值得回论文正文深挖。
- InternBootCamp 的环境抽象与 Intern-S1-Pro（万亿参数续作，arXiv:2603.25040）的关系可后续对照。

## 原文金句（1-2句）

> "a specialized generalist equipped with general understanding and reasoning capabilities with expertise to analyze multiple science modal data." —— 论文对 Intern-S1 的自我定位（摘要原文，非视频原话）。
