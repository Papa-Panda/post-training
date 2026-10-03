# EP94 — RLinf-VLA 实践：从零上手 VLA（OpenVLA）强化学习

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP94-rlinf-vla.html

> "Recent studies have demonstrated the potential of reinforcement learning (RL) to improve the task performance of vision-language-action (VLA) models through interaction. However, current efforts remain fragmented, lacking a unified platform for fair comparison across architectures and algorithms." —— 论文摘要，本期主线：VLA+RL 之前缺的不是算法，是平台

## 元信息

- 期号：94
- 标题：RLinf-VLA 实践：从零上手 VLA（OpenVLA）强化学习
- BV：BV1CF24BcEwb（台账中该 BV 旧登记为 B94《为具身智能构建动态本体自进化记忆框架》，与本场实际内容完全不符；本场为官网 EP94，B站字幕接口持续串台、无法取得本场字幕，详见下方提炼方式说明）
- 时长：直播计划 1 小时（讲授 + AMA），字幕不可得、无以字幕计的时长可核
- 直播时间：2025-12-02（周二）20:00–21:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：臧宏之（Hongzhi Zang，清华大学交叉信息研究院本科生，RLinf 框架 VLA 部分核心开发人员之一；RLinf-VLA 论文第一作者、Project Lead）
- 相关论文：Hongzhi Zang et al., *RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models*，https://arxiv.org/abs/2510.06710（v1 2025-10-08，后更新至 v3）
- 相关代码：https://github.com/RLinf/RLinf ；RLinf 系统论文（母框架整体设计）：*RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation*，https://arxiv.org/abs/2509.15965
- 官网预告：https://qingkeai.online/blog/RLinf-VLA

> ⚠️ 提炼方式说明：**字幕接口持续串台，经多轮实测仍无法获得本场字幕；本篇基于对应论文还原，非逐字稿。** 讲授提纲为：（1）RLinf-VLA 的设计思路与系统架构；（2）关于 VLA+RL 的算法技术设计：PPO / GRPO 等；（3）OpenVLA 的微调实践；（4）AMA 环节。其中（1）（2）（3）分别可由论文（统一接口、GPU 分配三模式、混合流水线；PPO/GRPO 与消融最佳实践；OpenVLA/OpenVLA-OFT 实验）还原，以下以论文为据写；AMA 与讲者现场口头发挥无书面材料，不复述、不虚构。凡数字均为论文口径（标注 arXiv 出处）。**编号冲突专节**：B站合集内 BV1CF24BcEwb 在仓库台账里登记为 B94《为具身智能构建动态本体自进化记忆框架》（bilibili-only 行），这是台账旧标题、与实际视频内容（官网 EP94 RLinf-VLA 实践）完全不符；官网 episode-index 中 EP94 本登记为 official-only（「官网有预告，B站合集无对应视频」），实际对应视频即此 BV。按既定裁决「纪要按实际视频内容编号」，本篇归 EP94；B94《动态本体自进化记忆框架》一期是否另有真实 BV，待后续枚举合集复核。

## 一句话总结

VLA 用 RL 后训练的共识迟迟没兑现成基线，部分原因是生态碎片化：每家工作的模型、算法、仿真器各不相同，既没法公平比较，也没法复用工程。RLinf-VLA 的答案是先把平台做出来：统一接口把多种仿真器（ManiSkill/LIBERO/RoboTwin）、多种 VLA 架构（OpenVLA/OpenVLA-OFT 等）与两套 RL 算法（PPO/GRPO）收进同一套系统，再用三档灵活的 GPU 分配策略（共存 / 分离 / 混合）加细粒度流水线把渲染、推理、训练的资源利用推满，混合流水线带来 1.61–1.88×（论文）训练加速、框架整体吞吐约 2.27×（论文），同一模型在 LIBERO 130 任务上做到 98.11% 成功率（论文，arXiv:2510.06710）。

## 核心

### 背景/问题：VLA+RL 生态是碎片化的

按官网预告，本讲分四段：先讲 RLinf-VLA 的设计思路与系统架构，再讲 VLA+RL 的算法技术设计（PPO / GRPO 等），然后是 OpenVLA 的微调实践，最后 AMA。论文给出的背景是：VLA 模型（如 OpenVLA）普遍行为克隆预训练后，会出现部署环境分布漂移、复杂指令理解不足、长程规划弱等短板；强化学习被认为是释放 VLA 潜力的关键后训练手段，但当时几乎没有成熟的 RL 框架专门面向 VLA——研究者每做一个算法就要先自建一套机器人 rollout + 训练的工程，显卡开销高、代码不可复用，新算法开发门槛居高不下。系统层面的深层原因是结构性的：RL for VLA 的流水线天然有渲染（仿真器渲染观测）、推理（policy 生成动作）、训练（参数更新）三个阶段，三者对 GPU 的占用模式完全不同——渲染吃显存与计算且峰值抖动，推理是低延迟小批量，训练是大批量高吞吐——塞进同一套静态分配里必然互相等待、利用率塌陷。

### 方法/设计：统一接口 + 三种 GPU 分配模式

RLinf-VLA 的主系统（建在 RLinf「渲训推一体化」框架之上，母框架论文 arXiv:2509.15965）做三件事：

第一层是统一接口：把仿真器（ManiSkill、LIBERO、RoboTwin，并支持 partial environment reset 这类具身特有能力）、VLA 架构（OpenVLA、OpenVLA-OFT，并预留对更多架构的接入）与 RL 算法（PPO、GRPO）都抽象成可插拔组件，让跨「架构 × 算法 × 仿真器」的对照实验成为配置问题而不是重写代码问题。

第二层是 GPU 分配的三档模式——共存（colocated）、分离（disaggregated），以及论文提出的新颖混合（hybrid）模式：三种模式对应渲染、推理、训练三阶段资源占比的不同切法，切换只需改配置。针对 GPU 并行化的仿真器（渲染也在 GPU 上跑），再加一层混合细粒度流水线（hybrid fine-grained pipelining）让三阶段真正重叠起来——这是 1.61×–1.88× 加速的主要来源（论文）。

第三层是算法侧的一组工程化设计：轻量 critic、loss 归一化、action masking、rollout 过滤等（论文），服务于稳定性与样本效率。

### 实验/实战：130 个 LIBERO 任务第一次被单一模型刷到 98%

系统对照：与基线框架相比，RLinf-VLA 吞吐提升 2.27×（论文 Fig.1 口径）；在 GPU 并行仿真器内部，混合细粒度流水线再带来 1.61×–1.88×（论文）。

效果对照：同一套 RL 训练出的单个统一模型，在 130 个 LIBERO 任务上取得 98.11% 成功率（论文称系首次）、25 个 ManiSkill 任务上 97.66%（论文）；RoboTwin 6 个任务平均成功率 84.63%，相比 SFT 基线平均提升 63.75%（论文）。归结起来，论文里报告的成功率提升幅度在约 20–85% 区间（论文）。

从全套对照实验里蒸馏出的最佳实践，是论文自认最有长期价值的部分：对 PPO，动作级（action-level）价值估计优于 chunk 级价值估计；partial reset（部分环境重置）显著改善样本效率——这类结论只有在「架构 × 算法 × 仿真器」可控变量扫过一遍后才敢说。

### 结论/观点

论文明确表达的判断有两条（区分事实与观点）：其一（观点），「碎片化」是 VLA+RL 进展慢的系统性原因，因此统一、高效、可扩展的开源框架本身是关键科学基础设施，而不是每个算法的附带工程；其二（由实验支撑的结论），在动作空间粒度、价值估计粒度、环境重置策略这几个设计维度上，PPO 的调校细节会显著左右最终成功率——上线前先在小基准上把这些开关扫一遍，比换算法更划算。讲者臧宏之为 RLinf 框架 VLA 部分核心开发与论文一作，本讲定位在「上手」：设计思路与系统架构 + PPO/GRPO 算法设计 + OpenVLA 微调实践，是把这篇框架论文翻译成一次可操作演示的尝试。

## 关键数字

| 指标 | 数值 | 来源（论文，arXiv:2510.06710） |
|---|---|---|
| 系统吞吐提升（vs 基线框架） | 2.27× | 论文 Fig.1 / 概述 |
| GPU 并行仿真器内混合流水线额外加速 | 1.61×–1.88× | 论文 §实验（ManiSkill 口径） |
| LIBERO 130 任务成功率（单一模型） | 98.11%（论文称首次） | 论文摘要 |
| ManiSkill 25 任务成功率（单一模型） | 97.66% | 论文摘要 |
| RoboTwin 6 任务平均成功率 | 84.63%（vs SFT 平均 +63.75%） | 论文摘要 |
| RL 相对基线成功率提升幅度 | 约 20–85% | 论文结论汇总 |
| 支持的仿真器 / VLA 架构 / 算法 | ManiSkill、LIBERO、RoboTwin / OpenVLA、OpenVLA-OFT 等 / PPO、GRPO | 论文系统设计 |

## 可迁移

- 对 coding data / RL infra 的直接可试点：LLM 的 RL 训练流水线里 rollout 生成（推理引擎）与训练（显存大户）的资源切分完全 analogous——「共存 / 分离 / 混合三档只改配置」的抽象值得借鉴：把 vLLM/SGLang rollout 与 FSDP/Megatron 训练的 GPU 分配策略做成可切换配置，而不是写死一种部署拓扑，再用小基准逐档试错。
- Infra 视角：框架把「渲染」也算进流水线调度的提醒对 agent RL 同样成立——sandbox/tool 环境的执行延迟分布（渲染的具身对应物）在 agent 训练里就是第三条曲线，调度策略如果不建模它，端到端利用率同样塌陷；partial reset 改善样本效率的结论，可以试着迁移到 agent 任务上（失败轨迹中途重置、复用好前缀）做一轮消融。

## 疑问 / 下一步

- 论文只给了框架整体吞吐 2.27×（论文）与混合流水线 1.61×–1.88×（论文）两个系统数字，没有分阶段的利用率拆解（渲染 / 推理 / 训练各阶段 GPU 空转率）；想看 hybrid 模式在不同 rollout 长度分布下是否稳定占优，待查论文实验附录或实测复现。

## 原文金句（论文，非讲授口播）

> "To address these challenges, we present RLinf-VLA, a unified and efficient framework for scalable RL training of VLA models." —— 论文标题与摘要：统一与高效被并列成一等目标

> "In particular, within GPU-parallelized simulators, the hybrid fine-grained pipelining mechanism further accelerates RL training, yielding a 1.61x-1.88x speedup." —— 论文贡献条目：系统加速不是跑分饰品，是框架的第一性设计

---
*未确证项：①本场确切讲授顺序与每节时长无字幕可核，讲纲以官网提纲为准；②论文数字版本（v1–v3）可能更新，本纪要未逐项标注版本差异；③AMA 与口头补充（如实测硬件配置）无法还原；④B94《为具身智能构建动态本体自进化记忆框架》真实 BV 待后续枚举复核。*
