# EP103 — Decoupled DMD & DMDR：在扩散模型步数蒸馏的实践及 Z-Image-Turbo 应用

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP103-decoupled-dmd-dmdr.html

> 「如果没有了 CA（CFG Augmentation），那么这个蒸馏是无法进行的；Distribution Matching 项我们认为其实是做了一个 CA 的正则。」——讲者对 Decoupled DMD 核心发现的判词

## 元信息

- 期号：EP103（官网预告期号；见下方编号说明）
- 标题：Decoupled DMD & DMDR: 在扩散模型步数蒸馏的实践及 Z-Image-Turbo 应用
- BV：BV1ZzrHB8ESf
- 时长：00:44:22（字幕末条 2659.78 秒；讲授约 33 分钟 + Q&A 约 11 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：江丹阳（字幕开场自报作「江登阳」、随即自述改正为「江丹阳」；西北工业大学本科，即将赴香港科技大学攻读博士，现于阿里巴巴通义实验室实习——字幕作「通信实验室」，应为识别误差；研究方向为图像生成与编辑。正式头衔以论文作者页为准）
- 相关论文：Decoupled DMD、DMDR（均与 Z-Image 同期放出；讲授未给出编号与版本，以论文原文为准）；背景工作 DMD / DMD2（讲者点名 DMD2 的开源 codebase 值得二次研究）
- 相关代码：Z-Image GitHub（讲授结尾给出链接）；DMDR 开源了基于 SiT 的 ImageNet demo 训练代码（字幕口径）
- B站链接：https://www.bilibili.com/video/BV1ZzrHB8ESf/
- 官网预告：https://qingkeai.online/blog/Decoupled-DMD%26DMDR （2026-01-13）
- 字幕原文存档：本地 `transcripts/EP103.txt`（962 条，带时间戳）

> 🔢 编号说明：本仓库台账中该 BV 的旧标题为 B103《Manimator：用 AI 自动制作高质量动画》，与视频实际内容（Decoupled DMD / DMDR / Z-Image）完全不符；官网 EP103 预告标题与视频内容一致，故按官网编号 EP103 落稿，台账旧标题视为过时残留。视频无主持人开场报期号环节（讲者直接自我介绍开场），编号依据为官网预告映射 + 内容互证。

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（已存档）清洗提炼。字幕识别误差较多（如「步数蒸馏」作「步数真流」、「少步模型」作「小波/小布模型」、「GAN」作「干」、「RLHF」作「IOOK」等），均按上下文还原；个别专名以论文与公开资料为准。凡讲授口径与论文版可能不同者，标注（字幕口径）。

## 一句话总结

讲者把 DMD 的 loss 沿 CFG 方向拆成两项并逐项消融，结论反直觉：真正让少步蒸馏能跑起来的是 CFG Augmentation 项，名义上的主角 Distribution Matching 项只起正则/稳定作用；据此给两项配不同的 renoise schedule（Decoupled DMD），再把 RL 作为 teacher 之外的额外监督与蒸馏同训、用 DMD 分布约束压住 reward hacking（DMDR），最终落地 Z-Image-Turbo。

## 核心

### 背景：DMD 是什么，为什么从它开始

步数蒸馏的目标是让扩散/Flow Matching 模型在极少步数下输出接近多步模型的分布，动机是工业部署成本：多步推理本身贵，带 CFG 时函数评估次数（NFE）还要乘二（字幕口径）。蒸馏路线分轨迹级与分布级两类，讲者团队选分布级的 DMD，理由是它已在大规模场景被验证（字幕口径）。

DMD 的建模方式：real 分布由固定多步模型（在真实数据上预训练）的 score 给出；fake 分布由另一个扩散模型在少步模型输出上继续做 pretrain loss 来追踪；两者之间用 reverse KL 逼近。讲者给出他的个人理解：DMD 本质也是一种对抗训练，但它把「建模真实分布」与「追踪生成分布」解耦到两个模型上，避免单个 discriminator 同时做两件事带来的不稳定——这也是他个人偏好 DMD 的原因。

DMD2 的延伸是引入 GAN 式的真实数据监督，让少步模型不被 teacher 的能力封顶；讲者推荐其开源 codebase 作为后续蒸馏研究的起点（字幕口径）。

### 第一篇 Decoupled DMD：CFG 在 DMD 里到底扮演什么角色

大规模实操 DMD 时团队遇到一个具体问题：real score 的前向通常要带很大的 CFG，而对 Flux-Dev 这类已经做过 CFG 蒸馏的模型，再做 DMD 型蒸馏时不带（或带 fake）CFG 效果很差。于是第一篇工作的出发点是一个机理问题：CFG 在 DMD 框架里到底起什么作用？

做法是把 real score 按 CFG 形式展开（条件与无条件两部分的组合），再把 DMD loss 代数拆成两项：

- **CFG Augmentation（CA）项**：等价于一个标准 CFG 前向；
- **Distribution Matching（DM）项**：理论分析中 DMD 的本来面目。

在 SDXL 上逐项消融（字幕口径）：

1. 只留 CA：蒸馏初期效果好，但随训练步数增加很快坍塌；
2. 只留 DM（不带 CFG）：得不到好的蒸馏效果；
3. 两项结合：视觉效果好。

进一步的数值分析给出坍塌机制：没有 DM 约束时，输出的均值与方差随训练急剧变大，最终过饱和或直接坍塌成坏图。替代正则的对照实验：只对均值/方差做简单数值约束，训练稳定性也能保住、能蒸出不错的效果；换成 DMD2 式 GAN 约束则训起来非常不稳定、很快坍塌。

由此得到全场核心结论：**没有 CA 蒸馏根本无法进行（讲者称之为蒸馏的「引擎」性质）；DM 项的实际角色是对 CA 做正则、让过程稳定**。注意这与命名直觉正好相反。

### Renoise schedule：CA 与 DM 该在什么噪声水平上做

Renoise 指把少步模型的输出 $x_0$ 重新加噪到不同 noise level，再在该水平上对齐多步模型学到的数据分布。基于「CA 是核心」的判断，讲者先在 CA 上探索 renoise 水平（ $\tau$ 越接近 0 表示加噪越重、样本越「脏」）：

- 一直 renoise 到很脏的水平：只能建模全局信息，丢失高频纹理与细节；
- 逐渐放宽（噪声变轻）：高频信息与细节逐渐加回来；
- 只在低噪声水平做：直接坍塌——模型没掌握全局信息时学细节没有好效果。

最终方案是给两项配不同 schedule：CA 的 renoise 门限 $\tau$ 大于当前训练步的时刻 $T$ （即只往「当前步之前」的噪声程度回加；理由是前面步数已经建模好的结构不要再被破坏，字幕以 $T = 0.25, 0.5, 0.75, 1$ 的分步采样为例说明），DM 则用全局 noise schedule 做整体约束。在 SDXL 上与不同方法对比，讲者称指标整体不错（未口播具体数值，字幕口径）。

一个开放问题：小规模场景（ImageNet 等）不用 CFG、只用 DMD 也能 work；随着多步模型 CFG 加大，质量与多样性之间出现明显 trade-off（字幕口径）。为什么大规模必须 CFG、小规模不必——团队只有 hypothesis 与分析，没有理论证明，讲者明列为 interesting future work（此点在讲授后段又重复强调一次）。

### 第二篇 DMDR：少步模型的 RL 怎么做约束

第二篇的问题是给少步模型引入 teacher 之外的额外监督。多步模型的视觉 RLHF 已有很多工作，但基本都在多步模型上做；GAN 作为额外监督的局限在于它不够「定向」——想单独提升 instruction following、OCR 等具体能力时，GAN 给不了这种监督（字幕口径）。自然的替代是 RL，但少步模型做 RL 有两个结构性麻烦：

1. 蒸馏之后模型已没有 pretrain loss 的概念，多步 RL 赖以约束的 pretrain loss / KL-to-reference 类约束（讲者举例 ImageReward 系的 pretrain loss 约束、Flow-GRPO 的 KL 约束）在少步模型上不直接成立；
2. 分两阶段（先蒸馏再 RL，或先 RL 再蒸馏）会有误差累积。

讲者对顺序问题的处理是留作讨论：蒸馏前做 RL 的实验不多，但有一个已知互动——多步 RL 的 rollout 若不带 CFG，RL 后 CFG 的作用会被削弱（不用 CFG 推理反而更好），而 Decoupled DMD 又证明 CFG 对蒸馏至关重要，所以「先 RL 后蒸馏」的顺序与 CFG 的关系尚无定论，欢迎探讨（字幕口径）。

DMDR 的主张是蒸馏与 RL 同训、二者互相增益：

- **RL unlock 蒸馏上限**：reward 信号是 teacher 分布之外的额外监督，让少步模型不只是模仿 teacher，而是生成真正想要的样本（与 DMD2 引 GAN 的动机同构，但监督更定向）；
- **DMD 提供 RL 的分布约束**：同训时模型始终被更强的多步模型分布约束，实验上 reward 曲线的方差显著小于无 DMD 约束的纯 RL，讲者据此认为它能避免 reward hacking；视觉对照也支持：无好约束时在少步模型上做 ReFL 类方法很快出现网格化、或按 reward model 偏好（如 HPSv2 式）过拟合的画面（字幕口径）。

框架适配性：讲者称该约束形式可适配任意 RL 算法，论文中主要加了一个 loss 分支、ReFL 效果较好，GRPO/DPO 类没有做很多（字幕口径，Q&A）。

初始蒸馏阶段的两个 trick（论文内容，字幕口径）：少步模型在最初几步推不出像样样本时，real score estimator 没见过这类分布、估计不准。两个干预手法：

1. **Dynamic renoise**：初期让 renoise 采样偏向很高的 noise level（更「脏」的地方更偏全局信息，与 teacher 分布在该破坏程度下更接近），再逐步退化回 Decoupled DMD 的 schedule；
2. **LoRA 共享**：借鉴 MagicDistillation 类工作（字幕作「magic desolation」），让 fake model 的 LoRA 以较小的 scale 同时加到 real score estimator 上，使其输出分布偏向少步模型；scale 随训练退化到零，最终还原 teacher 真正学到的分布。

结果口径（字幕）：方法在若干指标上达到 SOTA，且在未针对 DPG-Bench 与 GenEval 做优化的情况下，overall score 高于多步 teacher（未口播具体数值）。

### 多样性代价与 Z-Image-Turbo 落地

RL 做完后少步模型多样性明显偏低。把 CFG（Decoupled DMD）与 RL（DMDR）叠加看：CFG 强 + RL 强时，Inception Score 与感知质量很高、图像特别贴近目标 metric，但多样性很低——这是质量/多样性 trade-off 在 RL 阶段的再现（字幕口径）。

落地形态：两篇算法都用在 Z-Image 的蒸馏上（Z-Image-Turbo）。讲者展示了写实生成、中英文文字渲染与世界知识案例，并提到只用双语数据训练却零样本涌现多国语言文字能力（字幕口径）；对世界知识来源的猜测是文本编码器用了 LLM/VLM（与 Qwen-Image 类模型相同路线），知识本就在编码器里（字幕口径，Q&A）。开源计划：Z-Image 的「mini base」（只做 pretrain、未做 SFT/RL 的版本，多样性高）预计不久开源，适合研究者自行微调（字幕口径）。

### Q&A 中值得记下的判断

- **Flux-Dev 蒸馏效果不好**：已由 Decoupled DMD 回答——它是 CFG 蒸馏过的模型，没有真正的 CFG 可用；团队试过，产物会特别「油腻」，像把 bias 蒸了出来（字幕口径）。
- **Flow Matching 的 renoise 是否要配 time shift**：试过 uniform 与 lognormal 等设定，「用不用都差不多」（字幕口径）；DMD 对 Flow Matching 模型本身适用（预测量可转成 score 形式），DMDR 开源的 ImageNet demo 就是基于 SiT 的 Flow Matching 模型。
- **Decoupled DMD 需要两次前向**（分别估 CA 与 DM 的 score），但都不带梯度（更新方式类似 DreamFusion 的梯度估计），效率损耗不大（字幕口径）。
- **引入 RL 不会让蒸馏更快**，但少步模型 rollout 本身特别快，这是少步 RL 的实际好处（字幕口径）。
- **RL 与 DMD 两个梯度怎么平衡**：无显著 trick——先让两个 loss 到同一量级，再按想提升的维度调权重；但先要确认 reward model 本身有效，「视觉 RL 里 reward model 非常重要」（字幕口径）。
- **Fake score 能不能去掉**：实践上可以去掉、也能做一会儿好效果，但非常不稳定，很快崩（文生图上尤其明显）；它是对输出分布的追踪约束（字幕口径）。
- **蒸馏后的模型再微调**：社区反馈不太好训，模式已收敛/坍塌，传统 MSE loss 也用不了；讲者认为「蒸馏后微调」本身是个值得做算法创新的方向（字幕口径）。
- **与量化/稀疏化叠加**：团队没做，社区已有基于 Turbo 模型的量化工作；加速可从步数（时间维）、层/caching（层维）、量化剪枝（尺寸维）多维组合（字幕口径）。
- **同期工作**：Reward Forcing 用 RL 的 score 控制 DMD 前向权重、而非简单加权组合，讲者认为引入 RL 的思想相通（字幕口径）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 带 CFG 的多步推理成本 | 单次前向 | NFE 约乘二 | 字幕 |
| CA 单独使用 | 蒸馏初期有效 | 随步数增加很快坍塌 | 字幕（SDXL 实验） |
| DM 单独使用（无 CFG） | — | 得不到好的蒸馏效果 | 字幕（SDXL 实验） |
| 无 DM 约束时输出统计 | 训练进行中 | 均值/方差急剧变大 → 过饱和或坍塌 | 字幕 |
| 均值/方差数值约束替代 DM | — | 稳定性可保住、效果可用 | 字幕 |
| GAN 约束替代 DM | — | 初期有约束效果，随后不稳定坍塌 | 字幕 |
| 高噪声 renoise（CA） | — | 只建模全局信息，丢高频细节 | 字幕 |
| 低噪声 renoise（CA） | — | 直接坍塌 | 字幕 |
| DMDR 结果 vs 多步 teacher | 未针对 DPG-Bench/GenEval 优化 | overall score 高于 teacher（数值未口播） | 字幕 |
| Decoupled DMD 前向次数 | 常规 DMD | 两次无梯度前向，效率损耗不大 | 字幕（Q&A） |

注：讲授对 benchmark 只给结论性口径（「指标整体不错」「达到 SOTA」），未口播具体数值，表中不补论文数字。

## 可迁移

- 「引擎 vs 正则」的拆项消融方法可直接迁移到任何复合 loss 的排障：先逐项单独跑看谁是必要项，再看谁只影响稳定性；命名里的「主项」未必是真正的承重墙。放到 RL/post-training 的 KL 约束、entropy bonus 等复合目标上同样适用。
- 少步/加速模型的 RL 约束思路：当被训对象已没有 pretrain loss 可用（蒸馏后模型、量化后模型、缓存推理模型），用「仍在训练的原模型分布」做在线分布约束，比离线 reference 的两阶段方案少一层误差累积，且经验上能压 reward 方差——这与 LLM 侧蒸馏+RL 同训的配方可以对照。
- Infra 视角：Decoupled DMD 的两次前向都是无梯度的梯度估计式更新，提醒评估蒸馏方案时要把「带梯度前向」与「无梯度前向」分开计成本；少步 RL 的真实加速来自 rollout 便宜，而非算法本身更快。

## 疑问 / 下一步

- 为什么大规模场景必须 CFG、小规模（ImageNet）不必：讲者明说是未解的 open question，只有 hypothesis 没有理论分析——这是 Decoupled DMD 最值得追的一篇后续。
- CFG 与多样性的 trade-off 在蒸馏与 RL 两阶段重复出现，讲授没有给出可调的中间形态（如部分 CFG、动态 CFG 强度）实验，值得查论文是否有对应消融。
- DMDR 的 reward 曲线方差对比与 overall score 高于 teacher 的具体数值都在论文里，字幕未口播；下一步应核对论文中 reward model 的构成与「高于 teacher」的评测口径。
- 「先 RL 后蒸馏」与 CFG 削弱的互动只是留作讨论、无实验结论；若要在 Z-Image 类模型上复现顺序实验，这是第一个要验证的点。

## 原文金句（1-2句）

> 「如果没有了 CA 就是 cfg augmentation，那么这个蒸馏是无法进行的……distribution matching 项呢，我们认为其实是做了一个 CA 的正则。」——讲者对 Decoupled DMD 消融结论的判词（字幕 [12:36]–[12:50]，按干净口径转写）

> 「在视觉的 RL 里面，其实 reward model 是非常重要的，因为我没有一个很好的 rule 去衡量。」——Q&A 谈 RL 与 DMD 梯度平衡时讲者的前置提醒（字幕 [37:19]–[37:27]）

> 「蒸馏它本身的模式已经收敛了，就不太好去继续微调了。」——Q&A 谈蒸馏后微调的社区共识（字幕 [41:48]–[41:57]，按干净口径转写）
