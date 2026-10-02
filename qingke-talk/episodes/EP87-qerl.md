# EP087 — QeRL：量化技术增强强化学习 Reasoning 探索

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP87-qerl.html

> 「量化不仅可以在传统意义上减轻模型的内存大小、加速推理，我们发现它在强化学习当中还能够有助于大语言模型在 reasoning 训练当中的探索。」——讲者对全场核心发现的一句话概括

## 元信息

- 期号：87
- 标题：QeRL：量化技术增强强化学习 Reasoning 探索
- BV：BV1iAkDBrEVt
- 时长：00:57:46（讲授约 43 分钟 + Q&A 约 14 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：黄伟（Wei Huang，香港大学三年级博士生，导师 Xiaojuan Qi 齐晓娟；本工作于 NVIDIA 实习期间完成，字幕中自述姓名与学校，英文名与导师以论文作者页为准）
- 相关论文：Wei Huang, Yi Ge, Shuai Yang, Yicheng Xiao 等，*QeRL: Beyond Efficiency — Quantization-enhanced Reinforcement Learning for LLMs*，https://arxiv.org/abs/2510.11696
- 相关代码：https://github.com/nvlabs/qerl
- B站链接：https://www.bilibili.com/video/BV1iAkDBrEVt/
- 官网预告：青稞Talk 官网第 87 期预告文（链接未确认，仅记期号）
- 字幕原文存档：本地 `transcripts/EP87.txt`（1554 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（经清洗，专名可能有识别误差，如 QeRL 在字幕中多作「QEL/QERL」、Marlin 作「马令 kernel」、layer norm 作「LIANNUM」等，均以论文与公开资料为准）撰写。论文（arXiv:2510.11696）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

QeRL 把 NVFP4 量化主干 + LoRA 分支当作 RL 训练范式：量化权重同时压显存、加速 rollout（约占 GRPO 训练 70% 的时间），而量化噪声会天然抬高策略熵、防探索塌缩，使低比特模型在 RL 中的 reward 上升比 16-bit LoRA 更快、最终更高，基本追平全参微调；再加一个按幂指数衰减调度的自适应量化噪声（AQN，噪声经 layer norm 共享注入），把探索增益延续到训练后期。结果是 32B 模型可以在单张 H100 80GB 上做 RL 训练。

## 核心

### 背景/问题：RL 太贵，贵在显存与 rollout

讲者先定位 RL 的成本结构（字幕口径）：一是显存——GRPO 类算法每个样本要采一组候选（常用 8 或 16 条）再打分更新，激活与数据量随组成倍放大，而 reasoning 训练又偏好 7B–30B 以上的模型；二是速度——reasoning 输出很长，rollout 阶段在他们统计的 GRPO/DAPO 训练里约占总时间的 70%，前向求 logprob、反向更新、validation 占比都低。目标因此明确：做一个 low-cost RL，让经济型显卡也能训更大的模型。

量化是现成的杠杆（权重从 FP16 压到 INT8/FP4 可减重 2–8 倍甚至 16 倍），但已有两条路线各有硬伤，讲者逐一拆解：

- **FlashRL 路线**（INT8 rollout + BF16 update）：rollout 可加速约 1–2 倍，但把训推不一致放大——推理引擎与训练框架之间本来就有差异，再叠 INT8 与 BF16 的精度差，更新容易崩；且 BF16 权重持续更新后与 INT8 版本的差距会随步数累积。FlashRL 用 TIS 采样策略缓解，但结构性问题还在。
- **QLoRA 路线**（NF4 主干 + LoRA）：在 SFT 中被普遍观察到 loss 曲线始终差于 BF16 LoRA——低比特主干受损后记忆能力变弱，强监督记忆任务追不回来。

### 框架设计：量化主干 + LoRA，且训推同构

QeRL 的架构沿用 QLoRA 式范式但换了量化格式与用途（字幕口径）：

- 主干用 **NVFP4**（NVIDIA 的 4-bit 浮点格式）做 **weight-only 量化**，只更新 LoRA 分支（曲线示例中 rank 取 32）。选 NVFP4 而非 NF4/MXFP4 的理由（字幕口径）：NF4 靠查表反量化、没有推理 kernel，rollout 反而慢；MXFP4 与 NVFP4 收敛都较快，但 NVFP4 粒度更细、训练后期精度更高，final reward 与 benchmark 都更好。
- **训推同构是关键设计**：rollout 与 update 用的是同一个低比特主干 + LoRA 分支，不引入框架之外的额外训推不一致——这是它相对 FlashRL 路线（8-bit 采样、16-bit 训练）的结构性优势。update 阶段对精度敏感的问题则由高精度 LoRA 分支承接。
- 推理部署用 vLLM + Marlin kernel（NVFP4 权重 × BF16 激活），在 Hopper（H100）上即可跑，不必等 Blackwell。实现基于 HF TRL（transformers 的 TRL 库），讲者称后续计划适配 verl 与 TRL 官方仓库（Q&A 口径）。

### 核心发现一：量化噪声在 RL 里是探索增益，不是损伤

全场最反直觉的结论（字幕口径）：在 SFT 中量化让 loss 曲线变差，但在 RL 中 NVFP4 + LoRA 的 reward 上升速度与最终值都**优于** 16-bit LoRA，并与全参微调基本对齐。机制解释：量化等价于给权重注入噪声，decode 时 next-token 分布被摊平、置信度下降，本来概率不高的 token 更容易被采出——这在 SFT 里是对不上 ground truth 的损伤，在 RL 里恰恰是采样阶段需要的探索。实证签名是训练全程 policy entropy 保持相对更高，量化相当于"另辟蹊径地防了熵塌缩"。讲者还提到两个伴生现象（字幕口径）：QeRL 的平均回复长度更长（更容易吐出 however 这类反思 token、"aha moment"更多），且可承受的学习率比 LoRA 再大一个数量级（LoRA 在同等学习率下直接训崩），与 Thinking Machines Lab 关于 LoRA 可开大学习率的博客观察同向。

### 核心发现二：AQN——把量化噪声变成可调度噪声

量化噪声有个先天缺陷：它是静态的，只在训练起点（T0）起作用，训到几百步后就被逐渐抵消。讲者借 2017 年 parameter space noise 的经典思路（早期探索希望噪声大、后期收敛希望噪声小），提出 **AQN（Adaptive Quantization Noise）**：前期直接用量化噪声充当调度起点，之后按幂指数衰减手动补注噪声，延续后期的探索空间。

噪声往哪注是个具体的工程难题（字幕口径）：高精度噪声注不进 4-bit 权重空间（同比特噪声会直接摧毁权重分布）；注到 LoRA 分支也不行（分支低秩高敏感、会伤训练效率）。解法是 **noise sharing**：把噪声加到 layer norm 的 scale 上——QKV 共用一组噪声、FFN 的 gate/up/down 共用一组；讲者称他们证明了"加在 layer norm 上等价于对后续权重施加乘性噪声"，且加性与乘性噪声在 RL 探索中都被验证有效，乘性噪声只需注意把方差控制在低区间。这样注入不碰低比特 kernel、效率不受影响。消融（7B 示例，字幕口径）：AQN 主要在量化噪声匮乏的训练后期继续抬升 reward 曲线，模型越大增益越明显。

### 实验/实战：数字

设置（字幕口径）：模型为 Qwen2.5 系列 3B/7B/14B/32B 的 **base instruct 模型**（讲者刻意强调：没有经过 reasoning 训练或蒸馏 SFT，以此证明方法对"干净"预训练模型同样成立）；基准覆盖 GSM8K、MATH-500、AIME 2024/2025、AMC23 与 Big-MATH 高难度子集（level 3–5 / 4–5）；RL 算法为 GRPO/DAPO 类。

- **简单任务（GSM8K，3B/7B）**：QeRL 与全参微调基本完全持平，超过 16-bit LoRA（字幕口径）。论文口径的 7B 具体数字：GSM8K 90.8%、MATH-500 77.4%。
- **难任务收敛速度**：Big-MATH level 4–5 上 QeRL 约 250 步收敛到与全参相近的 reward 区间，16-bit LoRA 要 800 多步（字幕口径；论文附录有到 800 步的完整曲线）。难题上 QeRL 仍略低于全参，但 reward 速度与终值都远高于 16-bit LoRA。收敛步数口径：全参约 220–300 步进入平台期。
- **效率**：rollout 相比 16-bit 模型快约 1.8–2 倍（字幕口径；论文口径为 1.5 倍以上）；32B 模型在单张 H100 80GB 上约 10 秒/步（较大 batch）即可训练，16-bit LoRA 在单卡 80GB 上放不下 32B；同为 32B，QeRL 端到端约为 QLoRA 的 2–3 倍（字幕口径，NF4 查表反量化拖慢 rollout）。7B 训练显存约 15GB（Q&A 口径）。

### 结论/观点（区分事实与判断）

- 事实：量化 + LoRA 在 RL 中 reward 曲线反超 16-bit LoRA、追平全参，且显存与速度同时受益——效率与性能之间在这个设定下不需要做取舍。
- 判断（讲者立场）：量化噪声应被当作一种**探索机制**来设计和调度，而非单纯的精度损失；bit-width 存在最优区间——压得越低越好是错的，INT4 的噪声大会在几十到一百步内训崩（Q&A 口径），8-bit 的探索增益又弱于 4-bit（扩展实验口径）。QeRL 的范式可迁移到任何 RL 框架，本质是给 RL 加了一个低成本的熵保持器。

## Q&A 要点

- **MoE 未试**：全部实验在 dense 模型上；讲者认为范式与 MoE/dense 本质无关，稳定性差异主要在优化器层面（如 GSPO 对 MoE 更友好），换算法无障碍。
- **KV cache 量化未试**：NVFP4 已把显存压得足够低，长上下文训练未遇 OOM；量化 KV cache 可能进一步降显存提速，未验证。
- **W4A4 / 全参低比特更新**：讲者明确不看好近期可行性——update 阶段对精度敏感，低比特全参更新损失过大；团队内部有 FP8（W8A8）训练的工作在推进。LoRA 分支保持 16-bit 正是为了守住 update 精度。
- **与 FlashRL 的训推不一致对比**：QeRL 训推同构，不额外放大不一致；残余的不一致与普通推理/训练 kernel 差异同量级。
- **更长 step 的稳定性**：难任务已对比到 1000+ 步，结论不变；简单任务 100–200 步已收敛故未拉长。小模型（3B）输出长度不一定比 LoRA 长，长输出现象与模型规模有关（Q&A 口径）。
- **实验规模**：作者团队协作完成，二作（清华本科生）负责大量效率测试；实验用了较多 GPU 资源（NVIDIA 与韩松老师支持）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| rollout 占 GRPO/DAPO 训练时间 | — | 约 70% | 字幕 |
| rollout 加速（NVFP4 + Marlin vs 16-bit） | 16-bit 模型 | 约 1.8–2 倍 | 字幕 |
| rollout 加速（论文口径） | — | 1.5 倍以上 | 论文 |
| 32B 单卡训练 | 16-bit LoRA 单张 H100 80GB 放不下 | 单卡约 10 秒/步可训 | 字幕 |
| 端到端 vs QLoRA（32B） | QLoRA | 约 2–3 倍 | 字幕 |
| 7B 训练显存 | — | 约 15GB | 字幕（Q&A） |
| GSM8K（7B） | — | 90.8% | 论文 |
| MATH-500（7B） | — | 77.4% | 论文 |
| 难任务收敛步数（Big-MATH level 4–5） | 16-bit LoRA 800+ 步 | QeRL 约 250 步（全参约 220–300 步进平台） | 字幕 |
| LoRA rank（曲线示例） | — | 32 | 字幕 |
| 可承受学习率 | LoRA（同 LR 训崩） | 再高一个数量级 | 字幕 |
| INT4 噪声过大的后果 | — | 几十到一百步内训崩 | 字幕（Q&A） |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **单卡大模型 RL 的可行配方**：NVFP4（weight-only）+ 16-bit LoRA + vLLM/Marlin rollout，在 H100 上即可对 32B 做 GRPO——对显存受限的 agent RL 实验是直接可抄的配置；讲者的实现基于 TRL，计划适配 verl。
  2. **把量化当熵管理工具**：RL 训练中熵塌缩是常见问题，量化噪声提供了一个零额外组件的熵保持来源；若自研框架里已有熵 bonus/clip 类调参，可对照评估用低比特 rollout 替代一部分熵调控（注意讲者的边界：bit-width 过低会训崩，4-bit 浮点格式优于 INT4）。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：rollout 占 RL 训练约 70% 时间，任何 rollout 侧优化（量化、kernel、训推同构）的杠杆都比 update 侧大；"训推用同一份低比特权重 + 高精度低秩分支更新"是同时压显存、压不一致、保更新精度的组合拳。

## 疑问 / 下一步

- AQN 的噪声调度（幂指数衰减速率、初始方差）与任务难度、序列长度的关系，讲者只给了定性结论；agent 类长轨迹任务上噪声该怎么排，未展开。
- MoE 模型与 KV cache 量化均为未验证项；coding agent 场景（工具调用、多轮）下量化噪声对探索的增益是否与数学任务一致，值得对照 EP075 FlashRL、EP129 ARPO/AEPO 的高熵 token 讨论一起看。
- 讲者称实现计划适配 verl 官方仓库——若落地，可直接在本仓库的 RL infra 实验中试跑，对照其 32B 单卡口径复现。

## 原文金句（1-2句）

> 「量化不仅可以在我们传统意义上减轻模型的内存大小、减轻训练的内存负担，同时还可以加速；那额外呢我们发现量化带来了另外一个非常有趣的争议，就是说在强化学习当中它能够有助于我们在 reasoning 的训练当中大语言模型的探索。」——讲者开场对 QeRL 核心发现的概括（字幕 01:28 附近）

> 「我们甚至不需要牺牲性能……以更低的效率开销实现了更高的 performance，而不需要在效率和性能之间做 tradeoff。」——讲者总结 GSM8K 结果时的判断（字幕 33:43 附近）
