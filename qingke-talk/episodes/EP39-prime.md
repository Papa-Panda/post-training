# EP039 — PRIME: 结合隐式过程奖励的强化学习

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP39-prime.md

> 「我们这个方法其实是非常的通用的，你只要有一个模型，不论它是 base model 还是 SFT model，你就可以直接对它用 PRIME 进行计算。」——讲者在总结中对 PRIME 适用范围的概括（字幕约 38:11–38:25）

## 元信息

- 期号：39
- 标题：PRIME: 结合隐式过程奖励的强化学习
- BV：BV1UaPNecEcW
- 时长：56:21（讲授约 39 分钟 + Q&A 约 17 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：崔干渠（上海人工智能实验室青年科学家；字幕自述，「甘渠」为口播识别，以公开资料为准）
- 相关论文：PRIME: Process Reinforcement through Implicit Rewards（上海人工智能实验室），https://arxiv.org/abs/2502.01456 （2025-02）；前置工作 Implicit PRM / implicit process reward 相关预印本（2024-12，讲者口径）
- 相关代码：数据、代码、模型全部开源（讲者口径；具体仓库地址未在 talk 中点名）
- B站链接：https://www.bilibili.com/video/BV1UaPNecEcW/
- 官网期号：39（预告链接未确认，仅标期号）
- 字幕原文存档：本地 `transcripts/EP39.txt`（1418 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（已存档）清洗提炼（经人工清洗；个别专名可能有识别误差，PRM 在字幕中作「PM/P2M」、verifier 作「Wi-Fi」、implicit PRM 作「in place CPRM」，均以论文为准）。凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

这期讲 PRIME 怎么把过程奖励接进在线 RL：核心是先用 DPO 式的隐式奖励参数化（只用 outcome label 训练），让一个结果奖励模型免费吐出逐 token 的过程奖励；PRIME 在此之上用 REINFORCE leave-one-out 的框架，每步同时在线更新 PRM 与 policy，并把 PRM 输出当作 reward（不是 value）加进每个 token 的优势估计，结果是同等数据下收敛更快、数据效率显著高于纯 outcome reward，且能和 REINFORCE/GRPO/PPO 任意组合。

## 核心

### 背景/问题：过程奖励在 RL 里为什么一直没用起来

讲者先引用 DeepSeek-R1 论文中专门的 "unsuccessful attempts" 一节，指出 PRM 在 RL 中面临两个关键挑战：

1. **过程奖励本身难定义**。推理过程的"步骤"并不自然存在——o1/R1 的思考过程无法干净地切成 step，强制模型生成 step 1/step 2 标签或用换行符切分都不自然；且用"这一步绝对的正确与否"当奖励会带来 reward hacking：模型重复输出"1+1=2"这类正确但无用的废话，PRM 给高分、但对解题毫无帮助。
2. **PRM 很难在线更新**。固定不更新的 PRM 必然被 policy 过拟合、发生 reward hacking（OpenAI 2022 年的结论：reward model 给出的分数继续涨，真实 reward 却放缓甚至下降）；而在线更新的成本极高——步骤级标签获取昂贵（OpenAI 雇人标了约 80 万条），已有把 PRM 接入 PPO 的工作估算约 50 倍计算开销；DeepSeek 用 MCTS 自动估步骤正确性的方案同样有噪声、开销大。所以 DeepSeek 最终没有把 PRM 用进 R1。

### 方法/设计：隐式 PRM + PRIME 的在线闭环

- **隐式 PRM（前置工作）**：换一个奖励参数化——把完整回复的奖励定义为两个语言模型的 log-likelihood 之比（与 DPO 的奖励定义一致）。这个参数化带来一个关键好处：Q 值有闭式解。由 Q 值可得在 $t$ 与 $t-1$ 两个时刻 Q 的差值，这正是该步的过程奖励。于是只需要在 outcome label 上训练一个 ORM（cross-entropy 或 DPO loss 均可），推理时按公式即得每个 token 的过程奖励——不需要任何步骤级数据、不需要额外训练，计算开销也远低于 MCTS 估计：讲者口径，以约 1/40 的开销得到与 MCTS 相当的结果，以约 1/10 的开销得到更好的 PRM（在 BoN 场景验证）。
- 这样的过程奖励奖励的是"在解题道路上更进一步"的 progress，而不是每步的绝对正确与否，且天然是逐 token 粒度、可以任意组合成想要的粒度——同时化解了定义难与切分难两个问题。
- **PRIME 算法**：每步迭代收集 rollout → outcome verifier 给分 → 用这个分数更新隐式 PRM，同时 PRM 给每个 token 打过程分 → 两个分数一起更新 policy。优势估计用 REINFORCE leave-one-out（与 GRPO 思想相近：组内 reward 减去其余样本均值作 baseline）；与纯 outcome reward 的区别只在于：outcome-only 时所有 token 的优势完全一致，PRIME 则在每个 token 的优势上多加了 PRM 给出的过程项。
- PRM 的更新信号就是 outcome verifier 的分数，与 policy 更新并行、每步都发生——讲者强调这一步"必不可少"：不更新，算法完全拿不到收益。

### 实验/实战：数字按字幕口径

- **主实验**（数学 + 代码，SFT 后 RL）：加入过程奖励后收敛速度大加快，达到相似性能只用约 40% 的数据；同等步数下最终模型比纯 outcome reward 训练高约 7%（字幕口径）；都训 240 步时 PRIME 比纯 outcome 高约 4 个点；继续训到约 600 步性能仍在提升，数学性能超过 GPT-4o、Llama-3.1 70B、Qwen2.5-Max 7B（字幕口径）。
- **PRM 在线更新消融**：用约几十万条数据离线训练好的 PRM，若不在线更新，分类准确率随 policy 漂移持续下跌、性能不理想；直接用 SFT 模型初始化 PRM、只在当前 policy 的 rollout 上训练的曲线，分类准确率从约 50%（随机）升到约 75%，效果最好——预先用更多数据训 PRM 反而不如它（绿线涨幅更小）。讲者的解释是只在自身 rollout 上训练的 PRM 对当前分布最熟悉，且最省（不需要收集额外数据）。
- **与任意 RL 算法组合**：把过程奖励项分别加到 REINFORCE、GRPO、PPO 上，三者都有稳定提升。
- **当 reward 用还是当 value 用**：一个易被忽略的关键消融——把隐式 PRM 输出当 value（作 baseline）用时，包括加 linear head 的 value model 在内，几种变种都没有收益（这与 DeepSeek、Cohere 论文中"value model 没用"的观察一致）；只有当 reward 用才 work。Q&A 中讲者进一步解释差别在 return 的定义：当 reward 时，return 里包含 PRM 对整条回复的过程分总和；当 value 时则不包含——数学上只差了 PRM 总分一项，但对训练效果影响重大，讲者推测与 reward shaping 有关。
- **Zero 实验**（直接从 base model 做 PRIME）：从 Qwen2.5-Max 7B base 出发涨得非常快，但约 50 步就难再涨、与 test 波动吻合；SFT 初始化的模型后期可能反超。32B base 上收益更大：只需 16 步 RL 的 test accuracy 就超过其 instruct 版本，约 100 步时还在涨、涨了 10 个点以上，与 DeepSeek 的大模型收益更高结论一致。
- **数据效率对比**：最终 7B 模型数学平均约 +16.7% 绝对提升（字幕口径），AIME 2024 从 3.3% 提升到 26.7%（字幕口径）；对比 Qwen-Max 的 instruct 模型，PRIME 只用约 1/10 的 SFT 数据（约 20 万；且可能都不必要）、没有用任何额外 RM（对方用了 72B 的 RM），每条 query 的 rollout 次数也远少。代码上力扣、LiveCodeBench 也各提了约 10 个点（Q&A 字幕口径）。

### 结论/观点（讲者明确判断）

- PRIME 把 outcome label 里隐含的信用分配免费地摊到每个 token 上，是通用的插件式方法：只要有一个 outcome reward（数学校验、代码 test case 等），就能转成稠密的过程奖励；base model 或 SFT model 出发都可以，与 REINFORCE/GRPO/PPO 任意组合。数据、代码、模型全部开源。
- 关于上限（Q&A）：讲者的判断是理论上 dense reward 不改变上限——唯一的外部监督仍是 outcome label、PRM 没有引入额外监督，所以 optimal policy 不变；变化的是"能不能达到"：更快、更稳地逼近上限，在数据量受限（更难任务、更少数据）时尤其重要。
- 讲者坦承的应用边界：目前验证集中在 coding 与数学等可验证任务上；文科理解类问题难以有标准答案、选择题又容易蒙，噪声大，这类任务可能最终还是要依赖强的 reward model 做 RL。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 过程奖励估计相对 MCTS 的开销 | MCTS 估计 | 约 1/40 开销可相当；约 1/10 开销可更好 | 字幕 |
| 把 PRM 接入 PPO 的此前工作估算开销 | 普通 RL | 约 50 倍计算开销 | 字幕 |
| OpenAI 步骤级标注规模 | — | 约 80 万条人工标注 | 字幕 |
| 同等性能所需数据量 | 纯 outcome reward | 只用约 40% 的数据 | 字幕 |
| 同等步数下最终模型相对提升 | 纯 outcome reward | 约 7% | 字幕 |
| 240 步时相对提升 | 纯 outcome reward | 高约 4 个点 | 字幕 |
| 在线更新 PRM 的分类准确率变化 | 初始约 50%（随机） | 升到约 75% | 字幕 |
| Zero 设置（7B base）饱和点 | — | 约 50 步后难再涨 | 字幕 |
| 32B base 超过自身 instruct 版本所需 RL 步数 | — | 只需 16 步 | 字幕 |
| 7B 模型数学平均绝对提升 | SFT 模型 | 约 16.7% | 字幕 |
| AIME 2024 准确率 | 3.3% | 26.7% | 字幕 |
| SFT 数据量 | Qwen-Max 百万量级 + 72B RM | 约 20 万条、无额外 RM（约 1/10） | 字幕 |
| 一个 step 的构成（7B 实验） | — | 256 prompt × 4 response = 1024 rollout，global batch 256，即 4 次梯度更新，约 15 分钟/step（8 卡，旧框架） | 字幕（Q&A） |

## 可迁移

- 对 coding data / RL infra 工作的直接可试的点：如果已有 outcome verifier（单测、答案匹配），可以先零成本尝试"隐式奖励参数化 + 过程分摊"这类稠密化手段，而不必先投入步骤级标注或单独训练 PRM；且消融提醒：把 PRM 输出当 value 用大概率无效，要当 reward 项明确加进 return。
- Infra 视角：PRM 在线更新是流水线设计的一票否决项——PRM 的训练数据必须是当前 policy 的 rollout，固定 PRM 的离线打分服务架构会系统性翻车（准确率随漂移下跌）；同时 PRM 与 policy 同步更新意味着训练编排里要预留两条更新链路的资源与调度。

## 疑问 / 下一步

- 隐式 PRM 最终仍只受 outcome label 监督，"结果正确但过程错误"的高分路径（如碰巧蒙对）如何被压制？讲者的回答偏直觉（正确答卷共享的步骤模式会被自动总结出来），但 process reward 的噪声上限与对训练的影响量级未量化。
- PRIME 的过程分与任务难度的关系：32B 上 16 步超 instruct 很有冲击力，但这与"base 本底已强、RL 只需解锁"如何区分，talk 未给出 pass@k 式的分解证据。

## 原文金句（1-2句）

> 「我们这个方法其实是非常的通用的，你只要有一个模型，不论它是 base model 还是 SFT model，你就可以直接对它用 PRIME 进行计算。」——讲者总结（字幕约 38:11–38:25）

> 「如果我们不去对这个 PM（PRM）去进行更新呢，这个算法其实就完全不能取得任何的收益。」——讲者点明在线更新是 PRIME 的一票否决项（字幕约 26:01–26:08）

> 「理论上它的上限可能是不变的，但是实际中加了 dense reward，它在效率、以及在数据的这种高效性上（有帮助）。」——Q&A 中讲者对"dense reward 是否抬升上限"的判断（字幕约 53:34–54:21 附近，口径转写）
