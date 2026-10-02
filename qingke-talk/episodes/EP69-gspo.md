# EP69 — GSPO：大规模强化学习训练算法，迈向持续拓展的语言模型强化学习

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP69-gspo.html

> "Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization." —— GSPO 论文 Abstract，本期方法的核心主张

## 元信息

- 期号：69
- 标题：GSPO：大规模强化学习训练算法，迈向持续拓展的语言模型强化学习
- BV：BV1UC4az4EiU（B站有视频；字幕接口多次返回串台字幕、无法验证，未能取得可用字幕）
- 直播时间：2025-08-07 20:00–21:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：郑楚杰（通义千问研究员，Qwen3、QwQ 系列开源模型核心贡献者；2025 年博士毕业于清华大学，师从黄民烈教授；官网预告嘉宾介绍）
- 相关论文：Chujie Zheng, Shixuan Liu, Mingze Li 等（Qwen Team, Alibaba Inc.），*Group Sequence Policy Optimization*，https://arxiv.org/abs/2507.18071（v2, 2025-07-28）
- 相关代码：论文未附官方代码链接；GSPO 已用于最新 Qwen3 模型的 RL 训练（论文 Abstract）
- 官网预告：https://qingkeai.online/blog/PO5IkeL7

> ⚠️ 提炼方式说明：本期 B站有视频，但字幕接口多次返回与视频无关的串台字幕（无法通过标题/时长校验），未能取得可用字幕。本纪要根据该期对应的公开材料还原——官网预告（含讲者与提纲）+ GSPO 论文原文（arXiv:2507.18071）。讲授提纲以官网预告为准，公式与实验结论以论文为准。**非逐字稿**：talk 现场发挥、演示与问答未覆盖，实际内容可能与论文有出入。若日后取得字幕，应以字幕修订本纪要。

## 一句话总结

GSPO 指出 GRPO 在大规模训练中崩溃的根源是重要性采样的误用——token 级比值在每个位置只基于单个样本，根本起不到分布纠偏作用，只会注入随长度累积的高方差噪声；解法是把重要性比、裁剪、奖励和优化全部抬到序列级（长度归一化的序列似然比），让优化单位与奖励单位对齐，从而根治 MoE 训练的专家激活波动问题、不再需要 Routing Replay，且同算力下训练效率高于 GRPO，已支撑最新 Qwen3 模型的 RL 训练。

## 核心

### 背景/问题：GRPO 为什么难以拓展——重要性采样被用错了

按官网预告提纲，讲授分三部分：GRPO 难以拓展 RL 训练的观察与分析、GSPO 算法原理、GSPO 在大规模实践中的优势（训练效率 / MoE 稳定性 / infra 友好性）。论文 §3 给出了第一部分的核心论证：

重要性采样的原理（论文 §3，Eq.4）是用行为分布的样本重加权来估计目标分布下的期望，其成立前提是对行为分布做**多样本平均**，比值才能有效纠正分布偏移。而 GRPO（及 PPO）在每个 token 位置 $t$ 上只用当前实际采样到的那一个 token 计算比值 $w_{i,t}(\theta)$ ，单个样本的比值不具备任何纠偏能力，反而把高方差噪声注入梯度；噪声随响应长度累积、又被裁剪机制放大，最终导致**不可逆的模型崩溃**——论文明确写道，崩溃后即使回到旧 checkpoint、精心调裁剪范围、延长生成长度或更换训练数据，恢复训练也无济于事。

由此得出设计原则：**优化目标的单位应当与奖励的单位一致**（论文 §3）。奖励是发给整条序列的，off-policy 纠正就不该在 token 级做。

### 方法/设计：序列级重要性比 + 序列级裁剪

GSPO 保留 GRPO 的分组相对优势估计（同一 query 的 $G$ 条响应用组内奖励均值/标准差归一化），只把重要性比换成长度归一化的序列似然比（论文 §4.1，Eq.5–7）：

$$\mathcal{J}_{\mathrm{GSPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\big(s_i(\theta)\hat{A}_i,\,\mathrm{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_i\big)\right]$$

其中序列级重要性比为：

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$$

长度归一化（取 $1/|y_i|$ 次幂）是为了压低方差、把不同长度响应的比值约束到统一数值范围；否则少数 token 的似然变化就能让整条序列的比值剧烈波动，不同长度还得用不同裁剪范围。也正因定义不同，GSPO 的裁剪范围与 GRPO 差了三个数量级量级——GSPO 取 3e-4 / 4e-4 ，GRPO 对照组经调优取 0.2 / 0.27（论文 §5.1）。

梯度层面的区别（论文 §4.2）：GRPO 给一条响应里每个 token 按各自的比值加权，这些权重在正优势时落在 $(0,1+\varepsilon]$ 、负优势时落在 $[1-\varepsilon,+\infty)$ ，量级杂乱且影响随训练累积；GSPO 则让同一响应的所有 token 共享同一个序列级权重，消除了这个不稳定源。

另外论文 §4.3 给了 **GSPO-token** 变体：用 stop-gradient 技巧构造数值上恒等于 $s_i(\theta)$ 的逐 token 比值，使目标与裁剪条件与 GSPO 完全等价，但允许逐 token 调整优势 $\hat{A}_{i,t}$ ——为多轮 RL 等需要细粒度信用分配的场景保留口子。

### 实验/实战（论文 §5）

**主实验（§5.1）**：从 Qwen3-30B-A3B-Base 冷启动微调，评测 AIME'24（32 采样平均 Pass@1）、LiveCodeBench（202410–202502，8 采样平均 Pass@1）、CodeForces（Elo Rating），每个 rollout batch 切成 4 个 mini-batch 更新。论文以训练曲线报告（Fig.1）：GSPO 全程稳定，同算力、同 query 消耗下训练精度与基准表现均优于 GRPO，可通过持续加算力、刷新 query 集、延长生成长度获得连续提升。论文未给终点数值表，本纪要不抄图估数。

两个值得单独记的发现：

- **裁剪比例的悖论（§5.2）**：GSPO 与 GRPO 被裁掉的 token 比例差**两个数量级**（Fig.2）——GSPO 裁掉得多得多、用得少得多，训练效率却更高。这说明 GRPO 的 token 级梯度估计本身就噪声大、样本利用效率低。
- **MoE 稳定性（§5.3）**：MoE 每做一次梯度更新，同一条响应在新旧策略下激活的专家就有约 10% 不同（48 层 Qwen3-30B-A3B-Base 的实测），token 级比值因此剧烈波动而失效。团队此前的解法是 Routing Replay——缓存旧策略的路由并在新策略计算时重放，代价是额外显存/通信开销且限制 MoE 实际容量。GSPO 只依赖序列似然、对单个 token 似然不敏感（语言建模能力保证序列似然不会剧烈波动），从根本上绕开了这个问题，**不再需要 Routing Replay**。

**Infra 友好性（§5.4）**：训练引擎（如 Megatron）与推理引擎（如 SGLang、vLLM）之间存在数值精度差异，常规做法要回训练引擎重算旧策略似然。序列级似然对这种精度差异的容忍度远高于 token 级，GSPO 因此可以直接使用推理引擎返回的似然做优化，省去重算——在 partial rollout、多轮 RL 和训推分离框架里收益尤其明显。论文称这些优点已贡献于最新 Qwen3 模型的显著提升（Abstract）。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| 裁剪范围（左/右） | GRPO 经调优取 0.2 / 0.27 | GSPO 取 3e-4 / 4e-4（定义不同、量级不可直比） | 论文 §5.1 |
| 专家激活变化 | — | 每次梯度更新后同一响应约 10% 专家激活不同（48 层 Qwen3-30B-A3B-Base） | 论文 §5.3 |
| 被裁剪 token 比例 | GRPO | GSPO 高出两个数量级，效率反而更高 | 论文 §5.2、Fig.2 |
| Routing Replay 依赖 | GRPO 训 MoE 正常收敛所必需 | GSPO 不需要 | 论文 §5.1、§5.3 |
| 实验底座与评测 | Qwen3-30B-A3B-Base 冷启动 | AIME'24（32 采样 Pass@1）/ LiveCodeBench 202410–202502（8 采样）/ CodeForces Elo，曲线报告无终点数值表 | 论文 §5.1、Fig.1 |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **训 MoE 或长响应不稳定时，先换重要性比的粒度，再加稳定化补丁**：把 GRPO 的 token 级比值换成序列级（长度归一化）是算法层的一处改动，可省掉 Routing Replay 这类需要缓存/重放路由的工程补丁，显存与通信都更省。
  2. **把 clip fraction 当健康探针**：GSPO 裁掉的 token 比 GRPO 多两个数量级仍更高效——监控裁剪比例与优势分布，能比只看 reward 曲线更早发现重要性比粒度不匹配的问题。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. 序列级似然容忍训推引擎精度差异，可直接采用推理引擎返回的 logprob，砍掉"回训练引擎重算旧策略似然"这一遍开销；做 partial rollout / 多轮 agentic RL 时，这个简化直接决定流水线能不能拼起来。
  2. 算法选型时把"少一个稳定化 trick"计入 infra 成本：Routing Replay 之类的补丁都有显存、通信和容量上限代价，能在算法层消掉的不要留到系统层。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：GSPO-token 只给了数值等价性论证，论文未报告它在多轮 agentic RL 中的实测效果；当奖励本身就是细粒度的（如过程奖励、逐 token 验证信号）时，序列级比值会不会反过来成为瓶颈，值得找后续工作核对。
- 现场内容不可还原：郑楚杰的讲授侧重、与 Qwen3 实际训练经验的现场问答无法从论文推知；字幕若后续可取得，应以字幕为准修订本纪要。
- 终点性能数值只在 Fig.1 曲线中：如需精确终点 Pass@1 / Elo，需读原图估读，本纪要未做。

## 原文金句（1-2句）

> "the unit of optimization objective should match the unit of reward" —— 论文 §3，GSPO 全部设计的出发点：奖励给的是整条序列，纠偏就应在序列级做

> "GSPO makes it possible to directly use the likelihoods returned by the inference engine for optimization, thereby avoiding the need for recomputation with the training engine." —— 论文 §5.4，序列级似然对 RL infra 的直接简化
