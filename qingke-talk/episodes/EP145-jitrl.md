# EP145 — JitRL：无需梯度更新的即时强化学习

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP145-jitrl.html

> "We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective." —— JitRL 论文 Abstract：给 logits 加优势偏置不是启发式，而是带 KL 约束的 RL 目标的精确解

## 元信息

- 期号：145（官网期，无 B站视频）
- 标题：JitRL——无需梯度更新的即时强化学习（副题：Agentic RL 下半场）
- BV：无——B站合集无对应视频（注意编号冲突：B站合集第 145 期是《超似 3D 人体模型：Neural Image Processing》，与本期无关）
- 直播时间：2026-08-11 20:00–21:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：李一博（新加坡国立大学 NUS 计算学院博士生；官网预告嘉宾介绍）
- 相关论文：Yibo Li, Zijie Lin, Ailin Deng 等（NUS），*Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates*，https://arxiv.org/abs/2601.18510（v4, 2026-09-27）；ICML 2026 Spotlight（官网预告）
- 相关代码：https://github.com/liushiliushi/JitRL
- 官网预告：https://qingkeai.online/blog/JitRL

> ⚠️ 提炼方式说明：本期为官网期，B站合集无对应视频，字幕无从获取。本纪要根据该期对应的公开材料还原——官网预告（含讲者与提纲）+ JitRL 论文原文（arXiv:2601.18510）。Q&A 即兴内容未覆盖。

## 一句话总结

JitRL 让冻结权重的 LLM 在推理时"边用边学"：维护一个存了 $\langle$ 状态, 动作, 折扣回报 $\rangle$ 三元组的非参数经验记忆，推理时检索相似历史轨迹在线估计每个候选动作的优势 $\hat{A}(s,a)$ ，再把 $z'(s,a)=z(s,a)+\beta\cdot\hat{A}(s,a)$ 直接加到输出 logits 上。论文证明这正是"最大化优势 + KL 惩罚"目标的精确闭式解（定理 4.1），因此零梯度、零遗忘，还能给闭源 API 模型用；WebArena 与 Jericho 上超过所有 training-free 基线并击败微调方法 WebRL，货币成本约低 30 倍以上。

## 核心

### 背景/问题：持续学习的三条老路都不通

论文 §1 的问题设定：LLM 部署后权重冻结，持续适应只能三选一——传统 RL 要大量数据与算力、频繁更新、还会灾难性遗忘；ICL 类方法（Reflexion 等把经验写进 prompt）随任务变长上下文爆炸，且只能学"能用文字说清的"东西，学不了 RL 式的复杂技能。核心问题：**能否不用参数更新，实现等价于 RL 的持续策略改进？**

### 方法/设计：记忆 → 检索式价值估计 → 闭式 logit 更新

三件套环环相扣（论文 §4）：

1. **反思式记忆构建**：每个 episode 结束后，用 LLM 评估器对完整轨迹做逐步信用分配，生成 step-wise 奖励 $\{r_t\}$ ，再聚合成折扣回报：

$$G_t=\sum_{u=t}^{T}\gamma^{u-t}r_u$$

原始观测（完整 DOM、冗长游戏文本）先抽象成紧凑结构化状态，三元组 $(s_t,a_t,G_t)$ 存入动态记忆 $\mathcal{M}$ ——记忆不是文本日志，而是一个非参数的经验分布。

2. **检索式价值估计（免价值网络）**：推理时按 Jaccard 相似度取 top- $k$ 邻居 $\mathcal{N}(s)$ ，状态价值是邻居回报均值；某候选动作有历史证据时其 $Q$ 值是取过该动作的邻居回报均值，无证据时以概率 $\lambda$ 给乐观探索奖励 $\alpha/|\mathcal{N}(s)|$ （邻居越少越该探索）。优势即 $\hat{A}(s,a)=\hat{Q}(s,a)-\hat{V}(s)$ ，再按最大绝对值归一化。候选集是模型候选与记忆中动作的并集，模型从没提过的历史动作也能被"捞"回来（logit 初始化为 0 后参与竞争）。

3. **闭式 logit 更新（理论核心）**：目标是找在 KL 惩罚下最大化期望优势的策略 $\pi^{*}$ ，论文定理 4.1 证明其最优解取对数后恰好是 logits 的线性加法：

$$z'(s,a)=z(s,a)+\beta\cdot\hat{A}(s,a)$$

定理 4.2/4.3 进一步证明非平稳策略下 $\hat{V}$ 、 $\hat{Q}$ 、 $\hat{A}$ 估计依概率收敛到真值、整体更新收敛到最优策略。黑盒模型拿不到 logprobs 时，论文给了 verbalized logit 变体（让模型自报 0–100 置信度再变换为 logits），这是"闭源 API 也能 RL"的工程通道（官网预告重点强调的即插即用即指此）。

### 实验/实战（论文 §5）

评测协议模拟真实部署：每个任务连跑 $L=5$ 个 episode，报告 Avg（全部尝试平均成功率）与 Final（最后一个 episode 的成功率），两者差距即学习速度。training-free 基线统一用 Gemini-2.5-flash 骨干。

- **WebArena（Table 1）**：JitRL 全域微平均 Avg 46.98% / Final 51.35%，静态基线（Static）为 35.63% / 36.30%，最强 training-free 基线在 41–43% 量级。结构化域增益最大：Shopping 相对 Static +73.2%（论文 §5.2.1）；Final 与 Avg 的明显差距说明它确实在"边跑边学"。
- **对微调方法（Table 2）**：在 WebRL 的留出测试集 WebArena-Lite 上，JitRL Final 60.00%，高于 WebRL 的 46.06% 与 SFT 的 23.00%——纯推理期优化击败了权重更新方法。
- **Jericho 文字游戏（Table 3，50 个 episode）**：Zork1 平均分 53.0 / 最终 69，远超 GRPO（16.2 / 10）与最强 training-free 基线 EvoTest（46.8 / 54）；Library、Zork3 同样第一。学习曲线显示前 10–15 个 episode 快速起步、后期方差收窄；对照之下 GRPO 在稀疏奖励上全程高方差。
- **跨骨干与未见任务（Table 4–5）**：GPT-5-mini、DeepSeek-V3.2 上同样最优；只允许检索不相交任务的记忆时仍领先基线，说明迁移的是抽象程序性知识；跨任务记忆约占检索内容的近 50%（Table 6）。
- **消融（Table 8、Fig.4）**：同样的检索信息，logit 更新优于塞进 prompt（Admin 52.31 vs 48.35）——长上下文里模型会"看不见"检索线索，直接调分布更可靠；检索邻居数 $k$ 在 8–14 间稳健，过小估计方差大、过大噪声拖慢收敛。
- **成本（Table 9）**：WebRL（Llama-3.1-70B 在 H200 上训练）约 \$9900，JitRL 按 API 价格计约 \$290，其余 training-free 方法同为 \$200 上下——JitRL 以推理级成本拿到超过微调的效果，论文口径为货币成本降低 30 倍以上。
- **论文自陈局限（Limitations）**：只能在基座模型会提出的候选动作里重加权，发现不了基座永不生成的动作；依赖 LLM 评估器的信用分配质量；状态以文本表示，棋盘等空间模式难以文本化的任务上检索会失效。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| WebArena 全域成功率（Avg / Final） | Static 35.63% / 36.30% | JitRL 46.98% / 51.35% | 论文 Table 1 |
| WebArena Shopping 域 | Static | 相对 +73.2% | 论文 §5.2.1 |
| WebArena-Lite Final | WebRL 46.06%、SFT 23.00% | JitRL 60.00% | 论文 Table 2 |
| Jericho Zork1（Avg / Final 分） | GRPO 16.2 / 10 | JitRL 53.0 / 69 | 论文 Table 3 |
| 货币成本 | WebRL 约 \$9900 | JitRL 约 \$290（30 倍以上降低） | 论文 Table 9、Abstract |
| 检索邻居数 $k$ 稳健区间 | — | 8–14 | 论文 §5.6、Fig.4 |
| Logit 更新 vs Prompt 更新（Admin） | Prompt 版 48.35 | Logit 版 52.31 | 论文 Table 8 |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **"检索 + 优势"可以替代一部分在线 RL**：对重复性高的 agent 任务（同类工单、同类代码修复），维护 $\langle$ 状态, 动作, 回报 $\rangle$ 记忆并在解码时做优势加权，是零训练成本拿到持续改进的路径，且天然无灾难性遗忘——适合先在低风险流量上验证。
  2. **评估器即信用分配器**：JitRL 的回报质量完全押在 episode 后的反思打分上；做轨迹数据生产时，把"逐步归因打分 + 折扣回报"做成标准后处理，比只存最终成败标签的信息量大得多。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. 成本结构对比鲜明：一次 WebRL 式微调约 \$9900 vs JitRL 约 \$290（论文 Table 9），且后者是随用随付的推理成本——评估"要不要为持续学习建训练流水线"时，这是现成的量级参照。
  2. verbalized logit 变体说明：没有 logprobs 接口的黑盒模型也能接入策略优化层，代价是置信度自报的校准质量；为 agent 平台设计 API 时，暴露候选动作级分数值得作为一等接口考虑。
  3. 与本仓库 EP140 DRIFT 对照：DRIFT 在训练期用难度路由分配蒸馏/RL 信号，JitRL 在推理期用记忆做策略改进——两者是"自进化"在训练侧与推理侧的两种落点。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：候选集里"记忆独有动作"（模型当前不提、历史上出现过）的 logit 初始化为 0 再竞争，在动作空间很大的开放任务上候选集如何扩展与剪枝？想细看论文附录 F 的状态表示与动作处理。
- 现场内容不可还原：讲者对"Agentic RL 下半场"的判断、AMA 讨论无法从论文推知；若 B站后续补录视频，应以视频为准修订本纪要。

## 原文金句（1-2句）

> "Rather than merely retrieving text for in-context learning, our method treats memory as a non-parametric policy distribution." —— 论文 §2，JitRL 与一众记忆增强方法的分界线

> "JitRL serves as a highly efficient alternative to computationally intensive training methods." —— 论文 §5.2.1 对 WebArena-Lite 击败 WebRL 的定性
