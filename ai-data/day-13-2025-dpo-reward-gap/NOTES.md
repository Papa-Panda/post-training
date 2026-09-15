# Paper 模板

## 元信息
- Title: Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap
- Authors / Org: Xuan Qi, Rongwu Xu, Zhijing Jin - UW / Tsinghua / MPI
- Link / arXiv: https://arxiv.org/abs/2508.04149
- Date read: 2026-08-13
- Tags: [rl-data, dpo, preference-data, data-selection, alignment, coding-data]

## 一句话总结
做 DPO 对齐别全量上，算 DPO 隐式 reward 的 chosen-rejected gap，gap 小的难例留 10%，对齐效果打赢全量，证明难的偏好对才值钱。

## 核心
1.  **Motivation**: RLHF/DPO 都靠大偏好集，贵，且高质量偏好怎么选没人说清
2.  **Data Pipeline**: 拿 base 模型算 DPO implicit reward gap -> 按 gap 排序 -> gap 小的当难例留 10% -> 训 DPO/Reward Model。理论是 gap 小=模型分不清=学习信号大。
3.  **Key Tricks**: 
    - 隐式 reward gap 不用外部 RM，直接用 DPO 公式算，10% 就超全量【6968311559756508151†L25-L29】
    - 在 RewardBench 上 75% 维度打赢其他基线【6968311559756508151†L106-L110】
    - 比 External Margin / IFD-Z 的 Low-Gap 更稳，因为直击 DPO 学习潜力【6968311559756508151†L100-L105】
4.  **Results**: 10% 难例在 AlpacaEval 2.0 / RewardBench 上超 5 个强基线，接近或超全量，常在 Chat-Hard, Safety, Reasoning 上赢【6968311559756508151†L95-L98】

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - DPO 做 coding 偏好时（正确vs错解），用 implicit gap 选 reward 接近的难对，踢掉一眼就能分出来的简单对
  - 把 execution 信号和 gap 融合：gap 小且一个过测一个不过的 pair 优先级最高
- Infra 视角：RLHF 洗偏好对成本最高，用 10% 难对降 90% 采样和标注，7B 级最划算

## 和之前工作的关系
是 LESS 的偏好版。LESS 挑 SFT，这个挑 DPO，都是少即是多。和 Day 5 DataInf 互补，DataInf 踢脏数据，这个留难数据。和 Day 8-10 的后训练闭环吻合：Qwen2.5/Llama3.1 的两轮 RL 前先做这个 reward-gap 筛。

## 疑问 / 下一步
- coding 上 implicit gap 会不会被 execution 颠覆？gap 小但一个能跑一个不能跑，怎么融合两个信号？

## 原文金句
> By selecting preference data examples with smaller DPO implicit reward gaps, which are indicative of more challenging cases, we improve data efficiency【6968311559756508151†L25-L28】

> achieving superior performance with only 10% of the original data【6968311559756508151†L27-L30】

## 第二轮复习（2026-09-13）

> 本轮复核：arXiv 元信息与摘要今日核验（2508.04149，2025-08-06 提交，v2 2026-05-16；Xuan Qi / Rongwu Xu / Zhijing Jin）。ar5iv 全文今日不可达，核心数字（AlpacaEval 2.0 / RewardBench、5 个强基线、RewardBench 75% 维度、External Margin / IFD-Z 对照）沿用初读全文核对的行引用；机制部分为本轮重建。初读 NOTES 仅 34 行，未展开隐式奖励的数学——gap 与 DPO 梯度的关系、 $ (m,g) $ 二维分解、静态 vs 动态选择均为本轮新增。

### 1. 核心命题

Day 13 真正解决的 data 问题：**偏好数据的"学习价值"在训练前是否可观测**。DPO 的训练信号不是"pair 质量高低"，而是每个 pair 在 DPO 自己的 loss 里能产生多大的梯度——而这个量，在训练之前就能用模型自己算出来，不需要外部 RM。

DPO loss 的单 pair 梯度是 $ -\sigma(-g)\cdot\nabla_\theta g $ ，其中 $ g = \hat r(x,y_w)-\hat r(x,y_l) $ 是隐式奖励差， $ \hat r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} $ 。权重 $ \sigma(-g)=1/(1+e^{g}) $ ：gap 越小，权重越大；gap 已经很大（模型早就会区分）的 pair，权重趋于 0，对训练近乎零贡献。所以"选小 gap"不是"选难的"这种文学修辞，而是**在 DPO 自己的坐标系里做重要性采样**：留下的 10% 恰好是 DPO 梯度不为零的那些 pair。

这回答了初读疑问的一半：为什么 10% 能打赢全量——因为全量里 90% 的 pair 的 $ \sigma(-g) $ 接近 0，它们不是"水"，而是**已被学会**；DPO 在它们身上花的算力是纯浪费。这和 Day 04 LESS 的"5% 打赢全量"机制不同：LESS 是把更新预算集中到目标方向（去噪声），Day 13 是把更新预算集中到梯度非零处（去已学会）。"少即是多"在这里是"别在零梯度上浪费步数"的推论。

### 2. 图谱位置

- **直接上游 Day26 UltraFeedback（供给）**：UltraFeedback 回答"pair 从哪里来"（64k prompts × 4 模型回答 × GPT-4 细粒度打分）；Day 13 回答"留下哪些"。两者串成"造池 → 打分 → 难例选择"。注意 UltraFeedback 的 4-way score 是外部裁判（GPT-4）的坐标，Day 13 的 gap 是 DPO 自己的坐标——**筛选器的坐标必须和训练的坐标一致**，这是 Day 13 相对"用外部 RM margin 筛"（External Margin 基线）更稳的根因。
- **平行供给者 Day10 Llama 3.1/3.2（重点对比）**：Day 10 的第二轮复习已点出"供给 vs 筛选"：Llama 用 6 轮 RS+DPO 规模化生产 pairs，Day 13 告诉你其中 90% 可能是已学会的水。更深的联系是 Day 10 问答（2026-09-10）的 $ (m,g) $ 分解：记质量水位 $ m=(r_w+r_l)/2 $ ，落差 $ g=r_w-r_l $ 。RS 筛的是 $ m $ （要冲顶的高分正例），Day 13 筛的是 $ g $ （要小而为正的难对）——**两个过滤器活在正交坐标轴上**。RS 的顶分 $ y_w $ 配普通 $ y_l $ → $ g $ 大 → $ \sigma(-g)\approx 0 $ → 对 DPO 贡献为零：RS 正例的"顶"结构性地导致它在 DPO 里的"废"。这是 Day 10 第二轮复习思考题 (a) 的答案，也是 Day 13 的定位证明：**SFT 数据和 DPO 数据不能是同一批，筛选标准在数学上就是正交的**。
- **重点直接对比 Day04 LESS（同属"选择主线"，一题两解）**：初读说"LESS 的偏好版"，本轮精确化——相似的只有"少即是多"的外形，机制是镜像的：
  - 域：LESS 选 SFT 样本（单条 $ z $ ），Day 13 选偏好对（pair）。
  - 信号性质：LESS 是**关系型**（target 梯度相似度，要 few-shot 锚点）；Day 13 是**内在型**（pair 内部的 gap，无 target、无 warmup、无 anchor）——在这个维度上 Day 13 更像 Day 12 SuperFiltering 的兄弟（IFD 也是内在型、plug-and-play），只是把"指令难度"换成了"偏好区分难度"。
  - 成本：LESS 要 warmup + LoRA 梯度库 + JL 投影；Day 13 只要两次前向（ $ \pi_\theta $ 与 $ \pi_{\text{ref}} $ 的 logprob）——比 LESS 便宜一个数量级，且**不需要 warmup**（gap 的定义不依赖参数位置，只依赖两个分布的相对位置）。
  - 盲区互补：LESS 的盲区是"无多样性 + target 误设"；Day 13 的盲区是"不看绝对质量 $ m $ + 不看 prompt 多样性"（见 §4）。
- **后继呼应 Day11 LIMR**：都是"难例才值钱"，但 LIMR 的证据是**动态**的（rollout 轨迹对齐，训完才知道），Day 13 是**静态**的（训前一次打分）。LIMR §3.3 的警告（最难的可能是纯噪声）直接适用于 Day 13：静态 gap 分辨不出"可学的难"和"标注噪声的难"——这是 Day 13 相对 LIMR 的结构性弱点，也是 §6 思考题 (b) 的出发点。

### 3. 机制深挖

**(a) 隐式奖励的定义（符号逐项）**：对 prompt $ x $ 与响应 $ y $ ，
$$ \hat r(x,y) = \beta\,\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} $$
$ \pi_\theta $ = 打分时用的模型（通常是 SFT 后的模型，即 DPO 的起点）， $ \pi_{\text{ref}} $ = DPO 的参考模型（通常就是 SFT 模型本身）， $ \beta $ = KL 惩罚系数（DPO 超参）。 $ \hat r $ 不是外部 RM 学出来的，是 DPO loss 在数学上等价的那个"隐式"奖励——**它就是 DPO 优化目标里真实在用的奖励函数**。

**(b) gap 与 DPO 梯度的关系（本轮重建，初读未展开）**：记 pair 的 gap $ g = \hat r(x,y_w) - \hat r(x,y_l) $ （ $ y_w $ = chosen， $ y_l $ = rejected）。DPO loss $ \mathcal{L} = -\mathbb{E}[\log\sigma(g)] $ ，单 pair 梯度：
$$ \nabla_\theta \mathcal{L}_{\text{pair}} = -\,\sigma(-g)\,\nabla_\theta g,\qquad \sigma(-g)=\frac{1}{1+e^{g}} $$
逐项读： $ \nabla_\theta g $ 是"把 chosen 抬高、rejected 压低"的方向； $ \sigma(-g) $ 是该 pair 的**有效学习率**。 $ g\to +\infty $ （模型已能轻松区分）→ 权重 $ \to 0 $ ； $ g\approx 0 $ （模型分不清）→ 权重 $ \approx 0.5 $ （最大值）。所以"选小 gap" = **按 DPO 自己的有效学习率做重要性采样**，不是启发式，是 loss 结构的直接推论。

**(c) 为什么不用外部 RM（vs External Margin 基线）**：外部 RM 的 margin $ r_\phi(y_w)-r_\phi(y_l) $ 活在 $ \phi $ 的坐标系里；DPO 训练时真正走的是 $ \hat r $ 的坐标系。两个坐标系的"难"不是一回事——外部 RM 觉得难分的 pair，DPO 可能早就会了（权重为零）；反之亦然。初读记录的实验结论（gap 版比 External Margin / IFD-Z 的 Low-Gap 更稳【6968311559756508151†L100-L105】）的机制解释就是：**筛选信号与训练信号同坐标**，消掉了跨坐标的外推误差。这和 Day 09 问答里"评估器与训练参照系必须一致"的结论是同一条线。

**(d) Pipeline（数据视角）**：偏好池（人工 / UltraFeedback 式 AI 反馈 / RS 采样，pair 已带 chosen-rejected 标签）→ 对每条 pair 用 $ \pi_\theta,\pi_{\text{ref}} $ 各做一次前向得 logprob → 算 $ g $ → 按 $ g $ 升序 → 留最小的 10% → DPO。注意**标签不参与打分**：gap 只用模型自己的分布算，chosen/rejected 标签只在最后训练时用——筛选器是无监督的（相对标签而言）。

**(e) 成本账**：每条 pair 两次前向（可与推理 batch 复用），无训练、无 RM、无人工。相对"全量 DPO"的节省不在筛选（筛选本身便宜），而在**训练步数**：10% 数据 ≈ 10% 的 DPO 训练 FLOPs，且效果打赢全量——省的是训练算力，不是筛选算力。

### 4. 边界与反例

- **小 gap 混着两种东西：真难例与标注噪声**。 $ g $ 小的充分条件是"模型分不清"，但"分不清"有两个成因：(a) pair 真的难（chosen 只比 rejected 好一点点）；(b) 标签错了 / 两个回答质量无差异（annotator 抛硬币）。(b) 在 DPO 里学的是噪声，且**小 gap 选择会系统性富集 (b)**——因为噪声 pair 的 gap 分布天然集中在 0 附近。这是 Day 13 最大的方法论风险：池子越脏，"选难"越等于"选噪声"。纠正方向不是"别选小 gap"，而是设下限 $ g_{\min} $ ：只留 $ [g_{\min}, g_{\max}] $ 信息带（Day 10 问答已给出此结构），下限砍噪声角，上限砍死 pair。
- **gap 不看绝对质量 $ m $**：两个都很烂的回答（ $ m $ 极低）gap 也可以很小——"一样烂"不等于"难"。DPO 在这种 pair 上学的是"在垃圾里挑稍微不那么垃圾的"，对齐的是相对排序，不是绝对质量。UltraFeedback 式的细粒度绝对分（ $ m $ 的 proxy）正是补这个洞的——**Day 26 与 Day 13 的正确串联是二维的**：先用绝对分砍掉低 $ m $ 尾部，再在剩余池里按 gap 选难。
- **静态一次性打分**：gap 在 DPO 训练前算一次；训练 500 步后，当初的小 gap 可能已变大（学会了），当初的大 gap 相对变小。没有"训中重筛"机制——与 Day 10 的 6 轮重过滤、Day 11 的轨迹选择相比，这是静态选择的通用弱点。论文未做"多轮 gap 重筛"的消融。
- **$ (\pi_\theta,\pi_{\text{ref}}) $ 依赖**：gap 是相对于特定模型对定义的。换 base 模型、换 $ \beta $ ，排名会变。论文未验证"7B 选的 10% 给 13B 用"式的跨模型迁移（LESS 和 SuperFiltering 都做了跨尺度验证，Day 13 没做）——引用时不要默认它有 transfer。
- **10% 是预算不是定律**：论文证明"10% ≥ 全量"，没证明"10% 是最优的"。 $ 5\% $ vs $ 10\% $ vs $ 20\% $ 的曲线未报告；"越难越好"的单调性也未验证（LIMR 的教训：最难尾部可能是噪声）。
- **证据没证明什么**：初读记录"RewardBench 75% 维度打赢其他基线【6968311559756508151†L106-L110】"——这是相对基线的胜率，不是绝对分数；且**没有 coding / 推理专项评测**（HumanEval / MATH 类），对"coding 偏好对是否同样成立"无直接证据。初读疑问"coding 上 implicit gap 会不会被 execution 颠覆"依然开放——§5 给出可执行答案。

### 5. 迁移到 coding / post-training data

**可执行的映射：coding DPO 的"盲区挖掘"管线**（直接回答初读疑问"gap 小但一个能跑一个不能跑，怎么融合"）：

1. **造池**：对每个 coding prompt，用 SFT 模型采样 $ N=8 $ 个回答（温度 0.7–1.0 保多样性），跑单元测试得 pass/fail 标签。
2. **组 pair**：只组 (pass, fail) 对——chosen = pass，rejected = fail。跳过 (fail, fail)（两个都错，gap 无意义）和 (pass, pass)（无对比信号）。
3. **二维筛选**：算每对的隐式 gap $ g $ ，只留 $ g\in[g_{\min}, g_{\max}] $ ： $ g_{\min} $ 排除"模型几乎输出同一份代码"的噪声角（采样重复）， $ g_{\max} $ 排除模型早就会区分的死 pair。初值可设 $ g_{\min} $ 为全池 gap 分布的 5% 分位， $ g_{\max} $ 为 30% 分位——先小规模 DPO 验证再调。
4. **盲区优先**：在信息带内，按"execution 分歧大 × gap 小"排序——即**模型觉得两者差不多、但测试说一个过一个不过**的 pair 置顶。这是"模型盲区"的精确定义： $ \hat r $ 的坐标系里分不清，verifier 的坐标系里天壤之别。DPO 在这些 pair 上的 $ \sigma(-g) $ 权重最大，且学的是 verifier 能验证的真信号，不是标注噪声。
5. **对照实验**：三组 DPO——(a) 全量 (pass, fail) 对，(b) 随机 10%，(c) 盲区优先 10%——在 HumanEval+ / MBPP+ 上比 pass@1，并记录 $ g $ 分布随训练的变化（验证 §6(b) 的预测）。

**为什么这个映射对**：coding 有 execution 这个 Day 13 原论文没有的**外部真值**——它同时解决了 §4 的两个最大边界（标注噪声：测试即标签；both-bad：fail/fail 直接不组 pair）。Day 13 的方法在 coding 上不是照搬，而是"gap 选难 + execution 定真值"的二维版。

### 6. 今天的一道思考题

> **综合 Day13（DPO-Gap）、Day10（Llama RS + $ (m,g) $ 分解）、Day11（LIMR 轨迹选择）、Day26（UltraFeedback 造池）**：
>
> (a) **二维选择的象限图**。记质量水位 $ m=(r_w+r_l)/2 $ ，落差 $ g=r_w-r_l $ 。Day 13 只在 $ g $ 轴上切（留最小的 10%），Day 10 的 RS 只在 $ m $ 轴上切（留顶分）。请画出 $ (m,g) $ 平面的四个象限并回答：① "高 $ m $ + 小 $ g $"（双顶贴对角线）为什么是标注噪声角——用 $ \sigma(-g) $ 权重和 annotator 分歧率两个角度论证 DPO 在此学的是噪声；② "低 $ m $ + 小 $ g $"（一样烂）为什么 execution 信号能识别而 gap 不能——写出 coding 上的可操作规则（什么 pair 直接丢）；③ 真正的"金矿"象限是哪个，用一句话定义它（提示：Day 10 问答的结论是"中等水位、小落差"——请用 $ \sigma(-g) $ 和 $ m $ 的语言重新表述为什么）。
>
> (b) **静态 gap vs 动态轨迹：谁预测"学到了多少"？** LIMR 用训练后的轨迹相关性选数据（动态、后验），Day 13 用训练前的 gap 选数据（静态、先验）。设计一个实验检验"静态 gap 是否预测真实学习量"：在 DPO 训练的每个 checkpoint $ t $ 记录每条 pair 的 $ g_t $ ，定义该 pair 的学习量为 $ \Delta g = g_T - g_0 $ （gap 被拉开的程度）。问题：① 如果"小 $ g_0 $"的 pair 系统性地有更大的 $ \Delta g $ ，说明了什么（用 $ \sigma(-g) $ 权重解释）；② 如果反而"中等 $ g_0 $"的 $ \Delta g $ 最大、最小 $ g_0 $ 的 $ \Delta g\approx 0 $ ，说明了什么（提示：LIMR §3.3 "长期 near-zero 的硬题是纯噪声" + §4 的噪声角）；③ 这个实验需要几个 checkpoint、每个 checkpoint 算 $ g_t $ 的成本是多少（前向次数），和"直接全量训完看效果"相比，它回答了什么后者回答不了的问题？
>
> (c) **噪声富集的定量**。设池子标签噪声率为 $ \epsilon $ （chosen/rejected 标反或无差异），噪声 pair 的 gap 分布集中在 0 附近（方差 $ \sigma_n^2 $ ），干净难例的 gap 分布均值为 $ \mu_c>0 $ 。Day 13 留最小 10% gap：① 定性推导保留集里的有效噪声率 $ \epsilon' $ 与 $ \epsilon $ 的关系（是放大还是稀释，取决于什么）；② 不用人标，给出一个**可计算**的"噪声角污染度" proxy——coding 场景用 execution（写出公式），通用场景用"同一 prompt 多采样的一致性"（写出公式）；③ 基于 (c)② 的 proxy，写出 $ g_{\min} $ 的选择规则（不许回答"调参试出来"，要写出用 proxy 分布定阈值的规则）。

相关讨论（Gemini 网页版，2026-09-14）：https://gemini.google.com/app/debdc4a65f36cd4c

---

论文原文：https://arxiv.org/abs/2508.04149

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-13-2025-dpo-reward-gap/NOTES.md
