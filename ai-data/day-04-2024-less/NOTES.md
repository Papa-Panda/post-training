## 元信息
- Title: LESS: Selecting Influential Data for Targeted Instruction Tuning
- Authors / Org: Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen / Princeton NLP
- Link / arXiv: https://arxiv.org/abs/2402.04333
- Code: https://github.com/princeton-nlp/LESS
- Date read: 2026-08-06
- Tags: [sft, data-selection, influence, data-attribution, instruction-tuning, curation, targeted]

## 一句话总结
为想定向提升的能力（推理/BBH/MMLU）准备几条 few-shot 锚点，把大池子 27万 条指令的 low-rank 梯度跟锚点做相似度搜索，只训 top 5% 的数据，经常比训全量还好，且用 7B 选的数据能直接给 13B/Mistral 用。

## 核心
1.  **Motivation**: 全量指令微调混了太多水数据，想提升某个专项能力时，大部分数据是 noise。传统 BM25 / embedding 选的是表面像的，训了没用。需要按“对目标 loss 的实际影响”来选。
2.  **Data Pipeline**: 
    - Warmup: 随机抽 5% 数据把 base (Llama-2) 热一下，让梯度不要是纯噪
    - Gradient Datastore: 只取 LoRA adapter 的梯度，随机投影到 8192 维 (JL引理保点积)，建一次库可复用
    - Scoring: `score(z)=cos(mean_g_target, g_low(z))`，优化目标是 Adam 修正后的 influence `η * grad_target^T Gamma(z)`，其中 `Gamma = Adam_precond(grad)`
    - Select: 取分数最高的 5% (≈13k) 去做 instruction tuning
3.  **Key Tricks**: 
    - 优化器感知: 不是 SGD 点积，用 Adam 的 m/v 修正后的梯度
    - LoRA + 随机投影: 全参梯度 7B 维存不下，LoRA 降到几百万再 JL 到 8192
    - Warmup: 不 warmup 梯度全是噪，相似度排名失效
4.  **Results**: 5% LESS 选的数据训 Llama-2-7B，在 MMLU/BBH/TyDiQA 上常打赢 27万 全量；随机 5% / BM25 / RDS 都输。Transfer：7B 选的数据给 13B / Mistral-7B 用同样赢。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 拿 10 条 LiveCodeBench 难例当 target，用你 1B 小 proxy 建 gradient 库，筛 5% 合成 code 数据试 pass@k，对比随机 5%
  2. 用 self-influence 思路把 coding SFT 池子里 high high self-influence 的脏/需强记样本清掉，留推理型
- Infra 视角：gradient datastore 建一次多任务复用，成本 O(N) 建库后每次选数据都是 O(N) cosine，适合 fly-wheels；可扩展到 RLHF 数据筛选。

## 疑问 / 下一步
- 如果 target 是 code generation 而不是推理，few-shot 靶子要怎么写才能让梯度更准？是不是要用 execution trace 而不是最终答案？
- 小 proxy 太弱时 transfer 失效的临界点在哪？对 coding 1B proxy 够吗？

## 原文金句 (1-2句)
> Instruction tuning on a LESS-selected 5% of the data can often outperform training on the full dataset. — and the selected data is highly transferable across models.

> Our method goes beyond surface form cues to identify data that exemplifies the necessary reasoning skills.

## 官方 Repo
- GitHub: https://github.com/princeton-nlp/LESS — 包含 warmup / datastore / selection / train / eval 全流程代码


## 第二轮复习（2026-09-04）

### 1. 核心命题

LESS 真正解决的 data 问题：**指令池是能力异构混合物时，如何为"定向能力"选数据**。它不回答"哪条数据质量高"（那是 DEITA / Day 20 的问题）、不回答"哪条难"（SuperFiltering / Day 12 的问题）、也不回答"哪条文本像目标"（BM25 / DSIR 的问题），而是回答：**哪条数据被训练之后，能把模型推向目标 loss 的下降方向**。

核心命题是把"数据价值"从文本的内在属性重定义为**关系属性**——由 target 梯度与优化器几何共同定义。5% 打赢全量的本质不是"数据越少越好"，而是把有限的更新预算集中到目标对齐的方向上，稀释掉 95% 与目标正交或反向的梯度噪声。这是一个 target-scoped 的结论：LESS 从不声称这 5% 是普适更好的数据集，只声称它是对**具名目标**更好的数据集。"少即是多"在这里是"定向集中"的推论，不是信条。

### 2. 图谱位置

- **直接前驱 Day 03 TracIn**：LESS 继承"梯度点积 = 影响"的记账思想，但做了三处关键改动：(a) SGD 一步点积 → Adam 感知的 update representation；(b) 全参梯度 → LoRA 梯度 + JL 随机投影到 8192 维（可扩展性）；(c) 多 checkpoint 累加 → 单个 warmup 后 checkpoint + few-shot target 锚点（目标条件化）。TracIn 回答"这条数据在真实训练路径上是否推过目标"，LESS 回答"这条数据若被训一步，是否沿 Adam 真实更新方向推目标"。
- **谱系源头 Day 02 Influence**：LESS 是 influence 谱系的实用分支——放弃 endpoint 反事实，即下式中的曲率重优化：
$$S_{\mathrm{IF}}(z,z_*)=g_*^\top H^{-1}g_z$$
保留的是"一步更新对目标 loss 的即时作用"。Day 02 / 03 复习里推过：Adam 下一步的真实作用是
$$L_*(\theta+\Delta\theta_z)-L_*(\theta)\approx g_*^\top\Delta\theta_z$$
而不是 SGD 近似 $$\eta g_*^\top g_z$$。LESS 正是把那个注脚变成了方法：用 Adam 预条件替换 Hessian 曲率修正。
- **平行 / 互补 Day 05 DataInf（重点直接对比）**：同一块 LoRA 梯度 substrate，走了相反方向。DataInf 回到 Influence 路线，用经验 Fisher 做闭式近似
$$(1/n)\sum_i g_i g_i^\top+\lambda I\approx H$$
算的是**每条样本对 test loss 的近似精确影响分**（1 秒一条，适合扫脏 / 找负影响样本）；LESS 用 Adam 预条件 + cosine，算的是**候选与目标梯度的方向对齐**，建一次库后每次选择是 O(N) cosine（适合批量定向选）。对照：DataInf 的 Fisher 逆扮演**曲率**角色（对应 Influence 的 $$H^{-1}$$，回答"扰动被训练集拉回多少"），LESS 的 Adam 预条件扮演**优化器**角色（回答"训练实际会走哪一步"）；DataInf 是 per-sample 精确归因工具，LESS 是 target-conditioned 批量选择器。给 coding 的分工因此是：DataInf 扫脏（找拖累 HumanEval 的负影响样本），LESS 定向选（为 LiveCodeBench 难例挑 5%）。两者共享 LoRA 假设，恰好互补。
- **便宜替代 Day 12 SuperFiltering**：125M 弱模型 IFD 做 target-free 难度筛选，是 LESS 的"粗筛版"。两篇共同验证了"选择信号跨尺度迁移"（125M 选的约等于 7B 选的；7B 选的可给 13B / Mistral 用）。但二者选的不是同一种东西：SuperFiltering 选"内在难"（与目标无关），LESS 选"对目标有用"（目标条件化）。工程正确组合是 cascade：SuperFiltering 粗筛到 20% → LESS 精选到 5%，而非二选一（Day 12 NOTES 已有此判断，本轮复核确认为正确）。
- **后继呼应**：Day 11 LIMR 是 LESS 思想在 RL 轨迹上的版本（轨迹对齐选 1.3k 难例）；Day 17 / 18 LIMO / s1 走了另一条"少即是多"路线（人工策展质量而非梯度影响）；Day 20 DEITA 的三因子 $$s=c\times q\times \mathrm{diversity}$$ 里，LESS 只贡献了 influence 轴——缺的 diversity 轴正是第 4 节的结构性缺口。

### 3. 机制深挖

**(a) Warmup：让 score 有方差的必要条件。** 在 base Llama-2 上，指令数据的 loss 主要由"学 chat 格式 / 长度"主导，所有候选梯度共享一个巨大的"格式学习"方向，cosine 会坍缩（全都接近 1，排不出序）。随机 5% warmup 把参数推入"格式已学会、推理成分可分"的 basin，候选梯度才分裂出可比的方向差异。同时 warmup 产出 Adam 的 optimizer state（m, v）——这是 Adam-aware 表示的必需品。Warmup 不是修辞性的"热身"，它是信号存在的前提。

**(b) Adam-aware 表示。** SGD 近似下，一步更新对目标 loss 的影响是 $$-\eta g_*^\top g_z$$。但实际训练走 Adam：
$$\Delta\theta=-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$$
LESS 定义 $$\Gamma(z)=\hat m_z/(\sqrt{\hat v_z}+\epsilon)$$ 为"样本 z 诱导的 Adam 更新方向"，score 取
$$\mathrm{score}(z)=\cos(\bar\Gamma_{\mathrm{target}},\Gamma(z))$$
这就是 optimizer-aware influence：衡量的不是梯度本身，而是**优化器真正会走的那一步**。注意 $$\bar\Gamma_{\mathrm{target}}$$ 是对 target 锚点取平均——均值向量的方向即"目标技能"的估计方向。

**(c) 为什么用 cosine 而不用 dot。** 梯度 norm 被 token 数和"惊讶度"主导，与"有用性"弱相关；dot product 会被长序列 / 高 loss 异常值劫持。cosine 丢掉 magnitude 换排名鲁棒性——代价是同方向下"推得多 vs 推得少"无法区分。论文 ablation 证实 cosine 优于 dot：这是用校准精度换稳健性的深思熟虑 trade-off，不是不加思考的归一化。

**(d) LoRA + JL 投影：可扩展性的代价。** 7B 全参梯度 × 27 万条存不下 → 只取 LoRA adapter 梯度（数百万维）→ JL 随机投影到 8192 维保内积。藏了两个强假设：① 推理能力的"有用方向"能被低秩更新表达（LoRA 子空间限制）；② JL 保的是固定向量对的内积，而选择用的是均值目标向量与候选的 cosine——期望成立，但单个候选的排名噪声会被投影放大。DataInf 恰好站在同一个 LoRA substrate 上走了闭式路线：两篇的分叉点不在数据假设，而在"要不要显式处理曲率 / 优化器"。

**(e) Target 锚点 = 单均值向量的单簇检索。** target 是约 10 条 few-shot 例子的梯度均值。均值是一个方向 → 选择本质是单簇检索。若目标能力需要多种异构技能（如 BBH 同时要符号操作和世界知识），均值只指向主导技能，次要技能的数据得分系统性偏低。**LESS 没有 diversity 项**——这是它相对 Vendi / Day 19、D4 / Day 24、DEITA / Day 20 的结构性缺口，也是第 6 节思考题的出发点。

### 4. 边界与反例

1. **Target 锚点误设**：few-shot 例子必须代表目标"技能"而非"格式"。若锚点全是特定措辞的选择题，选出来的是"同格式"不是"同技能"。对 coding：只用最终答案做锚点会漏掉 execution trace 推理，因为
$$\nabla_\theta L(\text{answer})$$
与下式指向不同方向：
$$\nabla_\theta L(\text{trace})$$
这呼应初读疑问"是不是要用 execution trace 而不是最终答案"：答案是肯定的，至少对推理型 coding 任务。
2. **单均值坍缩 + 无去重**：top 5% 可能全是与均值最像的近重复模板（13k 条"let's think step by step"变体）。论文**没有度量选中集合内部的冗余度**。反例构造：池中若有 10k 条同一模板的改写，LESS 会整块选入；Vendi / D4 则会惩罚这种冗余。这是纯 influence 选择的阿喀琉斯之踵。
3. **静态一次性选择**：score 在 warmup checkpoint 上算一次；训练 100 步后，某条"当时有用"的数据可能已被其他选中样本覆盖（diminishing returns 未建模）。没有迭代重选机制——与 flywheel 式持续选择不兼容。
4. **Transfer 未探的边界**：7B → 13B / Mistral 验证了，但"1B 弱 proxy 选给 70B 用"的临界点没测。SuperFiltering 证明"难度排名"可从 125M 迁移，但"目标条件梯度方向"能否从弱 proxy 迁移是另一个问题：弱模型的梯度方向本身可能是噪的，cosine 会放大这种噪。
5. **证据没证明什么**：5% > 全量只在目标 benchmark（MMLU / BBH / TyDiQA）上成立；**没测通用能力的"定向税"**——为专项牺牲了多少通用性未知。也**没对比"top-5% + 随机 5%"混合**，无法区分"去掉噪声"和"选中精华"各自的贡献。严格说，论文证明的是"去掉 95% 不伤害目标"，而不是"选出的 5% 是最优的 5%"。
6. **LoRA 子空间限制**：influence 只在 LoRA 参数空间度量；若最终做全参训练，排名可能错位。论文自洽（选和训都用 LoRA），迁移到全参 SFT 时需重新验证。

### 5. 迁移到 coding / post-training data

**可执行方案：双锚点 LESS + 去重 cascade（针对 500k 合成 coding 池）**

1. 造**两个** target 锚点（对冲单均值坍缩）：锚点 A = 10 道 LiveCodeBench 竞赛级难例（含 reference solution + **execution trace**，不只有最终代码）；锚点 B = 10 个 repo 级 API / 库使用任务。分别算均值 Adam 表示（记为 $$\bar\Gamma_A,\bar\Gamma_B$$）。
2. Warmup：1B code proxy 在池中随机 5% 上 warmup（省成本；proxy transfer 有效性本身就是待验证假设，先小规模验证再放大）。
3. 建库：LoRA 梯度 + JL 投影到 8192 维，对 500k 全池算分；分别按与 $$\bar\Gamma_A,\bar\Gamma_B$$ 的 cosine 各取 top 2.5%，合并去重得约 5%。
4. **补上 LESS 原生缺失的多样性**：在选出的 5% 内跑 SemDeDup 式语义去重（Day 24），杀模板冗余——coding 池模板化严重，这一步不是可选。
5. 对照实验：训三组——LESS 双锚点 5%、随机 5%、SuperFiltering 式 IFD 10%——在 HumanEval+ / LiveCodeBench 上比 pass@k，并记录 MMLU 看定向税。
6. 预算不足时的便宜 cascade：先用 350M CodeT5 的 IFD 粗筛到 20%（Day 12），再在 20% 上做 LESS 精选到 5%。

### 6. 今天的一道思考题

综合 **Day 04 LESS、Day 19 Vendi、Day 24 D4 / SemDeDup、Day 20 DEITA**：LESS 用单一均值目标梯度做 cosine 检索选 top 5%，完全没有多样性项；Vendi 给出多样性的公理化度量（kernel 特征值熵），D4 用"原型式剪枝"在 embedding 空间保证覆盖，DEITA 把选择分解为 complexity × quality × diversity 三因子。

请设计一个"影响力 + 多样性"的联合选择目标：

- 写出数学形式（例如次模目标 $$\max_{|S|=k}\sum_{z\in S}s(z)-\lambda R(S)$$，或 DEITA 式的乘法三因子）。
- 明确 influence 得分与冗余项的记号：前者记为 $$s(z)$$
- 后者记为 $$R(S)$$
- diversity 项应取什么**可计算**形式（覆盖 Vendi 的 kernel 熵与 D4 的原型距离两种思路，说明各自计算代价）。
- 论证 $$\lambda$$ 的选择依据：下式两种极限分别退化成哪种已知的选择策略：
$$\lambda\to 0,\qquad \lambda\to\infty$$
- 构造一个具体反例说明纯 LESS 会选出 13k 条近重复模板而你的目标不会——用"模板簇"的梯度几何解释：为什么同一模板的 $$\Gamma(z)$$ 彼此 cosine 接近（提示：共享的模板 token 主导梯度方向）。
- 最后用 DEITA 三因子回答：LESS 的 influence score 对应 c / q / diversity 中的哪一个？缺的两个在 coding SFT 里分别对应什么可计算的 proxy（各举一个，并说明为什么 IFD 只能算其中之一）？

答案不许只复述 LESS 原文；必须实质用到至少两篇对比论文的机制。

## 原文链接
- Paper: https://arxiv.org/abs/2402.04333
- GitHub NOTES: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-04-2024-less

## 问答补充（2026-09-04）

本节收录 2026-09-04 晚侧聊中的用户问答（已按"先问题、再推导、最后几何含义"整理为结构化条目），不改动上方既有内容。

### Q1：BM25 是什么

BM25 是经典的信息检索相关度公式：给定查询 $q$ 与文档池，按相关度排序文档。核心三项——词频（边际递减）、逆文档频率（罕见词权重高）、长度归一化（惩罚堆词长文档）：

$$\text{BM25}(d,q)=\sum_{t\in q}\underbrace{\text{IDF}(t)}_{\text{词重要性}}\cdot\underbrace{\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\frac{|d|}{\text{avgdl}})}}_{\text{词频 + 长度惩罚}}$$

与 LESS 的关系：BM25 看的是**词有没有出现**（纯词面匹配），LESS 看的是**梯度方向是否对齐**（模型驱动）。在 LESS 的实验里，BM25 代表"只看文本相似度"的便宜对照做法。

### Q2：BM25 并不是 30 篇论文家族的一篇（纠错）

用户指出："BM25在我们的论文家族里没有出现啊"。经核对，正确的说法是：**BM25 是 LESS 论文自己实验部分的一个对照 baseline**（与 random、DSIR、RDS 并列），代表"只用词频/文本相似度"的选数方法。论文结论：这种词频选法几乎没比随机好多少，LESS 的梯度对齐方法 consistently 强于它。

修正说明：本轮问答开始时的回答曾把 BM25 表述为"LESS 里经常出现的 baseline"，容易被误读为论文路线中的一篇；现修正为——BM25 是 LESS 论文**实验内部**的对照基线，不是我们 30 篇阅读路线的一篇。

### Q3：LESS 的数学怎么做的

记号：$$z=(x,y),\qquad \mathcal{V}=\text{few-shot 目标样本}$$

设 $$g_z=\nabla_\theta L(z,\theta)$$ 为候选数据 $z$ 的训练梯度，$$g_{\mathcal{V}}=\nabla_\theta L(\mathcal{V},\theta)$$ 为目标验证集的梯度，$$\Delta\theta_z$$ 为模型在 $z$ 上实际走的 optimizer update。

LESS 的问题：**让模型用 $z$ 做一次真实更新之后，目标 loss 下降多少？**

$$L_*(\theta+\Delta\theta_z)-L_*(\theta)\approx g_*^\top\Delta\theta_z$$

关键改动在 $\Delta\theta_z$ 的写法：TracIn / Influence 用 SGD 更新 $\eta g_z$，但实际训练用 Adam：

$$\Delta\theta_z=-\eta\,\frac{\hat m_z}{\sqrt{\hat v_z}+\epsilon}$$

其中 $\hat m_z$ 是梯度一阶矩（起平滑作用），$\hat v_z$ 是二阶矩（按方向缩放）。LESS 定义 **Adam-aware 表示**：

$$\Gamma(z)=\frac{\hat m_z}{\sqrt{\hat v_z}+\epsilon}$$

score 是候选表示与目标均值表示的 cosine：

$$\mathrm{score}(z)=\cos\left(\bar\Gamma_{\mathcal{V}},\ \Gamma(z)\right)$$

其中 $\bar\Gamma_{\mathcal{V}}$ 是 few-shot 目标样本的平均 Adam 表示。

为什么用 cosine 而不是点积：TracIn 的点积 $g_*^\top g_z$ 同时包含方向和大小；LESS 故意丢掉大小只留方向——优点是排名对高梯度噪声样本不敏感，代价是同方向下"推得多 vs 推得少"无法区分。这是**有意的取舍**，不是随意改的。

工程上两层压缩：只取 **LoRA 梯度**（假设推理能力的"有用方向"能被低秩空间表达）；随机投影到 8192 维（Johnson–Lindenstrauss 保证内积大体保住）。注意：JL 保的是**内积**，而 LESS 用的是 **cosine**，投影后必须重归一化——这是"便宜但有噪声"的来源。

为什么必须 warmup：base 模型上所有指令的梯度共享一个巨大的"学 chat 格式"方向，cosine 会坍缩成几乎一样。随机 5% 数据 warmup 把参数推入"格式已学会、推理成分可分"的 basin；同时 warmup 产出 Adam 的 $m/v$ 状态，否则 $\Gamma(z)$ 算不出来。

图谱位置：前驱是 TracIn 的梯度点积思想；改动是把 $\eta g_*^\top g_z$ 换成 $\cos(\bar\Gamma_{\mathcal{V}},\Gamma(z))$；缺口是目标只是一个均值向量——异构多技能目标下均值只指向主导技能，且完全没有 diversity 项（呼应上方"第二轮复习"第 2 点：单均值坍缩 + 无去重）。

一句话：**LESS = 用 Adam 视角回答 TracIn 的问题，但丢了 magnitude 和 diversity。**

### Q4："LESS 中实际训练用的是 Adam 从而导出 cosine，没看懂"（追问）

关键澄清："从 Adam 导出 cosine"其实是**两个独立的决定**。

TracIn 用 $\eta\,g_*^\top g_z$，等于在假设"模型走的是 SGD 更新"；实际走的是 Adam 更新，正确的即时作用应该是：

$$g_*^\top\Delta\theta_z^{\text{Adam}}=-\eta\,g_*^\top\frac{\hat m_z}{\sqrt{\hat v_z}+\epsilon}$$

1. **用 Adam 表示代替原始梯度**——这是从优化器真实性来的，是 LESS 的核心改进；
2. **用 cosine 代替点积**——这是工程取舍，丢掉梯度大小换排名鲁棒性。

一句话：**Adam 解释了为什么用 $\Gamma(z)$，工程鲁棒性解释了为什么用 cosine。**

### Q5："所以并非严格数学，只是一种类比"（确认 + 精确化）

用户总结得到确认。严格说法分两层：

第一层数学上**成立**：把真实训练的 Adam 更新代入 Taylor 展开：

$$L_*(\theta+\Delta\theta_z)-L_*(\theta)\approx -\eta\,g_*^\top\frac{\hat m_z}{\sqrt{\hat v_z}+\epsilon}$$

只要 $\hat m_z,\hat v_z$ 是当时真实的 Adam 状态。

第二层就不严格了。LESS 没有直接用这个点积，而是用 $\cos(\bar\Gamma_{\text{target}},\Gamma(z))$；这个改动做了三件事，每件都有代价：

- **丢掉学习率 $\eta$**：所有样本都乘同一个 $\eta$，不影响排名；
- **目标梯度 $g_*$ 换成 $\bar\Gamma_{\text{target}}$**：目标也套了 Adam 预处理，但真实的 loss 下降用的是原始 $g_*$，不是预处理后的；
- **点积换 cosine**：丢掉梯度大小，只留方向。

即：$$g_*^\top\Delta\theta_z\quad\Longrightarrow\quad\cos(\bar\Gamma_{\text{target}},\Gamma(z))$$

中间是一个"去掉大小、只看对齐"的类比跳跃，**不是等价变换**。LESS 的价值在于把 TracIn 的 SGD 近似换成了更接近训练现实的 Adam 视角；但若把它当成严格的反事实数学，就走远了。

**知识图谱关系**：Q3–Q5 共同构成了"LESS 计分规则的完整推导链条"，上接本 NOTES"核心 §2 Data Pipeline（Adam 修正后的 influence）"与"第二轮复习 §1 核心命题"，下接 Day 19 Vendi / Day 24 D4 的"单均值 + 无 diversity"缺口讨论。Q1–Q2 澄清了 BM25 的准确位置：LESS 实验的文本相似度对照基线（与 random、DSIR、RDS 并列），非 30 篇阅读路线成员。
