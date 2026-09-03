# 元信息
- Title: Estimating Training Data Influence by Tracing Gradient Descent (TracIn)
- Authors / Org: Garima Pruthi, Frederick Liu, Mukund Sundararajan, et al. / Google
- Link / arXiv: https://arxiv.org/abs/2002.08484
- PDF: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
- Date read: 2026-08-05
- Review date: 2026-09-03
- Tags: [data-attribution, tracin, influence-functions, coding-data, data-cleaning, self-influence]
- Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/
- Code: https://github.com/frederick0329/TracIn

## 一句话总结
不用 Hessian，把训练过程上每个 checkpoint 的梯度点积加起来估计影响 —— TracIn，工程上把 Influence Functions 做到了大模型可用的版本，是现在 LLM 数据清洗最实用的基线。

## 核心
### 1. Motivation
Influence Functions 要 $H^{-1}$，贵、不稳、非凸不成立。作者问：既然模型是 SGD 一步步走出来的，能不能直接看路上谁推了谁一把？理想的 influence 应该是整个轨迹上损失下降的累积。

### 2. Data Pipeline
- 训练时存 $K$ 个 checkpoint $\theta_{t_1}... \theta_{t_K}$，学习率 $\eta_t$
- 对任一训练点 $z$ 和测试点 $z'$：
  $$ TracIn(z,z') = \sum_{t} \eta_t \nabla L(z',\theta_t)^T \nabla L(z,\theta_t) $$
- 实际用 TracInCP：只用 checkpoint，忽略同一 checkpoint 内不同 step 的差异，batch 内近似
- Self-influence：$z'=z$ 时，分数越高，模型越靠死记这条点才能记住它

### 3. Key Tricks (3个最值得抄的)
1. **不要Hessian，只要点积**：$\eta \nabla_{test}\cdot\nabla_{train}$ 就是 influence。实现上就是两次 backward，算 cosine/dot，比 Influence Functions 快 10-100x
2. **Checkpoint 选择**：论文用最后几个 + 均匀采样，3-5个就够。实践：早期 checkpoint 抓语法/去重噪声，后期 checkpoint 抓语义难例。对 code 建议：epoch 1,2,末尾 各1个，共3个起步
3. **Self-influence = 脏数据探测器**：把训练集按 self-influence 排序，top 1% 拿去人工看，基本是 mislabeled / 爬到的孤岛代码 / 极长尾 API / 重复但标签矛盾的数据。Google 用这个清 10% 数据不掉点

### 4. Results / 用途验证
- CIFAR/MNIST 上：和真 LOO 相关性 >0.8，Influence Functions LiSSA 只有 0.6 且慢 10x
- 找 mislabeled：按 self-influence 排序，查 20% 数据就能找到 80% 错误标签，比 Influence Functions 快
- 工程上已用到 BERT/ResNet，checkpoint 存 3-10 个就行，内存可接受

## 可迁移
- **对你现在 coding data 工作的 1-2 个直接可试的点：**
  1. **脏数据过滤器 (今晚就能跑)**：用你 1B proxy 训 3 epoch，存 3 个 checkpoint，算所有训练 code 的 self-influence。Top 2% 导出，看是不是过时语言/错的API/从 StackOverflow 拷的带问号的代码。清掉再训 7B，看 HumanEval 有没有稳
  2. **合成数据价值评估**：合成的 code 不是全留。用 TracIn：对 LiveCodeBench 难例算 $TracIn(合成样本, 难例)$，平均为正才留。比“过得了单元测试就留”更贴近真实 eval
  3. **冗余 prune**：Helpful 的样本彼此梯度余弦相似度高，只留 1 个，呼应你 infra 省钱思维

- **Infra 视角：**
  - 不用二阶，天然 DDPer-friendly，checkpoint 存量 $K \times P$，可以只存 LoRA 分支梯度来把成本压到 1/100
  - 可扩展到 RL data：RLHF 偏好数据哪条最有用，用 TracIn 对 reward model 难例打分

## 疑问 / 下一步
- 3个 checkpoint 的采样策略对 code 任务是否最优？试早期密集 vs 均匀
- TracInCP 有偏，TracIn 论文里 checkpoint 内用一阶近似，误差多大？看后续 2023 TracIn++ / D-TracIn
- 今晚实验：跑通 `tracin_demo.py` 20 行版，对你手头小 code 集算 self-influence 分布，是否长尾

## 原文金句
> The influence of a training example can be estimated by tracing the loss reduction it contributes during training.

> Self-influence is a strong signal for mislabeled and outlier examples.

## 复现链接
- Official TF: https://github.com/frederick0329/TracIn
- PyTorch toy: `tracin_demo.py` (本文件夹)
- Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/


## 第二轮复习（2026-09-03）

### 1. 核心命题

这篇真正解决的 data 问题是：**能否不做逐点删数重训、也不求 Hessian 逆，而直接从已经发生的训练轨迹中估计“某条训练数据在什么时候、朝哪个方向改变了某个目标样本的 loss”？**

TracIn 把 influence 从 Day 02 的“最终解附近删掉一点会怎样”改写成“训练过程中，每次用到这条数据时，它给目标 loss 带来了多少局部变化”。因此它衡量的不是数据的静态质量，而是一个依赖 **目标样本、模型状态和训练路径** 的关系量。对目标 $z'$ 为正，表示该训练点在轨迹上总体与降低 $z'$ 的 loss 同向（proponent）；为负则表示总体抬高了 $z'$ 的 loss（opponent）。这使数据审计从只看规则、loss 或 embedding 相似度，推进到“这条数据实际上怎样推动了训练”。

### 2. 图谱位置

- **前驱 — Day 02 Influence Functions**：Influence Functions 在终点 $\hat\theta$ 附近用 $-g_{z'}^\top H^{-1}g_z$ 近似 upweight / delete-one 的反事实，优点是问题定义清晰并有曲率校正，代价是依赖局部最优、Hessian 逆与 damping。TracIn 不回答同一个反事实；它沿实际优化路径累计贡献，不要求收敛，也不需要 $H^{-1}$。
- **直接对比 Day 02**：若训练集中有十个近重复样本，删除其中一个的局部反事实可能很小，因此 Influence Functions 会认为单个副本不重要；但每个副本都可能在 SGD 中多次提供同向更新，TracIn 会把这些实际推动记录下来。反过来，TracIn 的排名会随训练顺序、checkpoint 与 optimizer 改变，而 Influence Functions 在给定终点与 damping 下更接近“删掉这一点”的问题。
- **后继 — Day 04 LESS**：LESS 继承“目标梯度与候选梯度对齐”的骨架，但把目的从解释既有训练转为**为目标任务主动选 SFT 数据**，加入 warmup、LoRA 梯度、Adam-aware 更新与随机投影；它是 selection policy，不是忠实重放训练轨迹。
- **互补 — Day 05 DataInf**：DataInf 保留 Day 02 的曲率校正思路，用 LoRA 梯度和经验 Fisher 近似来降低求逆成本。TracIn 与 DataInf 不是简单的新旧替代：前者强调 path contribution，后者强调 endpoint counterfactual。

### 3. 机制深挖

#### 3.1 从“真实损失变化”到梯度点积

理想定义只在训练点 $z$ 真正被用于第 $t$ 次更新时记账：

$$\mathrm{TracIn}_{ideal}(z,z')=\sum_{t:z_t=z}\big[\ell(w_t,z')-\ell(w_{t+1},z')\big].$$

它有一个重要守恒性质：对全部训练点求和，恰好等于目标样本从训练开始到结束的总 loss 降幅。若 SGD 更新为 $w_{t+1}=w_t-\eta_t\nabla\ell(w_t,z_t)$，对目标 loss 作一阶展开：

$$\ell(w_t,z')-\ell(w_{t+1},z')\approx \eta_t\nabla\ell(w_t,z')^\top\nabla\ell(w_t,z_t).$$

于是，两条梯度同向时是 proponent，反向时是 opponent；梯度点积同时混合了**方向一致性**与**梯度幅值**。这不是纯语义相似度：表面不同的样本也可能训练同一能力，表面相似的样本也可能因标签/答案冲突而梯度相反。

#### 3.2 TracInCP 做了什么近似

逐 step 保存参数与样本访问记录不可行。TracInCP 只取 $K$ 个 checkpoint：

$$\mathrm{TracInCP}(z,z')=\sum_{k=1}^{K}\eta_k\nabla\ell(w_{t_k},z')^\top\nabla\ell(w_{t_k},z).$$

它把区间内样本真正被访问时的参数替换为 checkpoint 参数，并近似假设相邻 checkpoint 间每个样本访问一次、学习率恒定。论文也给出 counterfactual 解释：好像每条候选数据都在每个 checkpoint 被访问一次；因此甚至可以给未参加原训练的数据打分，但这时不能再说它“实际造成”了训练结果。

checkpoint 不是越晚越好：早期 loss 剧烈波动，一阶近似可能差；收敛后梯度很小，信息也少。论文发现高 loss-decrease 区间比等间隔 checkpoint 更有信息，并且不同 checkpoint 暴露不同类别的错标。正确做法是覆盖稳定下降阶段，而不是机械取“前三个”或只取最终模型。

#### 3.3 Self-influence 到底在测什么

令 $z'=z$：

$$\mathrm{TracInCP}(z,z)=\sum_k\eta_k\|\nabla\ell(w_{t_k},z)\|_2^2\ge 0.$$

所以 self-influence 没有 harmful sign；它主要是在累计样本沿训练路径的梯度能量。错标样本往往长期难拟合、梯度大，因此排名靠前；但罕见而正确的长尾、很长的 sequence、难题或 loss 归一化不当也会同样靠前。它是**人工审计优先级**，不是自动删除判决。

#### 3.4 数据流水线与可扩展实现

1. 训练时保留稳定下降阶段的 checkpoints 与对应学习率；
2. 在每个 checkpoint 对候选训练样本和目标样本计算相同 loss head 的 per-example gradient；
3. 用点积累积 target-conditioned influence，用梯度平方和累积 self-influence；
4. 可只取最后一层/选定层，也可用随机投影保存梯度 sketch；对多目标查询，把各 checkpoint sketch 拼接后放入近邻索引复用；
5. 排名后进入人工、执行测试、provenance 与 slice coverage 复核，不能直接把 top self-influence 当垃圾删除。

论文中的证据边界很具体：CIFAR-10 人工把 10% 标签改成模型最高分的错误类后，TracIn 在检查前 20% 数据时找回超过 80% 错标，而对比方法低于 50%；MNIST 上逐 step 的一阶 loss-change 近似与真实 step loss change 的 Pearson 相关为 0.978。ImageNet 扩展实验只对 ResNet-50 的全连接层取梯度，用第 30/60/90 个 checkpoint 并投影到 1,472 维；它说明方法可扩展做案例分析，但不是全参数 LLM 级验证。

### 4. 边界与反例

1. **路径依赖，不等于因果删点**：换随机种子、数据顺序、学习率、optimizer 或 checkpoint，轨迹与排名都会变。高 TracIn 表示在这次路径上同向推动，不保证删除/添加后重训得到同等幅度的变化。
2. **TracInCP 破坏精确守恒**：理想定义按真实 step 记账并可分解总 loss 降幅；checkpoint 版本把区间压成一个参数点，还假设每点每区间访问一次。重复采样、curriculum、动态混合权重或在线 RL 数据分布都会破坏该近似。
3. **Adam / momentum 不能直接套 SGD 点积**：真实参数更新包含一阶矩、二阶矩、weight decay、裁剪与调度。若仍用 $\eta g_{target}^\top g_{train}$，解释的是原始梯度对齐，不是实际 optimizer update；Day 04 LESS 的 optimizer-aware 设计正是在补这个缺口。
4. **高 self-influence 会误伤“难但对”的数据**：罕见 Rust unsafe bug、长上下文 repo repair、稀有 API migration 都可能因高 loss / 大梯度进入异常榜。若没有执行正确性、来源和语言覆盖约束，过滤会把长尾能力洗掉。
5. **长度与 loss-head 混杂**：token-sum loss 让长答案梯度天然更大；token-mean 又可能掩盖关键错误 token。prompt、reasoning、final answer、tests 若混在一个 loss 中，正负贡献还会相互抵消。
6. **目标集可被污染或过窄**：拿 HumanEval 当唯一 target，会偏向短函数生成并可能奖励近重复泄漏；“帮助 benchmark”不等于帮助真实 coding。目标必须去污染并按 generation、repair、test、语言长尾分 slice。
7. **论文没有证明**：没有证明 TracInCP 是无偏的 delete-one estimator，没有验证全参数 Transformer/LLM、LoRA proxy→大模型迁移、生成式 token-level attribution，也没有证明按分数删数一定提升最终能力。DBPedia、住房与 ImageNet 部分被作者明确定位为应用展示，不是系统评测。

### 5. 迁移到 coding / post-training data

做一个 **50k coding SFT 候选池的“审计优先级”实验**，先验证信号，不直接自动删数：

1. **先做硬门**：沿 Day 01 做 license、secret/PII、exact/MinHash、benchmark contamination 与 sandbox execution；保留 repo、commit、language、task type 元数据。
2. **建立去污染目标集**：从 code generation、repo bug repair、test generation、SQL/Rust 等长尾各取 50–100 条，分 slice 保留，不合并成单个平均目标。
3. **训练 proxy 并选 checkpoint**：用与候选池同分布的 0.5B–1B 模型或 LoRA；从“warmup 后稳定下降、最大 loss-decrease、接近收敛前”取 4–8 个 checkpoint，避开初始震荡和完全收敛尾部。对 LoRA/LM head 梯度作固定 1,024–8,192 维随机投影。
4. **同时算两类分数**：$S_{self}(z)=\sum_k\eta_k\|Pg_z^k\|^2$ 找高梯度异常；$S_s(z)=\sum_k\eta_k\langle Pg_z^k,Pg_{target,s}^k\rangle$ 看对每个能力 slice 的正负影响。另存 gradient norm 与 cosine，避免“只是因为序列长”主导点积。
5. **只 quarantine，不硬删**：优先人工/执行检查“高 self + 多个 slice 稳定负向”的交集，并把错答案、伪通过测试、过时 API、依赖不可复现、benchmark 泄漏、有效长尾分别标注。对 checkpoint / seed 排名不稳定的样本降置信度。
6. **固定 token budget 验证**：比较随机、规则门、规则门+TracIn quarantine、Day 04 LESS targeted selection 四组；报告 clean pass@1、repo repair、test generation、长尾语言、污染命中率和单位 GPU-hour 收益。只有当删数收益跨 seed 超过置信区间且长尾 slice 不退化，才把 TracIn 从审计队列升级为自动 data gate。

### 6. 今天的一道思考题

综合 **Day 02 Influence Functions、Day 03 TracIn、Day 04 LESS**：候选池里一条执行正确的 Rust repo-repair 样本，Influence Functions 在最终 checkpoint 上判断它对 repo-repair target helpful；TracIn 显示它在训练中期强 positive、后期 negative；LESS 因与目标平均梯度 cosine 很高而把它选入 top 5%。与此同时，它的 self-influence 位于 top 0.5%。

请设计一个决策协议，回答保留、降权、分阶段采样还是隔离，并说明：
- 如何用 checkpoint loss-decrease、gradient norm/cosine 分解与 Adam-aware update 区分“后期已学会后的过拟合”与真实负迁移；
- 如何用 Day 02 的局部删点反事实和小规模 leave-one-out 重训校准 Day 03 的路径归因；
- 如何给 LESS 加上 language / repo / task coverage 约束，防止目标平均梯度吞掉 Rust 长尾；
- 哪些跨 seed、跨 target slice、固定 token budget 的证据，才足以把该样本从默认保留改为删除。

## 原文链接
- Paper: https://arxiv.org/abs/2002.08484
- NeurIPS PDF: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
- Official code: https://github.com/frederick0329/TracIn
- GitHub NOTES: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-03-2020-tracin
